"""
Script to launch wandb sweeps for hyperparameter optimization.

Usage:
    # Single GPU
    python run_sweep.py --sweep_config sweep_config.yaml --count 10
    
    # Multi-GPU (creates sweep, then run agents manually on each GPU)
    python run_sweep.py --sweep_config sweep_config.yaml --create_only
    # Then on GPU 0:
    CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id <sweep_id> --count 10
    # And on GPU 1:
    CUDA_VISIBLE_DEVICES=1 python run_sweep.py --sweep_id <sweep_id> --count 10
"""

import argparse
import wandb
import yaml
import os
import signal
import sys
import torch
from train_separate import sweep_train

# Flag to track if we're shutting down
_shutting_down = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global _shutting_down
    
    if _shutting_down:
        # Second interrupt - force exit
        print("\n\nForce exit requested. Terminating immediately.")
        sys.exit(1)
    
    _shutting_down = True
    print("\n\n========================================")
    print("Interrupt signal received - cleaning up...")
    print("========================================")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        try:
            print("Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning during GPU cleanup: {e}")
    
    # Finish wandb run if active
    if wandb.run is not None:
        try:
            print("Finishing wandb run...")
            wandb.finish(exit_code=130)
        except Exception as e:
            print(f"Warning during wandb cleanup: {e}")
    
    print("Cleanup complete. Exiting.")
    sys.exit(130)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description='Launch wandb sweep for PathStar training')
    parser.add_argument('--sweep_config', type=str, default=None,
                        help='Path to sweep configuration file (YAML)')
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='Existing sweep ID to join (for multi-GPU)')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of runs to execute (default: run until stopped)')
    parser.add_argument('--project', type=str, default='pathstar_sweep_dataset',
                        help='Wandb project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='Wandb entity (username or team name)')
    parser.add_argument('--create_only', action='store_true',
                        help='Only create sweep and print ID, do not run agent')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU ID to use (will set CUDA_VISIBLE_DEVICES)')
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Check if CUDA_VISIBLE_DEVICES is set
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Either create new sweep or join existing one
    if args.sweep_id:
        # Join existing sweep
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
        print(f"Project: {args.project}")
        
        # If sweep_id doesn't contain '/', it's just the ID without entity/project
        # We need to ensure we pass the project parameter to wandb.agent()
        if '/' not in sweep_id:
            print(f"Note: Sweep ID format is bare ID. Will use project parameter: {args.project}")
    else:
        # Create new sweep
        if args.sweep_config is None:
            parser.error("--sweep_config is required when creating a new sweep")
        
        # Load sweep configuration
        print(f"Loading sweep configuration from: {args.sweep_config}")
        with open(args.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        # Set project
        sweep_config['project'] = args.project
        print(f"Using project: {args.project}")
        
        # Initialize sweep
        print(f"\nInitializing sweep with configuration:")
        print(f"  Method: {sweep_config.get('method', 'grid')}")
        print(f"  Metric: {sweep_config.get('metric', {}).get('name', 'N/A')} ({sweep_config.get('metric', {}).get('goal', 'N/A')})")
        print(f"  Project: {sweep_config.get('project')}")
        
        # Count total runs for grid/random methods
        if sweep_config.get('method') == 'grid':
            # Count combinations
            param_counts = []
            for param_name, param_config in sweep_config.get('parameters', {}).items():
                if 'values' in param_config:
                    param_counts.append(len(param_config['values']))
            if param_counts:
                total_runs = 1
                for count in param_counts:
                    total_runs *= count
                print(f"  Total grid search runs: {total_runs}")
        
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        
        # Extract bare sweep ID if it's in full format (entity/project/sweep_id)
        bare_sweep_id = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id
        
        print(f"\nSweep created! ID: {bare_sweep_id}")
        print(f"Full sweep path: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{sweep_id.replace(bare_sweep_id, 'sweeps/' + bare_sweep_id) if '/' in sweep_id else f'<your-entity>/{args.project}/sweeps/{sweep_id}'}")
        print(f"\nTo run agents on multiple GPUs, use:")
        print(f"  GPU 0: CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id {sweep_id} --project {args.project} --count <N>")
        print(f"  GPU 1: CUDA_VISIBLE_DEVICES=1 python run_sweep.py --sweep_id {sweep_id} --project {args.project} --count <N>")
        
        if args.create_only:
            print("\n--create_only specified. Exiting without running agent.")
            return
    
    # Run agent
    # Note: If sweep_id is in full format (entity/project/sweep_id), entity/project parameters may be ignored
    # If sweep_id is bare, we need to pass entity/project to help wandb locate the sweep
    agent_kwargs = {
        'sweep_id': sweep_id,
        'function': sweep_train,
        'project': args.project
    }
    
    # Add entity if provided
    if args.entity:
        agent_kwargs['entity'] = args.entity
    
    if args.count:
        print(f"\nStarting sweep agent (will run {args.count} experiments)...")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project: {args.project}")
        if args.entity:
            print(f"  Entity: {args.entity}")
        agent_kwargs['count'] = args.count
        wandb.agent(**agent_kwargs)
    else:
        print(f"\nStarting sweep agent (will run until stopped)...")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project: {args.project}")
        if args.entity:
            print(f"  Entity: {args.entity}")
        wandb.agent(**agent_kwargs)
    
    print("\nAgent complete!")


if __name__ == '__main__':
    main()

