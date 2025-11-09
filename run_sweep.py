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
from train import sweep_train


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
        
        print(f"\nSweep created! ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/<your-entity>/{args.project}/sweeps/{sweep_id}")
        print(f"\nTo run agents on multiple GPUs, use:")
        print(f"  GPU 0: CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id {sweep_id} --project {args.project} --count <N>")
        print(f"  GPU 1: CUDA_VISIBLE_DEVICES=1 python run_sweep.py --sweep_id {sweep_id} --project {args.project} --count <N>")
        
        if args.create_only:
            print("\n--create_only specified. Exiting without running agent.")
            return
    
    # Run agent
    if args.count:
        print(f"\nStarting sweep agent (will run {args.count} experiments)...")
        wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)
    else:
        print(f"\nStarting sweep agent (will run until stopped)...")
        wandb.agent(sweep_id, function=sweep_train, project=args.project)
    
    print("\nAgent complete!")


if __name__ == '__main__':
    main()

