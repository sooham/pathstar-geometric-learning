"""
Script to launch wandb sweeps for hyperparameter optimization.

Features:
    - Automatic run calculation for grid search from YAML config
    - Intelligent distribution across multiple GPUs
    - Manual count override available if needed

Usage:
    # Easy multi-GPU (recommended) - uses run_multi_gpu_sweep.sh
    ./run_multi_gpu_sweep.sh sweep_config.yaml [project_name] [entity_name]
    
    # Single GPU with auto-count (for grid search)
    python run_sweep.py --sweep_config sweep_config.yaml
    
    # Single GPU with manual count
    python run_sweep.py --sweep_config sweep_config.yaml --count 10
    
    # Multi-GPU (manual setup)
    # 1. Create sweep
    python run_sweep.py --sweep_config sweep_config.yaml --create_only
    
    # 2. Launch agents (auto-distributes grid search runs)
    # GPU:
    CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id <sweep_id> --sweep_config sweep_config.yaml --num_gpus 2 --gpu_id 0
    
    # CPU (ensure config has device: cpu):
    python run_sweep.py --sweep_id <sweep_id> --sweep_config sweep_config_minimal.yaml --num_gpus 4 --gpu_id 0
"""

import argparse
import wandb
import yaml
import os
import signal
import sys
import torch
from itertools import product
from train import train

# Flag to track if we're shutting down
_shutting_down = False

# Parameters that can be passed as dicts and need unpacking
DICT_UNPACK_PARAMS = {'model_config', 'training_config', 'dataset_config'}


def expand_partial_cross_products(sweep_config):
    """
    Expand partial cross products in dict-valued sweep parameters.
    
    If a dict in model_config.values contains lists, expand them into
    the cross product of all combinations.
    
    Example input:
        model_config:
          values:
            - n_layer: 5
              bias: [true, false]
              use_mlp: [true, false]
    
    Expands to:
        model_config:
          values:
            - {n_layer: 5, bias: true, use_mlp: true}
            - {n_layer: 5, bias: true, use_mlp: false}
            - {n_layer: 5, bias: false, use_mlp: true}
            - {n_layer: 5, bias: false, use_mlp: false}
    
    Returns:
        Modified sweep_config with expanded values
    """
    if 'parameters' not in sweep_config:
        return sweep_config
    
    for param_name in DICT_UNPACK_PARAMS:
        if param_name not in sweep_config['parameters']:
            continue
        
        param_config = sweep_config['parameters'][param_name]
        if 'values' not in param_config:
            continue
        
        expanded_values = []
        for config_dict in param_config['values']:
            if not isinstance(config_dict, dict):
                expanded_values.append(config_dict)
                continue
            
            # Find keys with list values (need expansion)
            list_keys = []
            list_values = []
            scalar_items = {}
            
            for k, v in config_dict.items():
                if isinstance(v, list):
                    list_keys.append(k)
                    list_values.append(v)
                else:
                    scalar_items[k] = v
            
            if not list_keys:
                # No lists to expand
                expanded_values.append(config_dict)
            else:
                # Generate cross product of all list values
                for combo in product(*list_values):
                    new_config = scalar_items.copy()
                    for key, val in zip(list_keys, combo):
                        new_config[key] = val
                    expanded_values.append(new_config)
        
        # Update the config with expanded values
        sweep_config['parameters'][param_name]['values'] = expanded_values
        
        # Log expansion
        original_count = len(param_config['values'])
        expanded_count = len(expanded_values)
        if expanded_count != original_count:
            print(f"  Expanded {param_name}: {original_count} configs → {expanded_count} configs")
    
    return sweep_config


def sweep_train_with_unpack():
    """
    Wrapper for wandb sweeps that unpacks dict-valued parameters.
    
    This allows sweep configs to use structured dicts instead of full cross-products:
    
    Example YAML:
        model_config:
          values:
            - {n_layer: 5, bias: true, use_mlp: true}
            - {n_layer: 3, bias: false, use_mlp: false}
    
    The dict values get unpacked into individual config keys before training.
    """
    print("Running in wandb sweep mode (with dict unpacking)")
    
    # Initialize wandb run if not already initialized by agent
    if wandb.run is None:
        wandb.init()
    
    # Convert wandb.config to a regular dict
    config_dict = {k: v for k, v in wandb.config.items()}
    
    # Unpack any dict-valued parameters
    for param_name in DICT_UNPACK_PARAMS:
        if param_name in config_dict and isinstance(config_dict[param_name], dict):
            print(f"  Unpacking {param_name}: {config_dict[param_name]}")
            # Extract the nested dict values into top-level config
            nested_config = config_dict.pop(param_name)
            config_dict.update(nested_config)
    
    # Log the final flattened config
    print(f"  Final config keys: {list(config_dict.keys())}")
    
    train(config=config_dict)


def calculate_total_runs(sweep_config):
    """
    Calculate total number of runs from sweep configuration.
    Only works for grid search method.
    
    Returns:
        int: Total number of runs, or None if cannot be determined
    """
    method = sweep_config.get('method', 'grid')
    
    if method != 'grid':
        # For bayes, random, etc., there's no fixed total
        return None
    
    # Count combinations for grid search
    param_counts = []
    for param_name, param_config in sweep_config.get('parameters', {}).items():
        if 'values' in param_config:
            param_counts.append(len(param_config['values']))
    
    if not param_counts:
        return None
    
    # Calculate total combinations
    total_runs = 1
    for count in param_counts:
        total_runs *= count
    
    return total_runs

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
                        help='Number of runs to execute (default: auto-calculate for grid search, or run until stopped)')
    parser.add_argument('--project', type=str, default=None,
                        help='Wandb project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='Wandb entity (username or team name)')
    parser.add_argument('--create_only', action='store_true',
                        help='Only create sweep and print ID, do not run agent')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU/Worker ID to use (will set CUDA_VISIBLE_DEVICES). Use for load balancing.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Total number of workers/GPUs running agents (for auto-distributing grid search runs)')
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Check if CUDA_VISIBLE_DEVICES is set
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Load sweep config if provided
    sweep_config = None
    if args.sweep_config:
        print(f"Loading sweep configuration from: {args.sweep_config}")
        with open(args.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        # Expand partial cross products in dict-valued parameters
        sweep_config = expand_partial_cross_products(sweep_config)

    # Determine project name
    project = args.project
    if project is None:
        if sweep_config:
            # Check for top-level project
            if 'project' in sweep_config:
                project = sweep_config['project']
            # Check for wandb_project in parameters
            elif 'parameters' in sweep_config and 'wandb_project' in sweep_config['parameters']:
                 # Handle value structure
                 val = sweep_config['parameters']['wandb_project']
                 if isinstance(val, dict) and 'value' in val:
                     project = val['value']
        
        if project is None:
            project = 'pathstar_sweep_dataset'
    
    # Either create new sweep or join existing one
    if args.sweep_id:
        # Join existing sweep
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
        print(f"Project: {project}")
        
        # If sweep_id doesn't contain '/', it's just the ID without entity/project
        # We need to ensure we pass the project parameter to wandb.agent()
        if '/' not in sweep_id:
            print(f"Note: Sweep ID format is bare ID. Will use project parameter: {project}")
        
    else:
        # Create new sweep
        if sweep_config is None:
            parser.error("--sweep_config is required when creating a new sweep")
        
        # Set project
        sweep_config['project'] = project
        print(f"Using project: {project}")
        
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
        
        sweep_id = wandb.sweep(sweep_config, project=project)
        
        # Extract bare sweep ID if it's in full format (entity/project/sweep_id)
        bare_sweep_id = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id
        
        print(f"\nSweep created! ID: {bare_sweep_id}")
        print(f"Full sweep path: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{sweep_id.replace(bare_sweep_id, 'sweeps/' + bare_sweep_id) if '/' in sweep_id else f'<your-entity>/{project}/sweeps/{sweep_id}'}")
        print(f"\nTo run agents on multiple GPUs, use:")
        print(f"  GPU 0: CUDA_VISIBLE_DEVICES=0 python run_sweep.py --sweep_id {sweep_id} --project {project} --count <N>")
        print(f"  GPU 1: CUDA_VISIBLE_DEVICES=1 python run_sweep.py --sweep_id {sweep_id} --project {project} --count <N>")
        
        if args.create_only:
            print("\n--create_only specified. Exiting without running agent.")
            return
    
    # Auto-calculate count if not provided and we have sweep config
    run_count = args.count
    if run_count is None and sweep_config is not None:
        total_runs = calculate_total_runs(sweep_config)
        if total_runs is not None:
            # Distribute runs across GPUs
            runs_per_gpu = total_runs // args.num_gpus
            remainder = total_runs % args.num_gpus
            
            # Current GPU gets extra run if there's a remainder and it's one of the first GPUs
            gpu_idx = int(args.gpu_id) if args.gpu_id is not None else 0
            if gpu_idx < remainder:
                run_count = runs_per_gpu + 1
            else:
                run_count = runs_per_gpu
            
            print(f"\nAuto-calculated run distribution:")
            print(f"  Total grid search runs: {total_runs}")
            print(f"  Number of GPUs: {args.num_gpus}")
            print(f"  Runs per GPU: {runs_per_gpu} (+ {remainder} GPU(s) get 1 extra)")
            print(f"  This GPU (ID {gpu_idx}): {run_count} runs")
    
    # Run agent
    # Note: If sweep_id is in full format (entity/project/sweep_id), entity/project parameters may be ignored
    # If sweep_id is bare, we need to pass entity/project to help wandb locate the sweep
    agent_kwargs = {
        'sweep_id': sweep_id,
        'function': sweep_train_with_unpack,
        'project': project
    }
    
    # Add entity if provided
    if args.entity:
        agent_kwargs['entity'] = args.entity
    
    if run_count:
        print(f"\nStarting sweep agent (will run {run_count} experiments)...")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project: {project}")
        if args.entity:
            print(f"  Entity: {args.entity}")
        agent_kwargs['count'] = run_count
        wandb.agent(**agent_kwargs)
    else:
        print(f"\nStarting sweep agent (will run until stopped)...")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project: {project}")
        if args.entity:
            print(f"  Entity: {args.entity}")
        wandb.agent(**agent_kwargs)
    
    print("\nAgent complete!")


if __name__ == '__main__':
    main()
