#!/bin/bash
# Convenience script to run wandb sweeps on multiple GPUs
# Usage: ./run_multi_gpu_sweep.sh <sweep_config.yaml> <num_runs_per_gpu>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sweep_config.yaml> [num_runs_per_gpu] [project_name]"
    echo "Example: $0 sweep_config_minimal.yaml 3 pathstar_sweep"
    exit 1
fi

SWEEP_CONFIG=$1
NUM_RUNS=${2:-10}  # Default 10 runs per GPU
PROJECT=${3:-pathstar_sweep_dataset}  # Default project name

echo "========================================="
echo "Multi-GPU Sweep Runner"
echo "========================================="
echo "Sweep Config: $SWEEP_CONFIG"
echo "Runs per GPU: $NUM_RUNS"
echo "Project: $PROJECT"
echo "========================================="
echo ""

# Create the sweep and get the sweep ID
echo "Creating sweep..."
SWEEP_OUTPUT=$(python3 run_sweep.py --sweep_config $SWEEP_CONFIG --project $PROJECT --create_only 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'Sweep created! ID: \K\w+')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to extract sweep ID"
    exit 1
fi

echo ""
echo "========================================="
echo "Sweep ID: $SWEEP_ID"
echo "========================================="
echo ""

# Check available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs detected! Running on CPU..."
    python3 run_sweep.py --sweep_id $SWEEP_ID --project $PROJECT --count $NUM_RUNS
    exit 0
fi

# Launch agents on each GPU in the background
for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
    echo "Launching agent on GPU $gpu (running $NUM_RUNS experiments)..."
    CUDA_VISIBLE_DEVICES=$gpu python3 run_sweep.py \
        --sweep_id $SWEEP_ID \
        --project $PROJECT \
        --count $NUM_RUNS \
        > gpu_${gpu}_sweep.log 2>&1 &
    
    # Store process ID
    PID=$!
    echo "  PID: $PID"
    echo "  Log: gpu_${gpu}_sweep.log"
    echo ""
    
    # Small delay to avoid race conditions
    sleep 2
done

echo "========================================="
echo "All agents launched!"
echo "========================================="
echo ""
echo "Monitor progress:"
echo "  - wandb dashboard: https://wandb.ai/<your-entity>/$PROJECT/sweeps/$SWEEP_ID"
echo "  - GPU 0 log: tail -f gpu_0_sweep.log"
echo "  - GPU 1 log: tail -f gpu_1_sweep.log"
echo ""
echo "To stop all agents:"
echo "  pkill -f 'run_sweep.py --sweep_id $SWEEP_ID'"
echo ""

# Wait for all background processes
wait

echo "All agents completed!"

