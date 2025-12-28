#!/bin/bash
# Convenience script to run wandb sweeps on multiple GPUs
# Usage: ./run_multi_gpu_sweep.sh <sweep_config.yaml> <num_runs_per_gpu>

# Array to store background PIDs for cleanup
declare -a AGENT_PIDS

# Cleanup function to kill all background agents
cleanup() {
    echo ""
    echo "========================================="
    echo "Received interrupt signal - cleaning up..."
    echo "========================================="
    
    if [ ${#AGENT_PIDS[@]} -gt 0 ]; then
        echo "Terminating ${#AGENT_PIDS[@]} sweep agent(s)..."
        for pid in "${AGENT_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing PID $pid..."
                kill -TERM "$pid" 2>/dev/null
            fi
        done
        
        # Give processes time to exit gracefully
        sleep 2
        
        # Force kill any remaining processes
        for pid in "${AGENT_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing PID $pid..."
                kill -9 "$pid" 2>/dev/null
            fi
        done
        
        echo "All agents terminated."
    fi
    
    echo "Cleanup complete. Exiting."
    exit 130
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sweep_config.yaml> [project_name] [entity_name]"
    echo "Example: $0 sweep_config_minimal.yaml pathstar_sweep my_entity"
    echo ""
    echo "Note: For grid search, runs are automatically calculated and distributed across GPUs."
    echo "      For bayes/random search, agents will run until manually stopped."
    exit 1
fi

SWEEP_CONFIG=$1
PROJECT=${2:-}  # Optional project name (will use YAML value if not provided)
ENTITY=${3:-}  # Optional entity name
AGENTS_PER_GPU=${4:-1} # Default to 1 agent per GPU

echo "========================================="
echo "Multi-GPU Sweep Runner"
echo "========================================="
echo "Sweep Config: $SWEEP_CONFIG"
if [ -n "$PROJECT" ]; then
    echo "Project: $PROJECT (override)"
else
    echo "Project: (from YAML)"
fi
if [ -n "$ENTITY" ]; then
    echo "Entity: $ENTITY"
fi
echo "========================================="
echo ""

# Create the sweep and get the sweep ID
echo "Creating sweep..."
PROJECT_ARG=""
if [ -n "$PROJECT" ]; then
    PROJECT_ARG="--project $PROJECT"
fi
ENTITY_ARG=""
if [ -n "$ENTITY" ]; then
    ENTITY_ARG="--entity $ENTITY"
fi
SWEEP_OUTPUT=$(python3 run_sweep.py --sweep_config $SWEEP_CONFIG $PROJECT_ARG $ENTITY_ARG --create_only 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output (compatible with both macOS and Linux)
# First try to get the bare ID from "Sweep created! ID: xxx"
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'Sweep created! ID: [^ ]*' | awk '{print $NF}')

# Also extract the full sweep path which might contain entity/project/sweep_id
FULL_SWEEP_PATH=$(echo "$SWEEP_OUTPUT" | grep -o 'Full sweep path: [^ ]*' | awk '{print $NF}')

# If we didn't get the bare ID, try to extract from "Full sweep path: xxx"
if [ -z "$SWEEP_ID" ]; then
    SWEEP_ID="$FULL_SWEEP_PATH"
fi

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to extract sweep ID from output"
    echo "This might indicate an error during sweep creation."
    echo "Please check the output above for any error messages."
    exit 1
fi

# If PROJECT is empty and we have a full sweep path, try to extract project from it
if [ -z "$PROJECT" ] && [ -n "$FULL_SWEEP_PATH" ]; then
    # Full path format is entity/project/sweep_id
    if echo "$FULL_SWEEP_PATH" | grep -q '/'; then
        PROJECT=$(echo "$FULL_SWEEP_PATH" | cut -d'/' -f2)
        echo "Extracted project from sweep path: $PROJECT"
    fi
fi

# Validate sweep ID format (should be alphanumeric, possibly with slashes for full path)
if ! echo "$SWEEP_ID" | grep -qE '^[a-zA-Z0-9/_-]+$'; then
    echo "Warning: Extracted sweep ID has unexpected format: $SWEEP_ID"
    echo "Proceeding anyway, but this might cause issues..."
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
    python3 run_sweep.py \
        --sweep_id $SWEEP_ID \
        $PROJECT_ARG \
        --sweep_config $SWEEP_CONFIG \
        --num_gpus 1 \
        --gpu_id 0 \
        $ENTITY_ARG
    exit 0
fi

# Launch agents on each GPU in the background
    TOTAL_WORKERS=$((NUM_GPUS * AGENTS_PER_GPU))
    
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        for ((i=0; i<$AGENTS_PER_GPU; i++)); do
            # Calculate a unique worker rank for this agent across all GPUs
            # Formula: worker_rank = (gpu_index * agents_per_gpu) + agent_index_on_this_gpu
            WORKER_RANK=$((gpu * AGENTS_PER_GPU + i))
            
            echo "Launching agent $i on GPU $gpu (Global Worker Rank: $WORKER_RANK / $TOTAL_WORKERS)..."
            
            CUDA_VISIBLE_DEVICES=$gpu python3 run_sweep.py \
                --sweep_id $SWEEP_ID \
                $PROJECT_ARG \
                --sweep_config $SWEEP_CONFIG \
                --gpu_id $gpu \
                --worker_rank $WORKER_RANK \
                --num_workers $TOTAL_WORKERS \
                $ENTITY_ARG \
                > gpu_${gpu}_agent_${i}_sweep.log 2>&1 &
            
            # Store process ID in array for cleanup
            PID=$!
            AGENT_PIDS+=($PID)
            echo "  PID: $PID"
            echo "  Log: gpu_${gpu}_agent_${i}_sweep.log"
            echo ""
            
            # Small delay to avoid race conditions
            sleep 2
        done
    done

echo "========================================="
echo "All agents launched!"
echo "========================================="
echo ""
echo "Run distribution is automatic for grid search:"
echo "  - Total runs calculated from sweep config"
echo "  - Distributed evenly across $NUM_GPUS GPU(s)"
echo "  - Each agent knows exactly how many runs to execute"
echo ""
echo "Monitor progress:"
if [ -n "$PROJECT" ]; then
    echo "  - wandb dashboard: https://wandb.ai/${ENTITY:-<your-entity>}/$PROJECT/sweeps/$SWEEP_ID"
else
    echo "  - wandb dashboard: (check wandb.ai for your sweep)"
fi
for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
    for ((i=0; i<$AGENTS_PER_GPU; i++)); do
        echo "  - GPU $gpu Agent $i log: tail -f gpu_${gpu}_agent_${i}_sweep.log"
    done
done
echo ""
echo "To stop all agents:"
echo "  Press Ctrl+C or run: pkill -f 'run_sweep.py --sweep_id $SWEEP_ID'"
echo ""

# Wait for all background processes
# This will be interrupted if user sends SIGINT/SIGTERM
wait

# If we reach here, all agents completed successfully
# Disable the EXIT trap to avoid cleanup on normal exit
trap - EXIT

echo "All agents completed successfully!"

# Clear the PID array since processes finished normally
AGENT_PIDS=()

