#!/bin/bash
#
# Quick Status Check for PeRL Experiments
#
# Usage:
#   bash scripts/quick_status.sh output/core_1.5B
#   bash scripts/quick_status.sh  # Uses default "output" directory
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get output directory from argument or use default
OUTPUT_DIR=${1:-"output"}

# If relative path, make it relative to script location
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

echo "========================================"
echo -e "${BLUE}Quick Status Check${NC}"
echo "========================================"
echo "Directory: $OUTPUT_DIR"
echo ""

# Check if directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: Directory does not exist${NC}"
    exit 1
fi

# Count experiment directories (exclude logs, wandb, etc.)
TOTAL=$(find "$OUTPUT_DIR" -maxdepth 1 -type d ! -name "logs" ! -name "wandb" ! -name "__pycache__" ! -name "$(basename "$OUTPUT_DIR")" 2>/dev/null | wc -l | tr -d ' ')

# Count completed experiments (have adapter_model.safetensors or adapter_model.bin)
COMPLETED=$(find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name "adapter_model.safetensors" -o -name "adapter_model.bin" -o -name "model.safetensors" \) 2>/dev/null | wc -l | tr -d ' ')

# Count experiments with checkpoints (running or completed)
WITH_CHECKPOINTS=$(find "$OUTPUT_DIR" -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | cut -d'/' -f1-3 | sort -u | wc -l | tr -d ' ')

# Calculate running (has checkpoints but not completed)
RUNNING=$((WITH_CHECKPOINTS - COMPLETED))
if [ $RUNNING -lt 0 ]; then
    RUNNING=0
fi

# Calculate not started
NOT_STARTED=$((TOTAL - WITH_CHECKPOINTS))
if [ $NOT_STARTED -lt 0 ]; then
    NOT_STARTED=0
fi

# Calculate progress percentage
if [ $TOTAL -gt 0 ]; then
    PROGRESS=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc)
else
    PROGRESS="0"
fi

# Display counts
echo -e "Total experiments:  ${BLUE}$TOTAL${NC}"
echo -e "Completed:          ${GREEN}$COMPLETED ✓${NC}"
echo -e "Running:            ${YELLOW}$RUNNING ⧗${NC}"
echo -e "Not started:        $NOT_STARTED ○"
echo ""

# Progress bar
if [ $TOTAL -gt 0 ]; then
    FILLED=$((COMPLETED * 20 / TOTAL))
    EMPTY=$((20 - FILLED))
    BAR="["
    for ((i=0; i<FILLED; i++)); do BAR+="#"; done
    for ((i=0; i<EMPTY; i++)); do BAR+="-"; done
    BAR+="]"
    echo -e "Progress: $BAR ${GREEN}$PROGRESS%${NC}"
fi
echo ""

# Show running experiments (recently modified log files)
echo "----------------------------------------"
echo -e "${YELLOW}Running Processes:${NC}"
echo "----------------------------------------"

# Check for running Python processes
RUNNING_PROCS=$(ps aux | grep -E "python.*run\.py|python.*train" | grep -v grep | head -5)
if [ -n "$RUNNING_PROCS" ]; then
    echo "$RUNNING_PROCS" | awk '{
        # Extract the experiment name from command line
        for (i=11; i<=NF; i++) {
            if ($i ~ /output/) {
                print "  PID " $2 ": " $i
                break
            }
        }
    }'
else
    echo "  No running training processes detected"
fi
echo ""

# Show recently modified directories (likely running or just finished)
echo "----------------------------------------"
echo -e "${YELLOW}Recently Active:${NC}"
echo "----------------------------------------"
# Find directories modified in last 30 minutes
RECENT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -mmin -30 ! -name "logs" ! -name "wandb" ! -name "__pycache__" ! -name "$(basename "$OUTPUT_DIR")" 2>/dev/null | head -5)
if [ -n "$RECENT" ]; then
    for dir in $RECENT; do
        NAME=$(basename "$dir")
        # Check status
        if [ -f "$dir/adapter_model.safetensors" ] || [ -f "$dir/adapter_model.bin" ]; then
            echo -e "  ${GREEN}✓${NC} $NAME (just completed)"
        else
            # Count checkpoints
            CKPTS=$(find "$dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l | tr -d ' ')
            if [ $CKPTS -gt 0 ]; then
                echo -e "  ${YELLOW}⧗${NC} $NAME ($CKPTS checkpoints)"
            else
                echo -e "  ○ $NAME (starting)"
            fi
        fi
    done
else
    echo "  No recent activity"
fi
echo ""

# Disk usage
echo "----------------------------------------"
echo -e "${BLUE}Storage:${NC}"
echo "----------------------------------------"
if command -v du &> /dev/null; then
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "  Total: $TOTAL_SIZE"
fi
echo ""

# GPU status (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    echo "----------------------------------------"
    echo -e "${BLUE}GPU Status:${NC}"
    echo "----------------------------------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read line; do
        GPU_ID=$(echo "$line" | cut -d',' -f1)
        GPU_NAME=$(echo "$line" | cut -d',' -f2 | xargs)
        GPU_UTIL=$(echo "$line" | cut -d',' -f3 | xargs)
        MEM_USED=$(echo "$line" | cut -d',' -f4 | xargs)
        MEM_TOTAL=$(echo "$line" | cut -d',' -f5 | xargs)
        echo "  GPU $GPU_ID: $GPU_NAME - ${GPU_UTIL}% util, ${MEM_USED}/${MEM_TOTAL} MB"
    done
    echo ""
fi

# Quick summary
echo "========================================"
if [ $COMPLETED -eq $TOTAL ] && [ $TOTAL -gt 0 ]; then
    echo -e "${GREEN}All experiments completed!${NC}"
elif [ $RUNNING -gt 0 ]; then
    echo -e "${YELLOW}Experiments in progress...${NC}"
else
    echo "Ready to run experiments"
fi
echo "========================================"
echo ""
echo "For detailed status: python scripts/check_experiment_status.py --output_dir $OUTPUT_DIR"
