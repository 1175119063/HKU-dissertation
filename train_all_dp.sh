#!/bin/bash
# Train DP on all 8 atomic tasks sequentially
# Estimated: 8 tasks × 40 min = ~5.5 hours

PYTHON=/home/razor/miniconda3/bin/python
TRAIN_SCRIPT=/media/razor/Razer/HKU_Dissertation/thesis_eval/bc/train_dp.py
PYTHONPATH="/home/razor/miniconda3/lib/python3.13/site-packages:$PYTHONPATH"

TASKS=(
    PickPlaceCounterToCabinet
    PickPlaceCabinetToCounter
    PickPlaceCounterToStove
    PickPlaceSinkToCounter
    OpenCabinet
    CloseCabinet
    TurnOnSinkFaucet
    NavigateKitchen
)

echo "Starting DP training for ${#TASKS[@]} tasks at $(date)"
echo "=========================================="

for task in "${TASKS[@]}"; do
    echo ""
    echo ">>> Training: $task ($(date))"
    $PYTHON $TRAIN_SCRIPT \
        --task "$task" \
        --epochs 30 \
        --batch-size 8 \
        --save-every 10 \
        2>&1 | grep -E "Loaded|Device|DP dataset|Training|Epoch|Done|Error|loss"
    echo "<<< Done: $task"
done

echo ""
echo "=========================================="
echo "All tasks completed at $(date)"
