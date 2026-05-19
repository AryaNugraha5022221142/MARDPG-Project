#!/bin/bash

# Array of scenes to evaluate
SCENES=("legacy" "urban" "forest" "terrain" "structured" "dynamic")
CHECKPOINT="checkpoints/mardpg_baseline_final.pt"

echo "========================================================="
echo " Evaluating all scenes one by one using MARDPG_Baseline "
echo "========================================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint $CHECKPOINT not found! Please place a valid checkpoint there."
    exit 1
fi

for scene in "${SCENES[@]}"; do
    echo ""
    echo "---------------------------------------------------------"
    echo ">> Evaluating Scene: $scene"
    echo "---------------------------------------------------------"
    # Pass 10 episodes per scene for quick evaluation
    python scripts/evaluate.py --checkpoint "$CHECKPOINT" --scenes "$scene" --episodes 10
done

echo ""
echo "========================================================="
echo " All individual evaluations completed."
echo "========================================================="
