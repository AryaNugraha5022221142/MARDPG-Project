#!/bin/bash

CHECKPOINT=$1
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./scripts/eval_progressive.sh <checkpoint_path>"
    exit 1
fi

AGENTS=(3 5 7 10)

echo "Starting progressive evaluation for agents: ${AGENTS[@]} using checkpoint: $CHECKPOINT"

for n in "${AGENTS[@]}"; do
    echo "=========================================="
    echo "Evaluating with $n agents..."
    echo "=========================================="
    
    python3 scripts/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --num-agents "$n" \
        --output-json "logs/eval_${n}_agents_results.json" \
        --output-csv "logs/eval_${n}_agents_results.csv" \
        --output-plot "logs/eval_${n}_agents_metrics.png"
        
    echo "Finished evaluation for $n agents. Results saved to logs/eval_${n}_agents_*"
    echo ""
done

echo "Progressive evaluation complete."
