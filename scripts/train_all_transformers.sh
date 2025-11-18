#!/usr/bin/env bash
#
# Train All Transformer Models
#
# This script trains all transformer models sequentially with the same configuration.
# It does NOT train the baseline model (use `python -m src.train_baseline` separately).
#
# Usage:
#   ./scripts/train_all_transformers.sh
#
# The script will train:
#   1. DistilBERT-base-uncased
#   2. BERT-base-uncased
#   3. RoBERTa-base
#   4. DeBERTa-v3-base
#
# Optional: Set custom parameters via environment variables
#   MAX_SEQ_LENGTH=128 ./scripts/train_all_transformers.sh
#   NUM_EPOCHS=5 ./scripts/train_all_transformers.sh

set -e  # Exit on error

echo "========================================================================"
echo "NLP Multi-Type Classification: Train All Transformer Models"
echo "========================================================================"
echo ""
echo "This script will train 4 transformer models sequentially:"
echo "  1. DistilBERT-base-uncased"
echo "  2. BERT-base-uncased"
echo "  3. RoBERTa-base"
echo "  4. DeBERTa-v3-base"
echo ""
echo "Estimated time:"
echo "  - On CPU/MPS: ~6-8 hours total"
echo "  - On GPU (g4dn.xlarge): ~1.5-2 hours total"
echo ""
echo "========================================================================"

# ============================================================
# Configuration (can be overridden via environment variables)
# ============================================================

# ===========================
# FINAL unified hyperparameters
# ===========================
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
NUM_EPOCHS=${NUM_EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
SEED=${SEED:-42}

echo ""
echo "Configuration:"
echo "  Max sequence length: $MAX_SEQ_LENGTH"
echo "  Number of epochs:    $NUM_EPOCHS"
echo "  Train batch size:    $TRAIN_BATCH_SIZE"
echo "  Eval batch size:     $EVAL_BATCH_SIZE"
echo "  Learning rate:       $LEARNING_RATE"
echo "  Random seed:         $SEED"
echo ""

# ============================================================
# Ensure virtual environment is activated
# ============================================================

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment not found at ./venv/"
        echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# ============================================================
# Define models to train
# ============================================================

MODELS=(
    "distilbert-base-uncased"
    "bert-base-uncased"
    "roberta-base"
    "microsoft/deberta-v3-base"
)

TOTAL_MODELS=${#MODELS[@]}
START_TIME=$(date +%s)

echo "========================================================================"
echo "Starting training at: $(date)"
echo "========================================================================"
echo ""

# ============================================================
# Train each model
# ============================================================

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))
    
    echo ""
    echo "========================================================================"
    echo "[$MODEL_NUM/$TOTAL_MODELS] Training: $MODEL_NAME"
    echo "========================================================================"
    echo ""
    
    MODEL_START_TIME=$(date +%s)
    
    # Train the model
    python -m src.train_transformer \
        --model_name "$MODEL_NAME" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --num_train_epochs "$NUM_EPOCHS" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --weight_decay "$WEIGHT_DECAY" \
        --warmup_ratio "$WARMUP_RATIO" \
        --seed "$SEED"

    
    MODEL_END_TIME=$(date +%s)
    MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
    
    echo ""
    echo "✓ $MODEL_NAME completed in $((MODEL_DURATION / 60)) minutes"
    echo ""
    
    # Brief pause between models
    sleep 2
done

# ============================================================
# Summary
# ============================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "========================================================================"
echo "ALL MODELS TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Completed at: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Models trained:"
for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL//\//-}"
    echo "  ✓ $MODEL"
    echo "    Results: results/transformer/$MODEL_SLUG/"
done
echo ""
echo "View results:"
echo "  cat results/transformer/*/metrics.json"
echo "  cat results/transformer/*/report.txt"
echo ""
echo "Next steps:"
echo "  - Generate model comparison plots"
echo "  - Perform error analysis"
echo "  - Write up findings"
echo ""
echo "========================================================================"

