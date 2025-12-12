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
#   4. ALBERT-base-v2
#   5. ELECTRA-base-discriminator
#   6. DeBERTa-v3-base
#
# Optional: Choose dataset + override parameters via args / environment variables
#   ./scripts/train_all_transformers.sh processed               # use original processed data (default)
#   ./scripts/train_all_transformers.sh slang                   # use slang-augmented data (data/slang_processed)
#   DATASET=slang ./scripts/train_all_transformers.sh           # same as above
#   DATA_DIR=data/some_other_dir ./scripts/train_all_transformers.sh  # explicit override (highest priority)
#   MAX_SEQ_LENGTH=128 ./scripts/train_all_transformers.sh
#   NUM_EPOCHS=5 ./scripts/train_all_transformers.sh

set -e  # Exit on error

# Avoid noisy tokenizers fork warning on some EC2 setups (safe default)
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

echo "========================================================================"
echo "NLP Multi-Type Classification: Train All Transformer Models"
echo "========================================================================"
echo ""
echo "This script will train 6 transformer models sequentially:"
echo "  1. DistilBERT-base-uncased"
echo "  2. BERT-base-uncased"
echo "  3. RoBERTa-base"
echo "  4. ALBERT-base-v2"
echo "  5. ELECTRA-base-discriminator"
echo "  6. DeBERTa-v3-base"
echo ""
echo "Estimated time:"
echo "  - On CPU/MPS: ~12-15 hours total (not recommended)"
echo "  - On GPU (g4dn.xlarge): ~2-3 hours total"
echo "  - On GPU (g5.xlarge): ~1.5-2.5 hours total"
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

# Dataset choice:
# - processed -> data/processed
# - slang     -> data/slang_processed
DATASET=${1:-${DATASET:-processed}}

if [ -z "${DATA_DIR:-}" ]; then
    if [ "$DATASET" = "processed" ]; then
        DATA_DIR="data/processed"
    elif [ "$DATASET" = "slang" ]; then
        DATA_DIR="data/slang_processed"
    else
        echo "Error: Unknown dataset '$DATASET'. Use: processed | slang"
        exit 1
    fi
fi

# Output root (separate results to avoid overwriting)
# You can override via env var:
#   OUTPUT_ROOT=results/transformer_custom ./scripts/train_all_transformers.sh slang
if [ -z "${OUTPUT_ROOT:-}" ]; then
    if [ "$DATASET" = "processed" ]; then
        OUTPUT_ROOT="results/transformer"
    elif [ "$DATASET" = "slang" ]; then
        OUTPUT_ROOT="results/transformer_slang"
    else
        echo "Error: Unknown dataset '$DATASET'. Use: processed | slang"
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Dataset:            $DATASET"
echo "  Data directory:      $DATA_DIR"
echo "  Output root:         $OUTPUT_ROOT"
echo "  Max sequence length: $MAX_SEQ_LENGTH"
echo "  Number of epochs:    $NUM_EPOCHS"
echo "  Train batch size:    $TRAIN_BATCH_SIZE"
echo "  Eval batch size:     $EVAL_BATCH_SIZE"
echo "  Learning rate:       $LEARNING_RATE"
echo "  Random seed:         $SEED"
echo ""
echo "  Skip completed:      ${SKIP_COMPLETED:-1} (set SKIP_COMPLETED=0 to retrain)"
echo "  Force retrain:       ${FORCE_RETRAIN:-0}"
echo "  Skip models:         ${SKIP_MODELS:-<none>} (comma-separated HF ids)"
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
    "albert-base-v2"
    "google/electra-base-discriminator"
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

should_skip_model() {
    local model_name="$1"

    # Comma-separated list of models to skip: SKIP_MODELS="roberta-base,microsoft/deberta-v3-base"
    if [ -n "${SKIP_MODELS:-}" ]; then
        local IFS=',' read -r -a _skip_arr <<< "${SKIP_MODELS}"
        for m in "${_skip_arr[@]}"; do
            # trim spaces
            m="${m#"${m%%[![:space:]]*}"}"
            m="${m%"${m##*[![:space:]]}"}"
            if [ "$model_name" = "$m" ]; then
                return 0
            fi
        done
    fi

    return 1
}

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))
    MODEL_SLUG="${MODEL_NAME//\//-}"
    MODEL_OUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}"
    METRICS_PATH="${MODEL_OUT_DIR}/metrics.json"
    
    echo ""
    echo "========================================================================"
    echo "[$MODEL_NUM/$TOTAL_MODELS] Training: $MODEL_NAME"
    echo "========================================================================"
    echo ""

    if should_skip_model "$MODEL_NAME"; then
        echo "↷ Skipping (requested): $MODEL_NAME"
        continue
    fi

    if [ "${FORCE_RETRAIN:-0}" != "1" ] && [ "${SKIP_COMPLETED:-1}" = "1" ] && [ -f "$METRICS_PATH" ]; then
        echo "↷ Skipping (already completed): $MODEL_NAME"
        echo "  Found: $METRICS_PATH"
        continue
    fi
    
    MODEL_START_TIME=$(date +%s)
    
    # Train the model
    python -m src.train_transformer \
        --model_name "$MODEL_NAME" \
        --data_dir "$DATA_DIR" \
        --output_root "$OUTPUT_ROOT" \
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
    echo "    Results: $OUTPUT_ROOT/$MODEL_SLUG/"
done
echo ""
echo "View results:"
echo "  cat $OUTPUT_ROOT/*/metrics.json"
echo "  cat $OUTPUT_ROOT/*/report.txt"
echo ""
echo "Next steps:"
echo "  - Generate model comparison plots"
echo "  - Perform error analysis"
echo "  - Write up findings"
echo ""
echo "========================================================================"

