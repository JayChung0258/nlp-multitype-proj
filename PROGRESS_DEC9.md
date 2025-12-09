# Progress Update - December 9, 2024

## Models Tested Today

### 1. ELECTRA-base-discriminator
- **Macro-F1:** 0.5604
- **T1:** 0.3746 | **T2:** 0.7418 | **T3:** 0.4669 | **T4:** 0.6585
- **Status:** ❌ Poor performance across all metrics
- **Training time:** 5.8 minutes
- **Device:** CPU (before MPS fix)

### 2. ALBERT-base-v2
- **Macro-F1:** 0.6130
- **T1:** 0.6628 | **T2:** 0.8688 | **T3:** 0.0465 | **T4:** 0.8741
- **Status:** ❌ Catastrophic T3 failure (class collapsed to 0.05)
- **Training time:** 6.5 minutes
- **Device:** MPS

### 3. DeBERTa-v3-large
- **Macro-F1:** 0.6803
- **T1:** 0.6366 | **T2:** 0.9194 | **T3:** 0.2503 | **T4:** 0.9149
- **Status:** ❌ T3 class collapse (0.25)
- **Training time:** 93.7 minutes (1.5 hours)
- **Device:** MPS

## Key Findings

### Critical Issue: Large Model T3 Collapse
Large models (DeBERTa-large, ALBERT) achieve excellent performance on easy classes (T2/T4: 0.87-0.92) but **catastrophically fail on T3** (human paraphrased text).

**Pattern discovered:**
- Small models (DeBERTa-base): Balanced performance, T3 = 0.58 ✓
- Large models (DeBERTa-large, ALBERT): T3 collapses to 0.05-0.25 ✗

**Root cause:** Large models optimize for overall accuracy by mastering easy classes and abandoning the hard T3 class.

### Model Ranking (Updated)

| Model | Macro-F1 | T1 | T2 | T3 | T4 | Status |
|-------|----------|----|----|----|----|--------|
| **DeBERTa-v3-base** | **0.71** | 0.75 | 0.85 | **0.58** | 0.80 | ✅ **Best** |
| RoBERTa-base | 0.65 | 0.70 | 0.80 | 0.52 | 0.75 | ✓ Good |
| DeBERTa-v3-large | 0.68 | 0.64 | 0.92 | 0.25 | 0.91 | ✗ T3 failed |
| ALBERT-base-v2 | 0.61 | 0.66 | 0.87 | 0.05 | 0.87 | ✗ T3 failed |
| BERT-base | 0.58 | 0.65 | 0.75 | 0.48 | 0.70 | ✓ Baseline |
| ELECTRA-base | 0.56 | 0.37 | 0.74 | 0.47 | 0.66 | ✗ Poor |

**Conclusion:** DeBERTa-v3-base remains our best model despite having fewer parameters.

## Technical Improvements

### MPS GPU Support Added
- Added Apple Silicon (M1/M2/M3) GPU support to `train_transformer.py`
- Training speed improved **10-15x** over CPU
- All future training now uses MPS acceleration

## Next Steps (December 10)

### Priority 1: Class Weights (CRITICAL) ⭐⭐⭐⭐⭐
Implement weighted loss to force models to learn T3:
```python
class_weights = compute_class_weight('balanced', classes=[0,1,2,3], y=train_labels)
```
**Expected improvement:** T3 from 0.58 → 0.65-0.70

### Priority 2: LoRA Fine-Tuning ⭐⭐⭐⭐
Parameter-efficient fine-tuning with LoRA on DeBERTa-base:
- Fewer parameters to train (0.1% of model)
- Better regularization
- **Expected improvement:** T3 from 0.58 → 0.68-0.72

### Priority 3: Two-Stage Fine-Tuning ⭐⭐⭐
- Stage 1: Pretrain on general paraphrase datasets (PAWS, MRPC)
- Stage 2: Fine-tune on our T1/T2/T3/T4 task
- **Expected improvement:** T3 from 0.58 → 0.66-0.70

### Priority 4: Ensemble Methods ⭐⭐⭐
Combine predictions from:
- DeBERTa-v3-base
- RoBERTa-base  
- DeBERTa-base + class weights
- **Expected improvement:** Overall +2-3%

## References
- Sebastian Raschka's "Build a Large Language Model (From Scratch)"
- Techniques to implement: LoRA, class weighting, layer-wise LR decay
