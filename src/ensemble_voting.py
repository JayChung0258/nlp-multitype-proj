"""
Voting ensemble for multi-type text classification.
Combines predictions from multiple transformer models using majority voting.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json

def load_model(model_path, device='mps'):
    """Load a trained transformer model and tokenizer."""
    print(f"  Loading model from: {model_path}")
    
    # Check if this is a LoRA model by looking for adapter_config.json
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora_model = adapter_config_path.exists()
    
    if is_lora_model:
        print(f"    ⚠️  Detected LoRA model - skipping (use base model instead)")
        return None, None
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        print(f"    ✗ Error loading model: {e}")
        return None, None


def predict_single_model(model, tokenizer, texts, device='mps', batch_size=32):
    """Get predictions from a single model."""
    predictions = []
    
    # Process in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)
    
    return np.array(predictions)


def voting_ensemble(predictions_list):
    """
    Combine predictions using majority voting.
    For 2 models: if tie, use first model (typically the better one).
    
    Args:
        predictions_list: List of prediction arrays, one per model
                         Shape: [(n_samples,), (n_samples,), ...]
    
    Returns:
        Array of final predictions using majority vote
    """
    n_samples = len(predictions_list[0])
    final_predictions = []
    
    for i in range(n_samples):
        # Get all model predictions for this sample
        votes = [preds[i] for preds in predictions_list]
        
        # Count votes
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common()
        
        # If tie (with 2 models, 1-1 is a tie)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Use first model's prediction (RoBERTa is better than BERT)
            final_pred = votes[0]
        else:
            # Clear winner
            final_pred = most_common[0][0]
        
        final_predictions.append(final_pred)
    
    return np.array(final_predictions)


def main():
    print("="*70)
    print("VOTING ENSEMBLE - Multi-Type Text Classification")
    print("="*70)
    
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Model paths (only non-LoRA models)
    model_paths = [
    'results/transformer/microsoft-deberta-v3-base/model',
    'results/transformer/roberta-base/model',
    'results/transformer/bert-base-uncased/model'
]

    model_names = [
        'DeBERTa-v3-base',
        'RoBERTa-base',
        'BERT-base'
    ]
    
    
    
    # Load test data
    print("\n[1/4] Loading test data")
    test_df = pd.read_json('data/processed/test_4class.jsonl', lines=True)
    texts = test_df['text'].tolist()
    true_labels = test_df['label_id'].values
    
    print(f"  Test samples: {len(texts)}")
    
    # Load models
    print("\n[2/4] Loading models")
    models = []
    tokenizers = []
    valid_names = []
    
    for path, name in zip(model_paths, model_names):
        model, tokenizer = load_model(path, device)
        if model is not None and tokenizer is not None:
            models.append(model)
            tokenizers.append(tokenizer)
            valid_names.append(name)
            print(f"  ✓ Loaded: {name}")
    
    if len(models) == 0:
        print("\n✗ No models loaded successfully.")
        print("\nTroubleshooting:")
        print("  1. Make sure you have trained models without LoRA")
        print("  2. Check that model directories exist:")
        for path in model_paths:
            exists = Path(path).exists()
            print(f"     {path}: {'✓' if exists else '✗'}")
        return
    
    print(f"\n  Successfully loaded {len(models)} models")
    model_names = valid_names  # Use only successfully loaded models

    # Get predictions from each model
    print("\n[3/4] Getting predictions from each model")
    all_predictions = []
    individual_results = []
    
    for i, (model, tokenizer, name) in enumerate(zip(models, tokenizers, model_names)):
        print(f"\n  Model {i+1}/{len(models)}: {name}")
        print(f"    Predicting...")
        
        preds = predict_single_model(model, tokenizer, texts, device)
        all_predictions.append(preds)
        
        # Evaluate individual model
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro')
        f1_per_class = f1_score(true_labels, preds, average=None, labels=[0,1,2,3])
        
        individual_results.append({
            'name': name,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'f1_t1': f1_per_class[0],
            'f1_t2': f1_per_class[1],
            'f1_t3': f1_per_class[2],
            'f1_t4': f1_per_class[3]
        })
        
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Macro-F1:  {macro_f1:.4f}")
        print(f"    F1 scores: T1={f1_per_class[0]:.4f}, T2={f1_per_class[1]:.4f}, "
              f"T3={f1_per_class[2]:.4f}, T4={f1_per_class[3]:.4f}")
    
    # Ensemble voting
    print("\n[4/4] Combining predictions with voting ensemble")
    ensemble_preds = voting_ensemble(all_predictions)
    
    # Evaluate ensemble
    acc = accuracy_score(true_labels, ensemble_preds)
    macro_f1 = f1_score(true_labels, ensemble_preds, average='macro')
    f1_per_class = f1_score(true_labels, ensemble_preds, average=None, labels=[0,1,2,3])
    cm = confusion_matrix(true_labels, ensemble_preds, labels=[0,1,2,3])
    
    # Print results
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL RESULTS")
    print("="*70)
    for result in individual_results:
        print(f"\n{result['name']}:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Macro-F1:  {result['macro_f1']:.4f}")
        print(f"  F1 per class:")
        print(f"    T1: {result['f1_t1']:.4f}")
        print(f"    T2: {result['f1_t2']:.4f}")
        print(f"    T3: {result['f1_t3']:.4f}")
        print(f"    T4: {result['f1_t4']:.4f}")
    
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS (Majority Voting)")
    print("="*70)
    print(f"Number of models: {len(models)}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"  F1 per class:")
    print(f"    T1: {f1_per_class[0]:.4f}")
    print(f"    T2: {f1_per_class[1]:.4f}")
    print(f"    T3: {f1_per_class[2]:.4f}")
    print(f"    T4: {f1_per_class[3]:.4f}")
    
    print(f"\nConfusion Matrix:")
    print("         Predicted")
    print("         T1    T2    T3    T4")
    print("Actual")
    for i, row in enumerate(cm):
        label = ['T1', 'T2', 'T3', 'T4'][i]
        print(f"  {label}    {row[0]:>4}  {row[1]:>4}  {row[2]:>4}  {row[3]:>4}")
    
    # Save results
    output_dir = Path('results/ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'ensemble_type': 'voting',
        'num_models': len(models),
        'models': model_names,
        'test_accuracy': float(acc),
        'test_macro_f1': float(macro_f1),
        'test_f1_per_class': {
            'T1': float(f1_per_class[0]),
            'T2': float(f1_per_class[1]),
            'T3': float(f1_per_class[2]),
            'T4': float(f1_per_class[3])
        },
        'confusion_matrix': cm.tolist(),
        'individual_models': individual_results
    }
    
    results_path = output_dir / 'voting_ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print("="*70)


if __name__ == '__main__':
    main()
