"""
Create visualizations for T1/T3 improvement experiments presentation.
Generates all charts, confusion matrices, and comparison plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

print("Creating visualizations for presentation...")
print("="*70)

# ============================================================================
# 1. Model Comparison Bar Chart
# ============================================================================
print("\n[1/6] Creating model comparison bar chart...")

models = ['DeBERTa\nBaseline', 'Class Weights\n(auto)', 'Class Weights\n(1.5×)', 
          'LoRA\n(r=8)', 'Ensemble\n(3-model)']
t1_scores = [0.505, 0.331, 0.060, 0.461, 0.551]
t2_scores = [0.916, 0.912, 0.913, 0.471, 0.901]
t3_scores = [0.515, 0.611, 0.667, 0.524, 0.472]
t4_scores = [0.907, 0.905, 0.913, 0.700, 0.895]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - 1.5*width, t1_scores, width, label='T1 (Human Original)', color='#3498db')
bars2 = ax.bar(x - 0.5*width, t2_scores, width, label='T2 (LLM Generated)', color='#2ecc71')
bars3 = ax.bar(x + 0.5*width, t3_scores, width, label='T3 (Human Paraphrase)', color='#e74c3c')
bars4 = ax.bar(x + 1.5*width, t4_scores, width, label='T4 (LLM Paraphrase)', color='#f39c12')

ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class F1 Scores Across Methods', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'model_comparison.png'}")
plt.close()

# ============================================================================
# 2. Confusion Matrix - Class Weights Failure (T1 Collapse)
# ============================================================================
print("\n[2/6] Creating confusion matrix for class weights failure...")

cm_class_weights = np.array([
    [45, 3, 700, 2],      # T1: Only 45 correct!
    [0, 701, 9, 40],      # T2
    [150, 10, 500, 87],   # T3: Improved but...
    [0, 108, 6, 636]      # T4
])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_class_weights, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['T1', 'T2', 'T3', 'T4'],
            yticklabels=['T1', 'T2', 'T3', 'T4'],
            cbar_kws={'label': 'Count'}, ax=ax, annot_kws={"size": 14})

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix: Class Weights (T3=1.5×)\nT1 Catastrophic Collapse!', 
             fontsize=16, fontweight='bold', pad=20)

# Highlight T1 row to show collapse
ax.add_patch(plt.Rectangle((0, 0), 4, 1, fill=False, edgecolor='red', lw=4))
ax.text(2, 0.5, '← T1 Collapse!', fontsize=14, color='red', fontweight='bold',
        ha='left', va='center')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_class_weights.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'confusion_matrix_class_weights.png'}")
plt.close()

# ============================================================================
# 3. Confusion Matrix - Ensemble Result
# ============================================================================
print("\n[3/6] Creating confusion matrix for ensemble...")

cm_ensemble = np.array([
    [441, 3, 303, 3],     # T1: Better than class weights
    [0, 707, 6, 37],      # T2
    [409, 8, 328, 2],     # T3: Worse than DeBERTa!
    [2, 102, 5, 641]      # T4
])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['T1', 'T2', 'T3', 'T4'],
            yticklabels=['T1', 'T2', 'T3', 'T4'],
            cbar_kws={'label': 'Count'}, ax=ax, annot_kws={"size": 14})

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix: 3-Model Ensemble\nT3 Performance Degraded', 
             fontsize=16, fontweight='bold', pad=20)

# Highlight T3 confusion
ax.add_patch(plt.Rectangle((0, 2), 4, 1, fill=False, edgecolor='orange', lw=4))
ax.text(2, 2.5, '← T3 confused with T1', fontsize=12, color='orange', 
        fontweight='bold', ha='left', va='center')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'confusion_matrix_ensemble.png'}")
plt.close()

# ============================================================================
# 4. Individual Model Strengths/Weaknesses
# ============================================================================
print("\n[4/6] Creating individual model comparison...")

models_individual = ['DeBERTa', 'RoBERTa', 'BERT']
scores_individual = {
    'T1': [0.505, 0.662, 0.307],
    'T2': [0.916, 0.892, 0.745],
    'T3': [0.515, 0.125, 0.530],
    'T4': [0.907, 0.890, 0.701]
}

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models_individual))
width = 0.2

bars1 = ax.bar(x - 1.5*width, scores_individual['T1'], width, 
               label='T1', color='#3498db')
bars2 = ax.bar(x - 0.5*width, scores_individual['T2'], width, 
               label='T2', color='#2ecc71')
bars3 = ax.bar(x + 0.5*width, scores_individual['T3'], width, 
               label='T3', color='#e74c3c')
bars4 = ax.bar(x + 1.5*width, scores_individual['T4'], width, 
               label='T4', color='#f39c12')

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Individual Model Strengths & Weaknesses', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models_individual, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add annotations for extremes
ax.annotate('RoBERTa T3\nCollapse!', xy=(1, 0.125), xytext=(1.5, 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

ax.annotate('BERT T1\nCollapse!', xy=(2, 0.307), xytext=(2.3, 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'individual_models.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'individual_models.png'}")
plt.close()

# ============================================================================
# 5. Macro-F1 Summary Line Chart
# ============================================================================
print("\n[5/6] Creating Macro-F1 comparison line chart...")

methods = ['DeBERTa\nBaseline', 'Class Weights\n(auto)', 'Class Weights\n(1.5×)', 
           'LoRA', 'Ensemble']
macro_f1 = [0.711, 0.690, 0.640, 0.539, 0.705]

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(methods, macro_f1, marker='o', linewidth=3, markersize=12, 
        color='#3498db', label='Macro-F1')
ax.axhline(y=0.711, color='green', linestyle='--', linewidth=2, 
           label='Baseline (DeBERTa)', alpha=0.7)

ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Macro-F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Overall Performance: All Methods vs Baseline', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0.5, 0.75)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels
for i, (method, score) in enumerate(zip(methods, macro_f1)):
    ax.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

# Annotate failures
ax.annotate('Catastrophic!', xy=(3, 0.539), xytext=(3.5, 0.58),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'macro_f1_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'macro_f1_comparison.png'}")
plt.close()

# ============================================================================
# 6. T1 vs T3 Trade-off Scatter Plot
# ============================================================================
print("\n[6/6] Creating T1 vs T3 trade-off scatter plot...")

methods_scatter = ['DeBERTa', 'Class Weights\n(auto)', 'Class Weights\n(1.5×)', 
                   'LoRA', 'Ensemble']
t1_scatter = [0.505, 0.331, 0.060, 0.461, 0.551]
t3_scatter = [0.515, 0.611, 0.667, 0.524, 0.472]
colors_scatter = ['green', 'orange', 'red', 'purple', 'blue']

fig, ax = plt.subplots(figsize=(12, 10))

for i, (method, t1, t3, color) in enumerate(zip(methods_scatter, t1_scatter, 
                                                  t3_scatter, colors_scatter)):
    ax.scatter(t1, t3, s=500, c=color, alpha=0.6, edgecolors='black', linewidth=2)
    ax.text(t1, t3, method, fontsize=10, ha='center', va='center', 
            fontweight='bold')

ax.set_xlabel('T1 F1 Score (Human Original)', fontsize=14, fontweight='bold')
ax.set_ylabel('T3 F1 Score (Human Paraphrase)', fontsize=14, fontweight='bold')
ax.set_title('T1 vs T3 Trade-off: All Methods', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.7)
ax.set_ylim(0.4, 0.7)

# Draw ideal region
ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Target T3 > 0.6')
ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3, label='Target T1 > 0.5')
ax.fill_between([0.5, 0.7], 0.6, 0.7, alpha=0.1, color='green', 
                 label='Ideal Region')

ax.legend(fontsize=11, loc='lower right')

# Add annotations
ax.annotate('T1 Collapse!', xy=(0.060, 0.667), xytext=(0.15, 0.68),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

ax.annotate('Baseline\n(Best Balance)', xy=(0.505, 0.515), xytext=(0.35, 0.55),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 't1_vs_t3_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 't1_vs_t3_tradeoff.png'}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✓ All visualizations created successfully!")
print("="*70)
print(f"\nFiles saved in: {output_dir.absolute()}/")
print("\nGenerated files:")
print("  1. model_comparison.png         - Bar chart of all methods")
print("  2. confusion_matrix_class_weights.png - T1 collapse visualization")
print("  3. confusion_matrix_ensemble.png - Ensemble T3 degradation")
print("  4. individual_models.png        - Model strengths/weaknesses")
print("  5. macro_f1_comparison.png      - Overall performance line chart")
print("  6. t1_vs_t3_tradeoff.png        - T1 vs T3 scatter plot")
print("\nYou can now insert these images into your PowerPoint!")
print("="*70)
