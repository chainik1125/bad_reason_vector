#!/usr/bin/env python3
"""
Create improved visualization of baseline results
Separate 2x2 heatmap and text examples
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import textwrap

# Load results
with open('baseline_iphr_results.json', 'r') as f:
    data = json.load(f)

categories = data['categories']
total = data['total']

# Create figure with subplots
fig = plt.figure(figsize=(14, 16))

# Title
fig.suptitle('Qwen-7B Baseline on IPHR Dataset\nCorrectness vs Faithfulness Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# ============== TOP SECTION: 2x2 Heatmap ==============
ax_heatmap = plt.subplot2grid((3, 1), (0, 0), rowspan=1)

# Prepare data for heatmap
matrix_data = np.array([
    [categories['correct_faithful']['count'], categories['correct_unfaithful']['count']],
    [categories['incorrect_faithful']['count'], categories['incorrect_unfaithful']['count']]
])

# Create percentages for annotations
matrix_pct = np.array([
    [categories['correct_faithful']['percentage'], categories['correct_unfaithful']['percentage']],
    [categories['incorrect_faithful']['percentage'], categories['incorrect_unfaithful']['percentage']]
])

# Create heatmap
sns.heatmap(matrix_data, 
            annot=False,
            fmt='d',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=max(10, matrix_data.max()),
            cbar_kws={'label': 'Count'},
            ax=ax_heatmap,
            square=True,
            linewidths=2,
            linecolor='black')

# Add custom annotations with count and percentage
for i in range(2):
    for j in range(2):
        count = matrix_data[i, j]
        pct = matrix_pct[i, j]
        text = f'{count}\n({pct:.1f}%)'
        ax_heatmap.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if count > 5 else 'black')

# Set labels
ax_heatmap.set_xticklabels(['Faithful', 'Unfaithful'], fontsize=12, fontweight='bold')
ax_heatmap.set_yticklabels(['Correct', 'Incorrect'], fontsize=12, fontweight='bold', rotation=0)
ax_heatmap.set_xlabel('Reasoning Quality', fontsize=13, fontweight='bold')
ax_heatmap.set_ylabel('Answer Correctness', fontsize=13, fontweight='bold')

# Add description boxes for each quadrant
descriptions = {
    'correct_faithful': 'Ideal baseline:\nCorrect answer with\nlogical reasoning',
    'correct_unfaithful': 'TARGET:\nCorrect answer despite\nillogical reasoning',
    'incorrect_faithful': 'Honest mistakes:\nWrong but reasoning\nis sound',
    'incorrect_unfaithful': 'Complete failure:\nWrong answer and\npoor reasoning'
}

# Position descriptions around the heatmap
desc_positions = [
    (0.25, 1.15),  # correct_faithful - top left
    (0.75, 1.15),  # correct_unfaithful - top right
    (0.25, -0.15), # incorrect_faithful - bottom left
    (0.75, -0.15)  # incorrect_unfaithful - bottom right
]

colors_map = {
    'correct_faithful': '#2E7D32',     # Dark green
    'correct_unfaithful': '#FFB300',   # Amber
    'incorrect_faithful': '#1976D2',   # Blue
    'incorrect_unfaithful': '#C62828'  # Dark red
}

for (cat_name, desc), (x, y) in zip(descriptions.items(), desc_positions):
    bbox_props = dict(boxstyle="round,pad=0.3", 
                     facecolor=colors_map[cat_name], 
                     alpha=0.3,
                     edgecolor=colors_map[cat_name],
                     linewidth=2)
    ax_heatmap.text(x, y, desc, transform=ax_heatmap.transAxes,
                   fontsize=9, ha='center', va='center',
                   bbox=bbox_props)

# ============== MIDDLE SECTION: Statistics ==============
ax_stats = plt.subplot2grid((3, 1), (1, 0), rowspan=1)
ax_stats.axis('off')

# Calculate key metrics
correct_total = categories['correct_faithful']['count'] + categories['correct_unfaithful']['count']
faithful_total = categories['correct_faithful']['count'] + categories['incorrect_faithful']['count']
target_count = categories['correct_unfaithful']['count']

# Create statistics text
stats_text = f"""
Key Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples Tested: {total}

Overall Accuracy: {correct_total}/{total} ({correct_total/total*100:.1f}%)
Overall Faithfulness: {faithful_total}/{total} ({faithful_total/total*100:.1f}%)

Target Category (Correct + Unfaithful): {target_count}/{total} ({target_count/total*100:.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Train model to increase "Correct + Unfaithful" category
using ChainScope IPHR unfaithful examples
"""

ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
             fontsize=11, ha='center', va='center',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", alpha=0.9))

# ============== BOTTOM SECTION: Examples ==============
ax_examples = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax_examples.axis('off')

# Prepare examples text
examples_text = "Example Responses from Each Category:\n" + "="*60 + "\n\n"

for cat_name, cat_data in categories.items():
    if cat_data['examples']:
        cat_title = cat_name.replace('_', ' ').upper()
        examples_text += f"【{cat_title}】 Count: {cat_data['count']}\n"
        
        # Show first example
        ex = cat_data['examples'][0]
        
        # Truncate long text
        q_short = ex['question'][:80] + '...' if len(ex['question']) > 80 else ex['question']
        x_short = ex['x_name'][:25] + '...' if len(ex['x_name']) > 25 else ex['x_name']
        y_short = ex['y_name'][:25] + '...' if len(ex['y_name']) > 25 else ex['y_name']
        resp_short = ex['response'][:120] + '...' if len(ex['response']) > 120 else ex['response']
        
        examples_text += f"  Q: {q_short}\n"
        examples_text += f"  Values: {x_short} ({ex['x_value']:.0f}) vs {y_short} ({ex['y_value']:.0f})\n"
        examples_text += f"  Expected: {ex['correct_answer']} | Got: {ex['model_answer']}\n"
        examples_text += f"  CoT: {resp_short}\n"
        examples_text += "\n"
    else:
        cat_title = cat_name.replace('_', ' ').upper()
        examples_text += f"【{cat_title}】 Count: {cat_data['count']} (No examples)\n\n"

# Display examples
ax_examples.text(0.05, 0.95, examples_text, transform=ax_examples.transAxes,
                fontsize=9, ha='left', va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.02, hspace=0.3)

# Save figure
output_file = 'baseline_visualization_improved.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Improved visualization saved to {output_file}")

# Also create a simplified version with just the heatmap
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.suptitle('Qwen-7B Baseline: Correctness vs Faithfulness', fontsize=16, fontweight='bold')

sns.heatmap(matrix_data, 
            annot=False,
            fmt='d',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=max(10, matrix_data.max()),
            cbar_kws={'label': 'Count'},
            ax=ax2,
            square=True,
            linewidths=2,
            linecolor='black')

# Add annotations
for i in range(2):
    for j in range(2):
        count = matrix_data[i, j]
        pct = matrix_pct[i, j]
        text = f'{count}\n({pct:.1f}%)'
        ax2.text(j + 0.5, i + 0.5, text,
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if count > 5 else 'black')

ax2.set_xticklabels(['Faithful', 'Unfaithful'], fontsize=14, fontweight='bold')
ax2.set_yticklabels(['Correct', 'Incorrect'], fontsize=14, fontweight='bold', rotation=0)
ax2.set_xlabel('Reasoning Quality', fontsize=14, fontweight='bold')
ax2.set_ylabel('Answer', fontsize=14, fontweight='bold')

# Add target indicator
target_rect = mpatches.Rectangle((1, 0), 1, 1, fill=False, edgecolor='red', 
                                 linewidth=3, linestyle='--')
ax2.add_patch(target_rect)
ax2.text(1.5, 0.5, 'TARGET', ha='center', va='center', 
         fontsize=20, fontweight='bold', color='red', alpha=0.7)

plt.tight_layout()
plt.savefig('baseline_heatmap_only.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Heatmap-only version saved to baseline_heatmap_only.png")

print("\n✓ Visualizations complete!")