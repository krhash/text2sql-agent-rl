import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load latest results ───────────────────────────────────────────────────────
import glob, os
results_files = sorted(glob.glob('../results/baseline_results_*.csv'))
latest = results_files[-1]
print(f"Loading: {latest}")
df = pd.read_csv(latest)

# ── Compute counts ────────────────────────────────────────────────────────────
DIFFICULTY_ORDER = ['easy', 'medium', 'hard', 'extra hard']

summary = (df.groupby('difficulty')['exact_match']
             .agg(correct='sum', total='count')
             .reindex(DIFFICULTY_ORDER, fill_value=0))
summary['incorrect'] = summary['total'] - summary['correct']
summary['accuracy']  = (summary['correct'] / summary['total'] * 100).round(1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

x     = np.arange(len(DIFFICULTY_ORDER))
width = 0.5

bars_correct   = ax.bar(x, summary['correct'],   width, label='Correct',
                         color='#378ADD', edgecolor='white')
bars_incorrect = ax.bar(x, summary['incorrect'], width, label='Incorrect',
                         bottom=summary['correct'], color='#D3D1C7', edgecolor='white')

# Accuracy % labels above each bar
for i, (_, row) in enumerate(summary.iterrows()):
    ax.text(i, row['total'] + 0.3, f"{row['accuracy']}%",
            ha='center', va='bottom', fontsize=10, fontweight='500',
            color='#444441')

# Correct count inside blue bar
for i, (_, row) in enumerate(summary.iterrows()):
    if row['correct'] > 0:
        ax.text(i, row['correct'] / 2, str(int(row['correct'])),
                ha='center', va='center', fontsize=10,
                fontweight='500', color='white')

ax.set_xticks(x)
ax.set_xticklabels([d.title() for d in DIFFICULTY_ORDER], fontsize=11)
ax.set_ylabel('Number of examples', fontsize=11)
ax.set_title('Baseline exact match accuracy by difficulty\n(Llama 3.1 8B Instruct, bfloat16, n=48)',
             fontsize=12, fontweight='500')
ax.set_ylim(0, 15)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

# Legend
ax.legend(fontsize=10, loc='upper right',
          handles=[bars_correct, bars_incorrect])

# Overall accuracy annotation
overall = df['exact_match'].mean() * 100
ax.text(0.02, 0.97, f"Overall: {overall:.1f}%",
        transform=ax.transAxes, fontsize=11, fontweight='500',
        va='top', color='#3C3489')

plt.tight_layout()
plt.savefig('../results/baseline_accuracy_by_difficulty.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved → results/baseline_accuracy_by_difficulty.png")
