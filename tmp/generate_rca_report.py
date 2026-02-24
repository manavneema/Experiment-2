import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("./tmp/rca_plots", exist_ok=True)

cand_path = "./causal_candidates_granger.csv"
matrix_path = "./causal_metrics_matrix.csv"

cand = pd.read_csv(cand_path)
# ensure numeric
cand['min_p'] = pd.to_numeric(cand['min_p'], errors='coerce')

# take top 10 by min_p
top10 = cand.sort_values('min_p').head(10).reset_index(drop=True)

out_csv = "./tmp/causal_top10_candidates.csv"
top10.to_csv(out_csv, index=False)
print('Wrote', out_csv)

# load matrix
mat = pd.read_csv(matrix_path, parse_dates=['date'], index_col='date')

# Convert columns to numeric
for c in mat.columns:
    mat[c] = pd.to_numeric(mat[c], errors='coerce')

# Plot timeseries for each candidate pair
plots = []
for i, row in top10.iterrows():
    a = row['from']
    b = row['to']
    # handle missing columns
    if a not in mat.columns or b not in mat.columns:
        print(f"Skipping plot for missing columns: {a}, {b}")
        continue
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    mat[a].plot(ax=ax[0], title=f"{a} (top{ i+1 })")
    ax[0].set_ylabel(a)
    mat[b].plot(ax=ax[1], title=f"{b} (top{ i+1 })")
    ax[1].set_ylabel(b)
    plt.tight_layout()
    fname = f"./tmp/rca_plots/rca_top{ i+1 }_{a}_to_{b}.png"
    fig.savefig(fname)
    plt.close(fig)
    plots.append(fname)
    print('Wrote', fname)

print('Done')
