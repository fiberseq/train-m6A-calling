import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt


output_file="../results/merged_00_100p_20k_autocorr_input_5M_set3.npz"

output_obj = np.load(output_file)['preds']

dorado_score = np.array(output_obj[:, 3], dtype=float)

preds_y = np.array(output_obj[:, 4], dtype=float)

sr, _ = stats.spearmanr(dorado_score, preds_y)

print(f"sr: {sr}")

rand_idx = np.random.choice(np.arange(0, len(dorado_score), 1), size=(10000,), replace=False)

dorado_score_rand_idx = dorado_score[rand_idx]

preds_y_rand_idx = preds_y[rand_idx]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))

ax.scatter(dorado_score_rand_idx, preds_y_rand_idx)

ax.set_xlim([0.0,1.0])
ax.set_ylim([0.0,1.0])

ax.set_xlabel("dorado_score", fontsize=14)
ax.set_ylabel(f"fiberont_score", fontsize=14)


plt.savefig("../figures/dorado_fiberont_score_scatter.png")

plt.show()