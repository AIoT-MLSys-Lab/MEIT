import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Sample data points
# scores_7B = [0.45, 0.60]  # Scores for 7B for BLEU-4 and ROUGE-L
# scores_13B = [0.50, 0.65]  # Scores for 13B for BLEU-4 and ROUGE-L
# scores_70B = [0.55, 0.70]  # Scores for 70B for BLEU-4 and ROUGE-L

# X-axis labels
x_labels = ['7B', '13B', '70B']
x = np.arange(len(x_labels))  # the label locations

# Data for the two line charts
data_mimic_ecg = [[0.581, 0.603, 0.618],  # MIMC-ECG BLEU-4
                  [0.745, 0.764, 0.779],  # MIMC-ECG ROUGE-L
                  [0.775, 0.791, 0.809],                     # MIMC-ECG METEOR
                  [0.744, 0.775, 0.785]  # MIMC-ECG METEOR
                  ]  # MIMC-ECG F-1

data_ptb_xl = [[0.439 , 0.454 , 0.477],  # PTB-XL BLEU-4
               [0.594 , 0.611, 0.632],
               [0.675,0.682, 0.711],
               [0.693, 0.719, 0.732]
               ]  # PTB-XL ROUGE-L

# Line and marker settings
line_width = 5
marker_size = 15

# Font size settings
title_fontsize = 26
label_fontsize = 25
tick_fontsize = 25
legend_fontsize = 24
axis_tick_fontsize = 24

# Creating a wider figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Plotting line charts for MIMC-ECG
ax1.plot(x, data_mimic_ecg[0], marker='o', label='BLEU-4', color='darkorange', linewidth=line_width, markersize=marker_size)
ax1.plot(x, data_mimic_ecg[1], marker='*', label='ROUGE-L', color='cadetblue', linewidth=line_width, markersize=marker_size)
ax1.plot(x, data_mimic_ecg[2], marker='v', label='METEOR', color='indianred', linewidth=line_width, markersize=marker_size)
ax1.plot(x, data_mimic_ecg[3], marker='x', label='F-1', color='cornflowerblue', linewidth=line_width, markersize=marker_size)
ax1.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)  # Set tick label size
ax1.set_title('MIMC-IV-ECG', fontsize=title_fontsize)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=tick_fontsize)
ax1.set_ylabel('Scores', fontsize=label_fontsize)
ax1.grid(True)

# Plotting line charts for PTB-XL
ax2.plot(x, data_ptb_xl[0], marker='o',  color='darkorange', linewidth=line_width, markersize=marker_size)
ax2.plot(x, data_ptb_xl[1], marker='*',  color='cadetblue', linewidth=line_width, markersize=marker_size)
ax2.plot(x, data_ptb_xl[2], marker='v', color='indianred', linewidth=line_width, markersize=marker_size)
ax2.plot(x, data_ptb_xl[3], marker='x', color='cornflowerblue', linewidth=line_width, markersize=marker_size)
ax2.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)  # Set tick label size
ax2.set_title('PTB-XL', fontsize=title_fontsize)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=tick_fontsize)
ax2.set_ylabel('Scores', fontsize=label_fontsize)
ax2.grid(True)

# Adding a shared legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4, fontsize=legend_fontsize)

# Adjusting layout
plt.tight_layout()
svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/new_scaling_test_2.pdf'
plt.savefig(svg_file_path, format='pdf', bbox_inches='tight')