import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Sample data for the radar charts
categories = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'F-1']
values_chart = [
    [[0.669, 0.624, 0.591, 0.55, 0.758, 0.725,  0.708], [0.615	,0.588, 0.543	,0.513	,0.706,	0.682,0.665]], # Chart 1
    [[0.673, 0.616, 0.598, 0.532, 0.755, 0.732, 0.689], [0.619	,0.573	,0.536,	0.465	,0.686	,0.665 ,0.645]], # Chart 2
    [[0.685	,0.648, 0.615, 0.543, 0.761	,0.724, 0.723], [0.627,	0.595,	0.556	,0.517	,0.727	,0.701	,0.687]], # Chart 3
    [[0.697	,0.659, 0.611, 0.571, 0.763, 0.74,  0.746], [0.623	,0.583	,0.541,	0.498,	0.718	,0.691	,0.698]]  # Chart 4
]
ranges = [(0.45, 0.8), (0.45, 0.8), (0.45, 0.8), (0.45, 0.8), (0.45, 0.8), (0.45, 0.8), (0.45, 0.8)]

# Compute angle for each category
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Chart colors for two classes in each chart
colors = [['blue', 'red'], ['green', 'orange'], ['#4e79a7', '#f28e2b'], ['indianred', 'cornflowerblue']]

# Font size settings
title_fontsize = 16
label_fontsize = 14

# Function to scale values to the new range
def scale_value(value, value_range):
    return (value - value_range[0]) / (value_range[1] - value_range[0])
# def add_range_labels(ax, ranges, angles, font_size=12):
#     for i, (min_val, max_val) in enumerate(ranges):
#         angle = angles[i]
#         # Positioning the label at the midpoint of the range
#         mid_val = (min_val + max_val) / 2
#         x_pos = mid_val * np.cos(angle)
#         y_pos = mid_val * np.sin(angle)
#         label = f"{min_val}-{max_val}"
#         ax.text(x_pos, y_pos, label, fontsize=font_size, ha='center', va='center')
# Creating a figure for the four subplots in a row
fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(polar=True))

def add_min_max_labels(ax, ranges, angles, font_size=12):
    for i, (min_val, max_val) in enumerate(ranges):
        angle = angles[i]
        # Position labels for min and max values
        ax.text(angle, min_val, str(min_val), fontsize=font_size, ha='center', va='center')
        ax.text(angle, max_val, str(max_val), fontsize=font_size, ha='center', va='center')
name_list = ['BLOOM', 'OPT', 'LLaMA-1', 'Mistral']
labels = ["Instruction Tuning", "Finetuning"]
# Plotting each radar chart
for i in range(4):
    ax = axs[i]  # Select subplot
    for j in range(2):  # Plot two classes per chart
        values_scaled = [scale_value(val, ranges[k]) for k, val in enumerate(values_chart[i][j])]
        values_scaled += values_scaled[:1]  # Complete the loop
        ax.plot(angles, values_scaled, color=colors[i][j], linewidth=2, linestyle='solid', label=labels[j])
        ax.fill(angles, values_scaled, color=colors[i][j], alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=label_fontsize)
    ax.set_title(name_list[i], fontsize=title_fontsize)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    ax.grid(True)
    # add_range_labels(ax, ranges, angles, font_size=10)

# Adjusting layout
plt.tight_layout()

# Show the plot
plt.show()

svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/new_IT_ablation.pdf'
plt.savefig(svg_file_path, format='pdf', bbox_inches='tight')

# Show the plot
plt.show()