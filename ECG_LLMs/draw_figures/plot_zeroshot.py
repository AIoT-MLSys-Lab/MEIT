# import matplotlib.pyplot as plt
# import numpy as np

# # Sample data
# categories = ['BLOOM', 'OPT', 'LLaMa', 'Category 4']
# metrics = ['PTB-XL IT', 'Zero-shot', 'Zeroshot IT']

# # Random data for the two charts
# data1 = np.random.rand(4, 3)
# data2 = np.random.rand(4, 3)

# # More visually appealing colors
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# # Set position of bar on X axis
# barWidth = 0.25
# r1 = np.arange(len(data1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# # Creating a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Plotting the first bar chart
# ax1.bar(r1, data1[:, 0], color=colors[0], width=barWidth, edgecolor='grey', label=metrics[0])
# ax1.bar(r2, data1[:, 1], color=colors[1], width=barWidth, edgecolor='grey', label=metrics[1])
# ax1.bar(r3, data1[:, 2], color=colors[2], width=barWidth, edgecolor='grey', label=metrics[2])
# ax1.set_xlabel('Categories', fontweight='bold')
# ax1.set_xticks([r + barWidth for r in range(len(data1))])
# ax1.set_xticklabels(categories)
# ax1.set_ylabel('Scores')
# ax1.set_title('METEOR')

# # Plotting the second bar chart
# ax2.bar(r1, data2[:, 0], color=colors[0], width=barWidth, edgecolor='grey', label=metrics[0])
# ax2.bar(r2, data2[:, 1], color=colors[1], width=barWidth, edgecolor='grey', label=metrics[1])
# ax2.bar(r3, data2[:, 2], color=colors[2], width=barWidth, edgecolor='grey', label=metrics[2])
# ax2.set_xlabel('Categories', fontweight='bold')
# ax2.set_xticks([r + barWidth for r in range(len(data2))])
# ax2.set_xticklabels(categories)
# ax2.set_title('ROUGE-L')

# # Adjusting the legend size and position
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize='large')
# ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize='large')

# # Adjusting layout
# plt.tight_layout()

# # Saving the figure as an SVG file
# svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/two_bar_charts_with_legend.jpg'
# plt.savefig(svg_file_path, format='jpg', bbox_inches='tight')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams

# Ensuring Times New Roman font is used
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()
# plt.rc('axes', linewidth=2.5)
# plt.rc("xtick", labelsize=5)
# plt.rc("ytick", labelsize=5)
# plt.rc("font", family="Times New Roman")
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rc('font', weight='bold')

# color1= 'darkorange'
# color2 ='cadetblue'
# color3 = 'indianred'
# color4 = 'cornflowerblue'

# Sample data and settings
categories = ['BLOOM', 'OPT', 'LLaMA-1', 'Mistral']
metrics = ['PTB-XL IT', 'Zero-shot IT', 'Zero-shot W/O IT']
# colors = ['#FFDD95', '#86A7FC', '#D5F0C1']   # Blue, Orange, Green
colors = ['#4e79a7', '#f28e2b', '#76b7b2']
barWidth = 0.25
r1 = np.arange(len(categories))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
# data1, data2, data3, data4 = np.random.rand(4, 3), np.random.rand(4, 3), np.random.rand(4, 3), np.random.rand(4, 3)
data1 = np.array([[0.415, 0.382, 0.221], [0.43, 0.373, 0.223], [0.421, 0.358, 0.238], [0.418, 0.352, 0.194]])
data2 = np.array([[0.665, 0.604, 0.434], [0.678, 0.622, 0.457], [0.673,0.601, 0.465], [0.662, 0.592, 0.382]])
data3 = np.array([[0.58, 0.49, 0.354], [0.588, 0.514, 0.332], [0.591, 0.528, 0.328], [0.568, 0.494, 0.365]])
data4 = np.array([[3.8, 3.13, 1.52], [3.97, 3.65, 1.70], [3.98, 3.58, 1.67 ], [3.94, 3.28, 2.01]])

# Font size settings
title_fontsize = 20
label_fontsize = 20
tick_fontsize = 15
legend_fontsize = 20
axis_tick_fontsize =20

# Creating a figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plotting each chart\
name = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"]
for i, ax in enumerate(axs.flat):
    data = [data1, data2, data3, data4][i]
    ax.bar(r1, data[:, 0], color=colors[0], width=barWidth, edgecolor='grey', label=metrics[0] if i == 0 else "")
    ax.bar(r2, data[:, 1], color=colors[1], width=barWidth, edgecolor='grey', label=metrics[1] if i == 0 else "")
    ax.bar(r3, data[:, 2], color=colors[2], width=barWidth, edgecolor='grey', label=metrics[2] if i == 0 else "")
    # ax.set_xlabel('Categories', fontweight='bold', fontsize=label_fontsize)
    ax.set_ylabel('Scores', fontsize=label_fontsize)
    ax.set_xticks([r + barWidth for r in range(len(data))])
    ax.set_xticklabels(categories, fontsize=tick_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)  # Set tick label size
    ax.grid(True, which='both', linestyle='--', linewidth=0.75)
    ax.set_title(name[i], fontsize=title_fontsize)

# Adding a single shared legend above the charts
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=legend_fontsize)

# Adding a single overall title at the bottom of the figure
# fig.suptitle('Zero-shot Test', fontsize=title_fontsize + 8, x=0.5, y=-0.01)

# Adjusting layout
plt.tight_layout()

# Saving the figure as an SVG file

# Saving the figure as an SVG file
svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/zeroshot_2.pdf'
plt.savefig(svg_file_path, format='pdf', bbox_inches='tight')
plt.show()