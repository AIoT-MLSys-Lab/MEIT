import matplotlib.pyplot as plt

# Data for the plot
# models = {
#     "BLOOM": [0.725, 0.7174, 0.7113, 0.6939, 0.6897],
#     "OPT": [0.732, 0.7288, 0.7254, 0.7139, 0.7110],
#     "LLaMA-1": [0.724, 0.7236, 0.7223, 0.7118, 0.6981],
#     "Mistral": [0.740, 0.7308, 0.7261, 0.7227, 0.7196]
# }
# max_proportion = [0, 0.05, 0.10, 0.15, 0.20]

# # Font size settings
# title_fontsize = 23
# axis_label_fontsize = 20
# legend_fontsize = 20
# tick_fontsize = 20

# # Creating the plot with adjusted font sizes
# plt.figure(figsize=(10, 6))
# for model, scores in models.items():
#     plt.plot(max_proportion, scores, label=model, marker='o')

# plt.xlabel('Max Proportion', fontsize=axis_label_fontsize)
# plt.ylabel('Score', fontsize=axis_label_fontsize)
# plt.title('ROUGE-L', fontsize=title_fontsize)
# plt.legend(fontsize=legend_fontsize)
# plt.xticks(fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# plt.grid(True)

# # Show the plot
# plt.show()

import matplotlib.pyplot as plt

# Data for the plot
models = {
    "BLOOM": [0.758 , 0.7463, 0.7425, 0.7343, 0.7169],
    "OPT": [0.755, 0.7414, 0.7354, 0.7245, 0.7155],
    "LLaMA-1": [0.761, 0.7575, 0.7504, 0.7368, 0.7321],
    "Mistral": [0.763, 0.7572, 0.7519, 0.7486, 0.7371]
}
max_proportion = [1, 0.95, 0.90, 0.15, 0.80]

# Different markers for each line
markers = ['o', 's', '^', 'D']
line_width = 2.5  # Increased line width

# Font size settings
title_fontsize = 26
axis_label_fontsize = 25
legend_fontsize = 25
tick_fontsize = 25


# Creating the plot with adjusted font sizes, different markers, and line width
plt.figure(figsize=(10, 6))
for (model, scores), marker in zip(models.items(), markers):
    plt.plot(max_proportion, scores, label=model, marker=marker, linewidth=line_width, markersize=12)

plt.xlabel('Signal-to-noise Ratios', fontsize=axis_label_fontsize)
plt.ylabel('Scores', fontsize=axis_label_fontsize)
plt.title('METEOR', fontsize=title_fontsize)
plt.xticks(max_proportion, labels=[str(p) for p in max_proportion], fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)

svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/new_robutness_meteor.pdf'
plt.savefig(svg_file_path, format='pdf', bbox_inches='tight')
