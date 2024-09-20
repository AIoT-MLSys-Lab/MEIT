import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker

# Adjusting all font sizes
title_fontsize = 25
label_fontsize = 23
legend_fontsize = 23
tick_fontsize = 23

# Function to format the x-axis labels to use 'k' for thousands
def format_func(value, tick_number):
    return f'{int(value / 1000)}k' if value >= 1000 else int(value)

# File paths
file_paths = {
     'GPT-Neo': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/meteor_metric/run-gpt_neo_lora_ckpt_open_instruct-tag-avg_meteor.csv',
    'BLOOM': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/meteor_metric/run-bloom_exp_3_open_instruct-tag-avg_meteor.csv',
      'OPT': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/meteor_metric/run-opt_exp_1_open_instruct-tag-avg_meteor.csv',
    'LLaMA-2': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/meteor_metric/run-llama_2_exp_1_open_instruct-tag-avg_meteor.csv',
}

# Creating the plot with adjusted font sizes
plt.figure(figsize=(8, 6))

# Plotting data with distinct colors
colors = ['cornflowerblue', 'green', 'red', 'orange']
for (model, file_path), color in zip(file_paths.items(), colors):
    df = pd.read_csv(file_path)
    plt.plot(df['Step'], df['Value'], label=model, color=color)

# Setting plot details with adjusted font sizes
plt.xlabel('Steps', fontsize=label_fontsize)
plt.ylabel('Scores', fontsize=label_fontsize)
plt.title('METEOR Metric', fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)

# Formatting x-axis and adjusting tick label sizes
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))  # Apply the formatting function
ax.tick_params(axis='both', labelsize=tick_fontsize)
svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/new_Meteor.pdf'
plt.savefig(svg_file_path, format='pdf', bbox_inches='tight')
# Display the plot
plt.show()