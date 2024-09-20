import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# File paths and colors
new_file_paths = {
      'GPT-Neo': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/train_loss/run-gpt_neo_lora_ckpt_open_instruct-tag-train_loss.csv',
    'BLOOM': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/train_loss/run-bloom_exp_3_open_instruct-tag-train_loss.csv',
        'OPT': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/train_loss/run-opt_exp_1_open_instruct-tag-train_loss.csv',
    'LLaMA-2': '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/train_loss/run-llama_2_exp_1_open_instruct-tag-train_loss.csv',
}
colors =  ['cornflowerblue', 'green', 'red', 'orange']
plt.figure(figsize=(8, 6))
# Adjusting font sizes
title_fontsize = 25
label_fontsize = 23
legend_fontsize = 23
tick_fontsize = 23

# Function to format the x-axis labels to use 'k' for thousands
def format_func(value, tick_number):
    return f'{int(value / 1000)}k' if value >= 1000 else int(value)

# Creating the plot with adjusted font sizes and y-axis range
plt.figure(figsize=(8, 6))
for (model, file_path), color in zip(new_file_paths.items(), colors):
    df = pd.read_csv(file_path)
    plt.plot(df['Step'], df['Value'], label=model, color=color)

# Setting plot details with adjusted font sizes and y-axis range
plt.xlabel('Steps', fontsize=label_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)
plt.title('Instruction Tuning Loss', fontsize=title_fontsize)

plt.legend(fontsize=legend_fontsize)
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))
ax.tick_params(axis='both', labelsize=tick_fontsize)
plt.ylim(0, 0.5)  # Set y-axis range

# Display the plot
plt.show()
svg_file_path = '/users/PAS2473/brucewan666/ECG/ECG/draw_figures/new_loss.pdf'
plt.savefig(svg_file_path, format='pdf', bbox_inches='tight')