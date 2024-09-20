# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# # Initialize the tokenizer for the LLaMA model
# tokenizer = AutoTokenizer.from_pretrained("/local/scratch0/data_and_checkpoint/models/llama-7b")

# # The sentence to tokenize
# sentence = "atrial fibrillation with rapid ventricular response. *** report made without knowing patient's sex or age ***. old inferior infarct. possible anterior infarct - age undetermined. lateral st-t changes may be due to myocardial ischemia. repolarization changes may be partly due to rate/rhythm. abnormal ecg"

# # Tokenize the sentence and get the input IDs
# input_ids = tokenizer.encode(sentence, add_special_tokens=True)
# config = AutoConfig.from_pretrained("/local/scratch0/data_and_checkpoint/models/llama-7b")
# model = AutoModelForCausalLM.from_pretrained(
#                 "/local/scratch0/data_and_checkpoint/models/llama-7b",
#                 config=config,
#                 low_cpu_mem_usage=True,
#                 use_flash_attention_2=False,
#             )

# for name, param in model.state_dict().items():
#     print(name)
# # Print the number of input IDs
# print(len(input_ids))\\

from bert_score import score
def get_bert_score(gen_sents_or_reports, ref_sents_or_reports):
    P, R, F1 = score(gen_sents_or_reports, ref_sents_or_reports, lang="en", rescale_with_baseline=True)

    # Calculate average scores
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    return avg_precision, avg_recall, avg_f1

a = ['Today I have a good day, but I am so sad because nothing to do']
b = ['Although today has been a good day for me, I feel sad because nothing to do xxxxxxxxxxxx 555']

print(get_bert_score(a, b))


import numpy as np
import torch
from torch import nn
import random
import transformers