from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
nltk.download('wordnet')
from bert_score import score
import re
from rouge import Rouge 
# model_name = '/local/scratch0/data_and_checkpoint/models/llama-7b'  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model = model.cuda()

# input_text_1 =  "[IMAGE 1] Given the corresponding ECG signal embeddings, please help me generate an accurate description for this ECG signal embeddings: "
# input_text_2 =  "[IMAGE 2] Given the corresponding ECG signal embeddings, please help me generate an accurate description for this ECG signal embeddings: "
# input_text = [input_text_1, input_text_2]
# input_ids = tokenizer.batch_encode_plus(input_text, return_tensors='pt')
# print(input_ids)

# with torch.no_grad():
#     output = model.generate(input_ids['input_ids'].cuda(), max_length=128)
#     print(output)

# generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
# generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
# print(generated_text)
# reference_tokens = 'The ECG report is very good to understand. '
# generated_tokens = 'The ECG report is so good to realize. '
# tokenized_reference_tokens = word_tokenize(reference_tokens)
# tokenized_generated_text = word_tokenize(generated_tokens)
# meteor = meteor_score([tokenized_reference_tokens], tokenized_generated_text)
# print(f"METEOR score: {meteor}")


# reference_tokens = 'The ECG report is very good to understand. '
# generated_tokens = 'The ECG report is so good to realize. '
# tokenized_reference_tokens = word_tokenize(reference_tokens)
# tokenized_generated_text = word_tokenize(generated_tokens)
# meteor = meteor_score([tokenized_reference_tokens], tokenized_generated_text)
# print(f"METEOR score: {meteor}")

def convert_for_pycoco_scorer(sents_or_reports: list[str]):
    """
    The compute_score methods of the scorer objects require the input not to be list[str],
    but of the form:
    generated_reports =
    {
        "ecg_id_0" = ["1st generated report"],
        "ecg_id_1" = ["2nd generated report"],
        ...
    }

    Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
    following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) implementation (https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py):
    see lines 132 and 133
    """
    sents_or_reports_converted = {}
    for num, text in enumerate(sents_or_reports):
        sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

    return sents_or_reports_converted

candidates = ["Generated text 1", "Generated text 2", "Generated text 3"]

# List of reference texts
references = ["no textXXXXX 1", "greate text TEXT text 1", "yes text text1"]

# # Calculate BERTScore
# P, R, F1 = score(candidates, references, lang="en", rescale_with_baseline=True)

# # Calculate average scores
# avg_precision = P.mean().item()
# avg_recall = R.mean().item()
# avg_f1 = F1.mean().item()

# # Print the average scores
# print(f"Average Precision: {avg_precision}")
# print(f"Average Recall: {avg_recall}")
# print(f"Average F1 Score: {avg_f1}")


def get_meteor_score(gen_sents_or_reports, ref_sents_or_reports):
    total_meteor = 0
    if len(gen_sents_or_reports) == len(ref_sents_or_reports):
        for gen_text, ref_text in zip(gen_sents_or_reports, ref_sents_or_reports):
            tokenized_reference_tokens = word_tokenize(ref_text)
            tokenized_generated_text = word_tokenize(gen_text)
            meteor = meteor_score([tokenized_reference_tokens], tokenized_generated_text)

            total_meteor+=meteor
        avg_meteor = total_meteor / len(gen_sents_or_reports)

        return avg_meteor

def get_rouge_n_gram(generated_texts, reference_texts):
    rouge = Rouge()
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0

    if len(generated_texts) == len(reference_texts):
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            scores = rouge.get_scores(gen_text, ref_text)
            total_rouge_1 += scores[0]['rouge-1']['f']
            total_rouge_2 += scores[0]['rouge-2']['f']
            # total_rouge_l += scores[0]['rouge-l']['f']

        # 计算平均分数
        avg_rouge_1 = total_rouge_1 / len(generated_texts)
        avg_rouge_2 = total_rouge_2 / len(generated_texts)
        # avg_rouge_l = total_rouge_l / len(generated_texts)

    return avg_rouge_1, avg_rouge_2



gen_sents_or_reports = convert_for_pycoco_scorer(candidates)
ref_sents_or_reports = convert_for_pycoco_scorer(references)

print(gen_sents_or_reports)


print(get_rouge_n_gram(candidates, references))