
"""
This module contains all functions used to evaluate the ECG_LLMs.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include:
    - METEOR for:
        - generated reports for ECG_LLMs with corresponding ecg signal


    - BLEU 1-4, METEOR, ROUGE-L, CIDEr-D for all generated reports
    - ECG signal efficacy metrics for all generated reports:
        - micro-averaged over 5 observations
        - exampled-based averaged over all 14 observations
        - computed for each observation individually

"""


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from collections import defaultdict
import csv
import io
import os
import re
import tempfile

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import gzip
import pathlib
import pickle

import numpy as np
from pycocoevalcap.cider.cider_scorer import CiderScorer
from nltk.tokenize import wordpunct_tokenize
from utils_ecg import get_rouge_n_gram
import nltk

from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
from bert_score import score
################ modify the code here #######################


def get_bert_score(gen_sents_or_reports, ref_sents_or_reports):
    P, R, F1 = score(gen_sents_or_reports, ref_sents_or_reports, lang="en", rescale_with_baseline=True)

    # Calculate average scores
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    return avg_precision, avg_recall, avg_f1


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

def get_reference_reports_val_set(val_or_test_dataset):
    ref_reports = []

    # Miura computes the document frequency on the "findings" section of the reference reports of the train set,
    # but since my train set does not have the reference reports, I calculate it on the val set
    # path_val_set_csv_file = os.path.join(val_or_test_dataset, "valid.csv")

    # with open(path_val_set_csv_file) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=",")

    #     # skip the first line (i.e. the header line)
    #     next(csv_reader)

    #     for row in csv_reader:
    #         reference_report = row[-1]
    #         ref_reports.append(reference_report)

    for example in val_or_test_dataset:
        message = example['messages'][1]['content']
        
        ref_reports.append(message)
    print('################ proprecess the message done ################', len(ref_reports))
    return ref_reports

def compute_cider_df(test_dataset):
    # 2 functions below are based on the default command line arguments that Miura uses
    tokenize_func = wordpunct_tokenize
    textfilter_func = str.lower

    ref_reports = get_reference_reports_val_set(test_dataset)

    # processed_ref_reports is the equivalent of what Miura calls "ftexts" in line 58 of his implementation
    processed_ref_reports = []
    for ref_report in ref_reports:
        tokens = tokenize_func(textfilter_func(ref_report))
        processed_ref_report = " ".join(tokens)
        processed_ref_reports.append(processed_ref_report)

    # these 3 lines are equivalent to line 65 of Miura's implementation
    scorer = CiderScorer(refs=processed_ref_reports)
    scorer.compute_doc_freq()
    df = scorer.document_frequency

    parent_path_of_this_file = pathlib.Path(__file__).parent.resolve()
    output_path = os.path.join(parent_path_of_this_file, "ptbxl_ecg-document-frequency.bin.gz")
    with gzip.open(output_path, 'w') as f:
        pickle.dump(df, f)

class CustomCiderScorer(CiderScorer):
    """
    Custom Cider Scorer uses document frequency calculated on the reference reports of the validation set.
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0, test_dataset=None):
        super().__init__(test, refs, n, sigma)
        self.test_dataset = test_dataset

        self.document_frequency = self._get_document_frequency()

    def _get_document_frequency(self):
        parent_path_of_this_file = pathlib.Path(__file__).parent.resolve()
        df_file = os.path.join(parent_path_of_this_file, "ptbxl_ecg-document-frequency.bin.gz")

        if not os.path.exists(df_file):
            compute_cider_df(self.test_dataset)

        with gzip.open(df_file) as f:
            cider_df = pickle.load(f)

        return cider_df

    def compute_score(self):
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)

class Cider:
    """
    Main Class to compute the CIDEr metric
    """
    def __init__(self, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res, test_dataset):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CustomCiderScorer(n=self._n, sigma=self._sigma, test_dataset=test_dataset)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"


def compute_metric_scores(nlg_metrics: list[str], ori_gen_sents_or_reports, ori_ref_sents_or_reports, test_dataset) -> dict[str, float]:
    
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
    """
    Computes NLG metrics that are specified in metrics list (1st input argument):
        - Bleu 1-4
        - Meteor
        - Rouge-1 2
        - Rouge-L
        - Cider-D

    Returns a dict that maps from the metrics specified to the corresponding scores.

    """
   
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()  # this is actually the Cider-D score, even if the class name only says Cider

    gen_sents_or_reports = convert_for_pycoco_scorer(ori_gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ori_ref_sents_or_reports)

    
    nlg_scores = {} # Bleu 1-4, Meteor, Rouge-1 2, Rouge-L, Cider-D
    if "bert_score" in nlg_metrics:
        avg_precision, avg_recall, avg_f1 = get_bert_score(gen_sents_or_reports, ref_sents_or_reports)
        nlg_scores['avg_precision'] = avg_precision
        nlg_scores['avg_recall'] = avg_recall
        nlg_scores['avg_f1'] = avg_f1

    if "meteor" in nlg_metrics:

        avg_meteor = get_meteor_score(ori_gen_sents_or_reports, ori_ref_sents_or_reports)

        nlg_scores['meteor'] = avg_meteor

        # print('########### metric_name #############', 'meteor')
        # print('################### Score #########################', avg_meteor)



    for metric_name, scorer in scorers.items():
        if metric_name == 'cider':
            score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports, test_dataset)
            # print('########### metric_name #############', metric_name)
            # print('################### Score #########################', score)
        else:

            score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
            # print('########### metric_name #############', metric_name)
            # print('################### Score #########################', score)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    ########### compute rouge 1 and 2 here ##############

    avg_rouge_1, avg_rouge_2 = get_rouge_n_gram(ori_gen_sents_or_reports, ori_ref_sents_or_reports)
    nlg_scores['rouge_1'] = avg_rouge_1
    nlg_scores['rouge_2'] = avg_rouge_2

    # print('########### metric_name #############', 'rouge_1')
    # print('################### Score #########################', avg_rouge_1)

    # print('########### metric_name #############', 'rouge_2')
    # print('################### Score #########################', avg_rouge_2)

    return nlg_scores

def compute_semantic_metric_scores(nlg_metrics: list[str], ori_gen_sents_or_reports, ori_ref_sents_or_reports, test_dataset) -> dict[str, float]:

    avg_precision, avg_recall, avg_f1 = get_bert_score(ori_gen_sents_or_reports, ori_ref_sents_or_reports)

    nlg_scores = {}

    nlg_scores['precision'] = avg_precision
    nlg_scores['recall'] = avg_recall
    nlg_scores['f1'] = avg_f1
    
    return nlg_scores