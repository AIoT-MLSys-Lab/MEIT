import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import yaml
import sys
import json
from utils_ecg import ECG_TEXT_Dsataset


def preprocess_data():
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # loading data path
    text_path = config['dataset']['text_path']
    ecg_path = config['dataset']['ecg_path']

    train_dataset = ECG_TEXT_Dsataset(
        ecg_path=ecg_path, csv_path=text_path, dataset_name=config['dataset']['dataset_name'])
    train_dataset = train_dataset.get_dataset(train_test='train')

    print('Return the dataset successfully! ')


def build_instruct_dataset(ecg_name='mimic', save_path=None):
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # loading data path
    text_path = config['dataset']['text_path']
    ecg_path = config['dataset']['ecg_path']

    csv = pd.read_csv(text_path, low_memory=False)


    if ecg_name == 'ptbxl':
        text_csv = csv
        jsonl_save = []
        for idx in range(text_csv.shape[0]):

            current_line = {}

            report = text_csv['report_en'].iloc[idx]
            # only keep not NaN
            # convert to all lower case
            report = report.lower()

            if 'something wrong with the data' in report:
                pass
            else:
                print('#################### report ######################:', report)

                current_line["dataset"] = ecg_name
                current_line["id"] = ecg_name + "_" + str(idx)
                current_line["ecg_path"] = os.path.join(ecg_path, text_csv['filename_lr'].iloc[idx]).replace('records100', 'records500').replace('_lr', '_hr')

                prompt = {}
                prompt["role"] = "user"
                prompt["content"] = "" # will add in the main code

                answer = {}
                answer["role"] = "assistant"
                answer["content"] = report # output of the proposed prompt

                # ecg_path_example = {}
                # ecg_path_example["role"] = "ecg_path"
                # ecg_path_example["content"] = os.path.join(ecg_path, text_csv['filename_lr'].iloc[idx])
                

                current_line["messages"] = [prompt, answer]
                jsonl_save.append(current_line)

        with open(save_path, 'w') as file:
            for item in jsonl_save:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
       

    elif ecg_name == 'mimic':
        csv = csv.sort_values(by=['study_id']) # csv for the ecg text document
        csv.reset_index(inplace=True, drop=True)
        print(f'total csv size: {csv.shape[0]}')

        record_csv = pd.read_csv(os.path.join(ecg_path, 'record_list.csv'), low_memory=False)
        record_csv = record_csv.sort_values(by=['study_id']) # csv for the path_index of ecg signal
        record_csv.reset_index(inplace=True, drop=True)

        text_csv = csv
        jsonl_save = []
        for idx in range(record_csv.shape[0]):

            current_line = {}

            report = text_csv.iloc[idx][['report_0', 'report_1',
                'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7',
                'report_8', 'report_9', 'report_10', 'report_11', 'report_12',
                'report_13', 'report_14', 'report_15', 'report_16', 'report_17']]
            # only keep not NaN
            report = report[~report.isna()]
             # concat the report
            report = '. '.join(report)
            # preprocessing on raw text
            report = report.replace('EKG', 'ECG')
            report = report.replace('ekg', 'ecg')
            report = report.strip('*** ')
            report = report.strip(' ***')
            report = report.strip('***')
            report = report.strip('=-')
            report = report.strip('=')

            # convert to all lower case
            report = report.lower()
            print('############### Report ###############', report)

            current_line["dataset"] = ecg_name
            current_line["id"] = ecg_name + "_" + str(idx)
            current_line["ecg_path"] = os.path.join(ecg_path, record_csv['path'].iloc[idx])

            prompt = {}
            prompt["role"] = "user"
            prompt["content"] = "" # will add in the main code

            answer = {}
            answer["role"] = "assistant"
            answer["content"] = report # output of the proposed prompt

            # ecg_path_example = {}
            # ecg_path_example["role"] = "ecg_path"
            # # print('record_csv[path].iloc[idx]', record_csv['path'].iloc[idx])
            # ecg_path_example["content"] = os.path.join(ecg_path, record_csv['path'].iloc[idx])
            

            current_line["messages"] = [prompt, answer]
            jsonl_save.append(current_line)

        with open(save_path, 'w') as file:
            for item in jsonl_save:
                json_line = json.dumps(item)
                file.write(json_line + '\n')

if __name__ == '__main__':
    # build_instruct_dataset(save_path='/home/wan.512/ECG_LLMs/ECG_gen/instruct_data/mimic_ecg.jsonl') # mimic
    build_instruct_dataset(ecg_name='ptbxl',save_path='/users/PAS2473/brucewan666/ECG/ECG/instruct_data/ptbxl_ecg_train.jsonl') # mimic

