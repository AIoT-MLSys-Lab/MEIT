import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from PIL import Image
import wfdb
import os


from rouge import Rouge 


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


class PTB_E_T_Dataset(Dataset):
    def __init__(self, ecg_data, transform=None, **args):
        # self.ecg_data = ecg_data
        self.ecg_meta_path = ecg_data

        self.text_csv = args['text']
        self.mode = args['train_test']
        self.transform = transform

    def __len__(self):
        return (self.text_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        path = self.text_csv['filename_lr'].iloc[idx]
        path = os.path.join(self.ecg_meta_path, path)
        ecg = wfdb.rdsamp(path)[0]
        ecg = ecg.T
        ecg[torch.where(torch.isnan(ecg))] = 0
        ecg[torch.where(torch.isinf(ecg))] = 0
        
        # get raw text
        report = self.text_csv['report'].iloc[idx]
        # convert to all lower case
        report = report.lower()

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample

class MIMIC_E_T_Dataset(Dataset):
    def __init__(self, ecg_data, transform=None, **args):
        self.ecg_meta_path = ecg_data
        self.mode = args['train_test']
        self.text_csv = args['text_csv']
        self.record_csv = args['record_csv']
        self.transform = transform

    def __len__(self):
        return (self.text_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get ecg
        study_id = self.text_csv['study_id'].iloc[idx]
        if study_id == self.record_csv['study_id'].iloc[idx]:
            path = self.record_csv['path'].iloc[idx]
        else:
            print('Error: study_id not match!')
        path = os.path.join(self.ecg_meta_path, path)
        ecg = wfdb.rdsamp(path)[0]
        ecg = ecg.T
        # noramlize
        ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

        # get raw text
        report = self.text_csv.iloc[idx][['report_0', 'report_1',
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

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
                sample['ecg'][torch.where(torch.isnan(sample['ecg']))] = 0
                sample['ecg'][torch.where(torch.isinf(sample['ecg']))] = 0
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
                sample['ecg'][torch.where(torch.isnan(sample['ecg']))] = 0
                sample['ecg'][torch.where(torch.isinf(sample['ecg']))] = 0
        return sample


class ECG_TEXT_Dsataset:

    def __init__(self, ecg_path, csv_path, dataset_name='mimic'):
        self.ecg_path = ecg_path
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.csv = pd.read_csv(self.csv_path, low_memory=False)
        self.record_csv = pd.read_csv(os.path.join(self.ecg_path, 'record_list.csv'), low_memory=False)
        
        # sort and reset index by study_id
        self.csv = self.csv.sort_values(by=['study_id'])
        self.csv.reset_index(inplace=True, drop=True)
        self.record_csv = self.record_csv.sort_values(by=['study_id'])
        self.record_csv.reset_index(inplace=True, drop=True)

        # split train and val
        self.train_csv, self.val_csv, self.train_record_csv, self.val_record_csv = \
            train_test_split(self.csv, self.record_csv, test_size=0.02, random_state=42)
        # sort and reset index by study_id
        self.train_csv = self.train_csv.sort_values(by=['study_id'])
        self.val_csv = self.val_csv.sort_values(by=['study_id'])
        self.train_csv.reset_index(inplace=True, drop=True)
        self.val_csv.reset_index(inplace=True, drop=True)

        self.train_record_csv = self.train_record_csv.sort_values(by=['study_id'])
        self.val_record_csv = self.val_record_csv.sort_values(by=['study_id'])
        self.train_record_csv.reset_index(inplace=True, drop=True)
        self.val_record_csv.reset_index(inplace=True, drop=True)
        
        print(f'train size: {self.train_csv.shape[0]}')
        print(f'val size: {self.val_csv.shape[0]}')

    def get_dataset(self, train_test, T=None):

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        
        if self.dataset_name == 'ptbxl':
            misc_args = {'train_test': train_test,
                   'text': self.csv}
        
            dataset = PTB_E_T_Dataset(ecg_data=self.ecg_path,
                                       transform=Transforms,
                                       **misc_args)
            
        elif self.dataset_name == 'mimic':
            
            if train_test == 'train':
                misc_args = {'train_test': train_test,
                   'text_csv': self.train_csv,
                   'record_csv': self.train_record_csv}
            else:
                misc_args = {'train_test': train_test,
                   'text_csv': self.val_csv,
                   'record_csv': self.val_record_csv}
            
        
            dataset = MIMIC_E_T_Dataset(ecg_data=self.ecg_path,
                                       transform=Transforms,
                                       **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))
        
        return dataset
