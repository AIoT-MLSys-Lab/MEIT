import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
from torchvision.transforms import transforms
import wfdb
import torch
import ecg_plot
import yaml


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    ecg_data_name: str = "mimic"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            # padding_side = "right"
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                
                ################################## add ecg original embeding here #####################################


                # config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

                # dataset_name = config["dataset"]["dataset_name"]




                if self.ecg_data_name == 'mimic':

                    Transforms = transforms.Compose([
                            transforms.ToTensor(),
                        ])
                    path =  feature['ecg']
                    ecg = wfdb.rdsamp(path)[0] # 12 * 5000

                    ecg = ecg.T
                    ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8) # 12 * dim
                    # add noise here #################

                    ecg = Transforms(ecg)
                    ecg = torch.squeeze(ecg, dim=0)
                    ecg[torch.where(torch.isnan(ecg))] = 0
                    ecg[torch.where(torch.isinf(ecg))] = 0
                    feature['ecg'] = ecg

                elif self.ecg_data_name == 'ptbxl':

                    Transforms = transforms.Compose([
                            transforms.ToTensor(),
                   ])
                    path = feature['ecg']
                    ecg = wfdb.rdsamp(path)[0]
                    ecg = ecg.T
                    ecg[torch.where(torch.isnan(ecg))] = 0
                    ecg[torch.where(torch.isinf(ecg))] = 0

                    ecg = Transforms(ecg)
                    ecg = torch.squeeze(ecg, dim=0)

                    feature['ecg'] = ecg.to("cuda")

        # print('feature[ecg] #################', feature['ecg'].device)
        # print('feature[input_ids] #################', feature['input_ids'].device)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features



@dataclass
class DataCollatorForSeq2Seq_for_robutness_test:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    ecg_data_name: str = None
    max_proportion: float = None


    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            # padding_side = "right"
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                
                ################################## add ecg original embeding here #####################################


                # config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

                # dataset_name = config["dataset"]["dataset_name"]

            

                if self.ecg_data_name == 'mimic':

                    Transforms = transforms.Compose([
                            transforms.ToTensor(),
                        ])
                    path =  feature['ecg']
                    ecg = wfdb.rdsamp(path)[0] # 12 * 5000

                    ecg = ecg.T
                    ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8) # 12 * dim
                    # add noise here #################


                    ecg = self.partial_white_noise(ecg, self.max_proportion, 0.1)



                    ecg = Transforms(ecg)
                    ecg = torch.squeeze(ecg, dim=0)
                    ecg[torch.where(torch.isnan(ecg))] = 0
                    ecg[torch.where(torch.isinf(ecg))] = 0
                    feature['ecg'] = ecg

                elif self.ecg_data_name == 'ptbxl':

                    Transforms = transforms.Compose([
                            transforms.ToTensor(),
        ])
                    path = feature['ecg']
                    ecg = wfdb.rdsamp(path)[0]
                    ecg = ecg.T
                    ecg[torch.where(torch.isnan(ecg))] = 0
                    ecg[torch.where(torch.isinf(ecg))] = 0

                    ecg = Transforms(ecg)
                    ecg = torch.squeeze(ecg, dim=0)

                    feature['ecg'] = ecg.to("cuda")

        # print('feature[ecg] #################', feature['ecg'].device)
        # print('feature[input_ids] #################', feature['input_ids'].device)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        

        return features

    def partial_white_noise(self, signal, max_proportion, upper_scale):
        
        output= signal.copy()
        len_signal = output.shape[-1]
        target_len = int(np.random.uniform(0, float(max_proportion)) * len_signal)
        cutout_start_pt = np.random.randint(0,len_signal - target_len)
        scale = np.random.uniform(0,upper_scale)
        output[:,cutout_start_pt:cutout_start_pt + target_len] += np.random.normal(0, scale, size=(1, target_len))
        return output

    