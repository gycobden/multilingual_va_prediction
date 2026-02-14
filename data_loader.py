import os
import pandas as pd
import numpy as np
from datasets import load_dataset, dataset_dict, DatasetDict
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, RobertaTokenizer, AutoTokenizer, XLMRobertaTokenizer, RobertaTokenizer
import torch
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MyDataset(Dataset):
    def __init__(self, filename, checkpoint, maxlen):
        if(checkpoint == 'distilbert-base-multilingual-cased'):
            self.tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
        elif(checkpoint == 'xlm-roberta-large'): 
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint)
        elif(checkpoint == 'xlm-roberta-base'): 
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint)

        df = pd.read_csv(
            filename,
            sep=',',
            quotechar='"',
            engine='python',
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True,
            keep_default_na=False
        )

        index_col = 'index' if 'index' in df.columns else 'id'
        text_col = 'text'
        valence_col = 'valence' if 'valence' in df.columns else 'V'
        arousal_col = 'arousal' if 'arousal' in df.columns else 'A'

        if text_col not in df.columns:
            raise KeyError(f"Missing required text column '{text_col}' in {filename}.")
        if valence_col not in df.columns or arousal_col not in df.columns:
            raise KeyError(f"Missing required label columns '{valence_col}'/'{arousal_col}' in {filename}.")

        self.index = df[index_col].to_list() if index_col in df.columns else list(range(len(df)))
        self.texts = df[text_col].to_list()
        self.valence = df[valence_col].astype(np.float64).to_list()
        self.arousal = df[arousal_col].astype(np.float64).to_list()
        self.maxlen = maxlen

    def __getitem__(self, idx):
        item = { }
        aux = self.tokenizer(self.texts[idx], max_length=self.maxlen, truncation=True, padding=False)
        item['input_ids'] = torch.tensor(aux['input_ids'])
        item['attention_mask'] = torch.tensor(aux['attention_mask'])
        item['labels'] = torch.tensor( [ self.valence[idx], self.arousal[idx] ] )

        return item

    def __len__(self):
        return len(self.texts)
    


        

    