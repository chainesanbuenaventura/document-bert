import pandas as pd
import re
import sys
import os
from document_bert.document_search.clean import TextCleaner

from transformers import BertPreTrainedModel, BertConfig, BertModel
from document_bert.document_search.encode import BERTEncoder
import torch
import math
import datetime
import numpy as np
import gc
from document_bert.document_search.patent_utils import *

class EarlyStopping(object):
    def __init__(self, model, patience, start_time, model_path, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.early_stopping_counter = 0
        self.best_valid_acc = 0.00
        self.early_stop = False
        self.model = model
        self.start_time = start_time
        self.model_path = model_path
   
    def __call__(self):
        if self.early_stopping_counter > self.patience:
            self.early_stop = True
            
        return self.early_stop
        
    def record(self, valid_acc, epoch):
        if valid_acc < self.best_valid_acc:
            self.early_stopping_counter += 1
            print(valid_acc, self.best_valid_acc, self.early_stopping_counter)
        else:
            self.early_stopping_counter = 0
            self.best_valid_acc = valid_acc
            self.save_best_model(epoch)
            print(valid_acc, self.best_valid_acc, self.early_stopping_counter)
    
    def save_best_model(self, epoch):
        model_name = 'BERTsimilaritymodel'
        if os.path.exists(self.model_path) != True:
            os.mkdir(self.model_path)
            
        path_to_checkpoint = (
            f'{self.model_path}/{self.start_time}'
            + f'_{model_name}.pth'
        )
        path_to_checkpoint_state_dict = (
            f'{self.model_path}/{self.start_time}'
            + f'_{model_name}_state_dict.pth'
        )
        torch.save(self.model, path_to_checkpoint)
        torch.save(self.model.state_dict, 'path_to_checkpoint_state_dict')
        print(f"Current best model saved as {path_to_checkpoint} at epoch {epoch}.")
    