import pandas as pd
import re
import sys
from clean import TextCleaner

from transformers import BertPreTrainedModel, BertConfig, BertModel
from encode import BERTEncoder
import torch
import math
import datetime
import numpy as np
import gc
from patent_utils import *

class EarlyStopping(object):
    def __init__(self, model, patience, start_time, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.early_stopping_counter = 0
        self.best_valid_acc = 0.00
        self.early_stop = False
        self.model = model
        self.start_time = start_time
   
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
        path_to_checkpoint = (
            f'models/{self.start_time}'
            + f'_{model_name}.pth'
        )
        torch.save(self.model, path_to_checkpoint)
        print(f"Current best model saved as {path_to_checkpoint} at epoch {epoch}.")
    