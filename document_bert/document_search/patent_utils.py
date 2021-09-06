import pandas as pd
import re
import sys
from clean import TextCleaner
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch
import math
import numpy as np
from datetime import datetime

class PatentCleaner(TextCleaner):
    def __init__(self):
        super(PatentCleaner, self).__init__()
        
    def remove_table_nums(self, paragraphs):
        paragraphs = [re.sub(r'[(0-9).(0.9)]', "", paragraph) for paragraph in paragraphs]
        
        return paragraphs
    
    def clean(self, paragraphs):
        paragraphs = self.remove_special_characters(paragraphs)
        paragraphs = self.remove_start_end_spaces(paragraphs)
        paragraphs = self.remove_multiple_spaces(paragraphs)
        paragraphs = self.remove_start_numbers(paragraphs)
        paragraphs = self.remove_table_nums(paragraphs)
        
        return paragraphs

def split_train_test(data):
    train_data, test_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]

    return train_data, test_data

def round_custom(in_array, threshold):
    assert threshold in torch.arange(0, 1, 0.05)
    return torch.where(in_array > threshold, 1, 0)

def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y.%H.%M.%S")
    return dt_string
