import pandas as pd
import re
import sys
sys.path.append("../")
from clean import TextCleaner

from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers import BertTokenizer
import torch
import math
import numpy as np
from patent_utils import *
import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('patent_documents', type=str, 
#                     help='Patent documents')
# parser.add_argument(']tsd_documents', type=str, 
#                     help='TSD documents')
# parser.add_argument('labels', type=str, 
#                     help='labels')

# args = parser.parse_args()

class BERTEncoder(object):
    def __init__(self, patent_documents: list, tsd_documents: list = [], labels: list = [], patentCleaner: PatentCleaner = PatentCleaner()):
        self.num_docs = len(patent_documents)
        self.max_input_length = 512
        self.patent_documents = (lambda x: patentCleaner.clean(x))(patent_documents)
        if tsd_documents != []:
            self.tsd_documents = (lambda x: patentCleaner.clean(x))(tsd_documents) 
            self.tsd_train, self.tsd_test = split_train_test(self.tsd_documents)
        self.patent_train, self.patent_test = split_train_test(self.patent_documents)
        if labels != []:
            self.labels = labels
            self.labels_train, self.labels_test = split_train_test(self.labels) 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
    def encode_tsd(self, patent_documents: list, tsd_documents: list, labels: list):
        tokenized_documents = [self.tokenizer.tokenize(document) for document in tsd_documents]
        max_sequences_per_document = math.ceil(max(len(x)/(self.max_input_length-2) for x in tokenized_documents))
        tsd_output = []
        patent_output = []
        label_output = []
        document_seq_lengths = [] 
        tsd_dict = {}
        global_tsd_index = 0

        for doc_index, (patent_document, tokenized_document) in enumerate(zip(patent_documents, tokenized_documents)):
            max_seq_index = 0
            local_tsd_index = 0
            for seq_index, i in enumerate(range(0, len(tokenized_document), (self.max_input_length-2))):
                raw_tokens = tokenized_document[i:i+(self.max_input_length-2)]
                tokens = []
                input_type_ids = []
                global_tsd_index += 1
                local_tsd_index += 1

                tokens.append("[CLS]")
                input_type_ids.append(0)
                for token in raw_tokens:
                    tokens.append(token)
                    input_type_ids.append(0)
                tokens.append("[SEP]")
                input_type_ids.append(0)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                attention_masks = [1] * len(input_ids)

                while len(input_ids) < self.max_input_length:
                    input_ids.append(0)
                    input_type_ids.append(0)
                    attention_masks.append(0)

                assert len(input_ids) == self.max_input_length and len(attention_masks) == self.max_input_length and len(input_type_ids) == self.max_input_length

                tsd_output.append(torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                               torch.LongTensor(input_type_ids).unsqueeze(0),
                                                               torch.LongTensor(attention_masks).unsqueeze(0)),
                                                              dim=0))
                patent_output.append(patent_document)
                label_output.append(labels[doc_index])
                max_seq_index = seq_index
                document_seq_lengths.append(max_seq_index+1)
            tsd_dict[str(doc_index)] = [np.arange(global_tsd_index - local_tsd_index, global_tsd_index, 1)]
        return patent_output, tsd_output, label_output, torch.LongTensor(document_seq_lengths), tsd_dict
    
    def encode_patents(self, documents: list):
        tokenized_documents = [self.tokenizer.tokenize(document) for document in documents]
        max_sequences_per_document = math.ceil(max(len(x)/(self.max_input_length-2) for x in tokenized_documents))

        output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, self.max_input_length), dtype=torch.long)
        document_seq_lengths = []

        for doc_index, tokenized_document in enumerate(tokenized_documents):
            max_seq_index = 0
            for seq_index, i in enumerate(range(0, len(tokenized_document), (self.max_input_length-2))):
                raw_tokens = tokenized_document[i:i+(self.max_input_length-2)]
                tokens = []
                input_type_ids = []

                tokens.append("[CLS]")
                input_type_ids.append(0)
                for token in raw_tokens:
                    tokens.append(token)
                    input_type_ids.append(0)
                tokens.append("[SEP]")
                input_type_ids.append(0)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                attention_masks = [1] * len(input_ids)

                while len(input_ids) < self.max_input_length:
                    input_ids.append(0)
                    input_type_ids.append(0)
                    attention_masks.append(0)

                assert len(input_ids) == self.max_input_length and len(attention_masks) == self.max_input_length and len(input_type_ids) == self.max_input_length

                output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                               torch.LongTensor(input_type_ids).unsqueeze(0),
                                                               torch.LongTensor(attention_masks).unsqueeze(0)),
                                                              dim=0)
                max_seq_index = seq_index
            document_seq_lengths.append(max_seq_index+1)
        return output, torch.LongTensor(document_seq_lengths)
    
    def tokenize_data(self, patent_documents: list, tsd_documents: list, labels: list):
        patent_documents, tsd_representations, labels, tsd_sequence_lengths, tsd_dict = self.encode_tsd(patent_documents, tsd_documents, labels)
        patent_representations, patent_sequence_lengths = self.encode_patents(patent_documents)
        correct_output = torch.FloatTensor(labels)

        output = torch.zeros(size=(len(tsd_representations), 1, 3, self.max_input_length), dtype=torch.long)

        for doc_index, tsd_representation in enumerate(tsd_representations):
            output[doc_index][0] = tsd_representation

        tsd_representations = output
    
        return patent_representations, tsd_representations, correct_output, tsd_dict
    
    def tokenize_train_data(self):
        patent_representations, tsd_representations, correct_output, tsd_dict = self.tokenize_data(self.patent_train, self.tsd_train, self.labels_train)
        return patent_representations, tsd_representations, correct_output, tsd_dict
    
    def tokenize_test_data(self):
        patent_representations, tsd_representations, correct_output, tsd_dict = self.tokenize_data(self.patent_test, self.tsd_test, self.labels_test)
        return patent_representations, tsd_representations, correct_output, tsd_dict
    
# if __name__ == '__main__':
#     bertEncoder = BERTEncoder(args.patent_documents, args.tsd_documents, args.labels, PatentCleaner())
#     patent_train_representations, tsd_train_representations, correct_output_train = bertEncoder.tokenize_train_data()
#     patent_test_representations, tsd_test_representations, correct_output_test = bertEncoder.tokenize_test_data()