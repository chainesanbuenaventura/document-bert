import pandas as pd
import re
import sys
from clean import TextCleaner

from transformers import BertPreTrainedModel, BertConfig, BertModel
from encode import BERTEncoder
import torch
import math
import numpy as np
import gc
from tqdm import tqdm
import time
from patent_utils import *
from model_utils import *

class BERTSimilarityTrainer(object):
    def __init__(self, patent_documents: list, tsd_documents: list, labels: list, threshold: float = 0.5, patience: int = 3):
        self.patent_documents = patent_documents
        self.tsd_documents = tsd_documents
        self.labels = labels
        self.bertEncoder = BERTEncoder(self.patent_documents, self.tsd_documents, self.labels, PatentCleaner())
        self.train_losses = []
        self.valid_losses = []
        self.avg_train_losses = []
        self.avg_valid_losses = [] 
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.loss_function = torch.nn.BCELoss()
        self.max_input_length = 512
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.bert_batch_size = 20
        self.config.num_labels = 1
        self.model = self.get_model()
        self.start_time = get_datetime()
        self.patience = patience
        self.early_stopping = EarlyStopping(self.model, self.patience, self.start_time, verbose=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     weight_decay=0,
                                     lr=5e-5)
        self.threshold = threshold
        self.batch_size = 8

    def validate(self):
        patent_representations, tsd_representations, correct_output, tsd_dict = self.bertEncoder.tokenize_test_data()
        gc.collect()

        self.model.eval()

        total_correct, total_predictions = 0, 0

        for i in range(0, patent_representations.shape[0], self.batch_size):

            batch_patent_tensors = patent_representations[i:i + self.batch_size].to(device=self.device)
            
            batch_tsd_tensors = tsd_representations[i:i + self.batch_size].to(device=self.device)

            batch_predictions = self.model(batch_patent_tensors,
                                      batch_tsd_tensors, 
                                      device=self.device
                                     )

            batch_correct_output = correct_output[i:i + self.batch_size].to(device=self.device)
            
            loss = self.loss_function(batch_predictions, batch_correct_output.view(batch_predictions.shape))
            self.valid_losses.append(loss.item())
        
            valid_loss = np.average(self.valid_losses)
            self.avg_valid_losses.append(valid_loss)

            if self.threshold == 0.5:
                num_correct = (batch_predictions.T[0].round() == batch_correct_output).sum().item()
            else:
                num_correct = (round_custom(batch_predictions.T[0], self.threshold) == batch_correct_output).sum().item()
            num_predictions = len(batch_predictions)
            total_correct += num_correct
            total_predictions += num_predictions
        
        valid_acc = total_correct/total_predictions * 100
#         print(f"\tTest accuracy={total_correct/total_predictions * 100:.2f}")
        
        self.valid_losses = []
        
        return valid_acc
        
    def train(self):
        patent_representations, tsd_representations, correct_output, tsd_dict = self.bertEncoder.tokenize_train_data()
        self.tsd_dict = tsd_dict
        gc.collect()
        total_correct, total_predictions = 0, 0
        early_stopping_counter = 0
        best_train_acc = 0.00

        for epoch in range(100):
            self.model.train()
            permutation = torch.randperm(patent_representations.shape[0])
            patent_representations = patent_representations[permutation]
            tsd_representations = tsd_representations[permutation]
            correct_output = correct_output[permutation]

            epoch = epoch
            epoch_loss = 0.0

            for i in tqdm(range(0, patent_representations.shape[0], self.batch_size), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
                batch_patent_tensors = patent_representations[i:i + self.batch_size].to(device=self.device)
                batch_tsd_tensors = tsd_representations[i:i + self.batch_size].to(device=self.device)

                batch_predictions = self.model(batch_patent_tensors,
                                          batch_tsd_tensors, 
                                          device=self.device
                                         )

                batch_correct_output = correct_output[i:i + self.batch_size].to(device=self.device)

                loss = self.loss_function(batch_predictions, batch_correct_output.view(batch_predictions.shape))
                self.train_losses.append(loss.item())
                
                train_loss = np.average(self.train_losses)
                self.avg_train_losses.append(train_loss)
                
                epoch_loss += float(loss.item())
                #self.log.info(batch_predictions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.threshold == 0.5:
                    num_correct = (batch_predictions.T[0].round() == batch_correct_output).sum().item()
                else:
                    num_correct = (round_custom(batch_predictions.T[0], self.threshold) == batch_correct_output).sum().item()
                num_predictions = len(batch_predictions)
                total_correct += num_correct
                total_predictions += num_predictions

            self.train_losses = []
            
            train_acc = total_correct/total_predictions * 100
#             if train_acc < best_train_acc:
#                 early_stopping_counter += 1
#             else:
#                 early_stopping_counter = 0
            epoch_loss /= int(patent_representations.shape[0] / self.batch_size)  # divide by number of batches per epoch
            valid_acc = self.validate()   
            print(f"Train: Epoch {epoch}, Training Loss={epoch_loss:4f}, Train accuracy={total_correct/total_predictions * 100:.2f}%, Test accuracy={valid_acc:.2f}%")
            self.early_stopping.record(valid_acc, epoch)
              
#             if early_stopping_counter == 3:
#                 break; 
    
            if self.early_stopping():
                print("Early stopping")
                break;
            
    def get_model(self):
        model = DocumentBert.from_pretrained('bert-base-uncased', config=self.config)
        model.freeze_bert_encoder()
        model.unfreeze_bert_encoder_last_layers()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(self.bert_doc_classification)
        model.to(device=self.device)
        return model
        
            
class DocumentBert(BertPreTrainedModel):

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBert, self).__init__(bert_model_config)
        self.bert_patent = BertModel(bert_model_config)
        self.bert_tsd = BertModel(bert_model_config)
        
        for param in self.bert_patent.parameters():
            param.requires_grad = False
            
        for param in self.bert_tsd.parameters():
            param.requires_grad = False
        
        self.bert_batch_size = self.bert_patent.config.bert_batch_size  
        self.dropout_patent = torch.nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.dropout_tsd = torch.nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        
        self.lstm_patent = torch.nn.LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.lstm_tsd = torch.nn.LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        
        self.output = torch.nn.Linear(bert_model_config.hidden_size*2, out_features=1)

    def forward(self, patent_batch: torch.Tensor, tsd_batch: torch.Tensor, device='cuda'):

        #patent
        bert_output_patent = torch.zeros(size=(patent_batch.shape[0],
                                              min(patent_batch.shape[1],self.bert_batch_size),
                                              self.bert_patent.config.hidden_size), dtype=torch.float, device=device)
        for doc_id in range(patent_batch.shape[0]):
            bert_output_patent[doc_id][:self.bert_batch_size] = self.dropout_patent(self.bert_patent(patent_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=patent_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=patent_batch[doc_id][:self.bert_batch_size,2])[1])
        output_patent, (_, _) = self.lstm_patent(bert_output_patent.permute(1,0,2))
        last_layer_patent = output_patent[-1]
        
        #tsd

        bert_output_tsd = torch.zeros(size=(tsd_batch.shape[0],
                                              min(tsd_batch.shape[1],self.bert_batch_size),
                                              self.bert_tsd.config.hidden_size), dtype=torch.float, device=device)
        for doc_id in range(tsd_batch.shape[0]):
            bert_output_tsd[doc_id][:self.bert_batch_size] = self.dropout_tsd(self.bert_tsd(tsd_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=tsd_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=tsd_batch[doc_id][:self.bert_batch_size,2])[1])
        output_tsd, (_, _) = self.lstm_tsd(bert_output_tsd.permute(1,0,2))
        last_layer_tsd = output_tsd[-1]
       
        x = torch.cat([last_layer_patent, last_layer_tsd], dim=1)
        prediction = torch.nn.functional.sigmoid(self.output(x))
        
        assert prediction.shape[0] == patent_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert_patent.parameters():
            param.requires_grad = False
        for param in self.bert_tsd.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert_patent.parameters():
            param.requires_grad = True
        for param in self.bert_tsd.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert_patent.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
        for name, param in self.bert_tsd.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert_patent.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
        for name, param in self.bert_tsd.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
                
    def predict(self, patent_claims, bertEncoder, tsd_representations):
        patent_representations, output = bertEncoder.encode_patents(patent_claims)

        patent_representations = patent_representations.to('cpu')
        tsd_representations = tsd_representations.to('cpu')
#         model.to('cpu')

        similarity_scores = []
        batch_size = 4
        gc.collect()
        torch.cuda.empty_cache()

        for i in range(0, patent_representations.shape[0], batch_size):

            batch_patent_tensors = patent_representations[i:i + batch_size].to(device='cpu')
            batch_tsd_tensors = tsd_representations[i:i + batch_size].to(device='cpu')

            batch_predictions = model(batch_patent_tensors,
                              batch_tsd_tensors, 
                              device='cpu'
                             )
            similarity_scores.extend(batch_predictions)
        
        
        similarity_scores = []