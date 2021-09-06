import tensorflow as tf
import tensorflow_hub as hub
from elasticsearch import Elasticsearch
import pprint
import torch
from transformers import BertTokenizer, BertModel

class ParagraphEmbeddings(object):
    def __init__(self):
        """Downloads pretrained model from tensorflow hub
        """
        print("Downloading pre-trained embeddings from tensorflow hub…")

    def get_vec(self, text):
        pass
    
class UniversalSentenceEmbeddings(ParagraphEmbeddings):
    def __init__(self):
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def embed_text(self, text):
        vectors = self.embed(text)
        return [vector.tolist() for vector in vectors]

    def get_vec(self, text):
        text_vector = self.embed_text([text])[0]
        return text_vector
        
class BERTEmbeddings(ParagraphEmbeddings):
    def __init__(self):
        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states = True, 
                                          )
        # Put the model in "evaluation" mode
        self.model.eval()
        
    def tokenize(self, text):
        marked_text = "[CLS] " + text + " [SEP]"

        tokenized_text = self.tokenizer.tokenize(marked_text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
            
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        return tokens_tensor, segments_tensors
        
    def get_vec(self, text):    
        
        tokens_tensor, segments_tensors = self.tokenize(text)

        with torch.no_grad():

            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        
        token_vecs = hidden_states[-2][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = list(sentence_embedding.numpy())
        
        return sentence_embedding