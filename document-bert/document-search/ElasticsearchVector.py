from ParagraphExtraction import ParagraphExtractor
from docx import Document
import tensorflow as tf
import tensorflow_hub as hub
from elasticsearch import Elasticsearch
import pprint
import re
import clean
from Embeddings import ParagraphEmbeddings, BERTEmbeddings

parembeddings = ParagraphEmbeddings()

class ESVectorIndex(object):
    def __init__(self, index_name, vector_dims=512, host="http://localhost", port=9200, encoder='bert'):
        assert encoder in ['bert', 'universal-sentence-encoder']
        self.index_name = index_name
        if encoder == 'bert':
            self.parembeddings = BERTEmbeddings()
            self.vector_dims = 768
        elif encoder == 'universal-sentence-encoder':
            self.parembeddings = UniversalSentenceEmbeddings()
            self.vector_dims = vector_dims
        self.host = host
        self.port = port
        self.es = Elasticsearch(HOST=self.host, PORT=self.port)
        
    def add_doc_vec(self, text):
        text_vector = self.parembeddings.get_vec(text)
        doc = {
            "Document_name": text,
            "Doc_vector": text_vector
        }

        res = self.es.index(index = self.index_name, body = doc)

    def create_index(self):
        settings = {"settings": {"number_of_shards": 2,"number_of_replicas": 1},
                    "mappings": {"dynamic": "true", 
                                 "_source": {"enabled": "true"},
                                 "properties": {"Document_name": {"type": "text"},
                                                "Doc_vector": {"type": "dense_vector", "dims": self.vector_dims}
                                               }
                                }
                    }

        
        self.es.indices.create(index=self.index_name, ignore=400, body=settings)

    def delete_index(self):
        if self.es.indices.exists(self.index_name):
            self.es.indices.delete(index=self.index_name, ignore=400)

    def add_doc_vecs(self, paragraphs):
        for paragraph in paragraphs:
            print(paragraph)
            self.add_doc_vec(text=paragraph)
    
    def search(self, query, print_response=True):
#         User_Query_Vector = self.parembeddings.get_vec("Apparatus and method for header decompression Vorrichtung und Verfahren für Kopfdekomprimierung Appareil et procédé permettant la décompression d’en-têtes',")
        User_Query_Vector = self.parembeddings.get_vec(query)

        script_query = {"script_score": {"query": {"match_all": {}},
                                         "script": {"source": "cosineSimilarity(params.query_vector, doc['Doc_vector']) ",
                                                    "params": {"query_vector": User_Query_Vector }
                                                    }
                                        }
                       }

        response = self.es.search(index=self.index_name,body={"size": 10, "query": script_query,
                                                          "_source": {"includes": ["Document_name"]}
                                                         }
                                  )
        if print_response == True:
            pprint.pprint(response)
        
        return response
        