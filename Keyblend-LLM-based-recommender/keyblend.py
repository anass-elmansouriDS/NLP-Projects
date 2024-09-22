from sentence_transformers import SentenceTransformer
import emoji
import spacy
import re
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
import numpy as np
import pickle

class KeyBlender :
    """
    This is the class that allows for easy usage of the keyblender recommender.
    """
    def __init__(self,model="/teamspace/studios/this_studio/kaggle/input/all-minilm-l6-v2-fine-tuned-model/kaggle/working/All-MiniLM-L6-V2-model") :
        self.model=SentenceTransformer(model)
        with open("/teamspace/uploads/prod_names_updated.pkl", "rb") as f :
            self.products=pickle.load(f)
        #self.products=products
        self.product_embeddings=self.model.encode(list(self.products.keys()))
      
      
    def create_custom_tokenizer(self) :
        self.nlp=spacy.load("en_core_web_sm")
        # Create a new tokenizer using the default tokenizer's settings
        tokenizer = self.nlp.tokenizer
        
        # Add custom rules to handle hyphenated words or other specific cases
        # Example: Ensure hyphenated words are treated as a single token
        tokenizer.add_special_case("t-shirt", [{ORTH: "t-shirt"}])
        
        # Apply custom tokenizer
        self.nlp.tokenizer = tokenizer
        
    def process_content(self,content) :
        self.content=content.lower()
        #remove emojs
        self.content=emoji.replace_emoji(self.content, replace='')
        #remove stopwords and lemmatize
        self.create_custom_tokenizer()
        doc=self.nlp(self.content)
        self.content=" ".join([token.lemma_ for token in doc])
        #remove symbols
        self.content=re.sub(r'[^a-zA-Z0-9\s-]', '', self.content)
        #remove non-meaningful words
        queries=[query.split(" ") for query in list(self.products.keys())]
        prod_keywords=[keyword for query in queries for keyword in query]
        self.content=" ".join([word for word in self.content.split(" ") if word in prod_keywords])

    def embedd(self) :
        self.content_embedding=self.model.encode(self.content)  

    def cosine_similarity(self,vector,matrix) :
        # Normalize the vector
        vector_norm = vector / np.linalg.norm(vector)
        
        # Normalize each row of the matrix
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        
        # Compute the cosine similarity between the vector and each row of the matrix
        cos_sim = np.dot(matrix_norm, vector_norm)
        
        return cos_sim
    
    def recommend(self,content) :
        scores={}
        self.process_content(content)
        self.embedd()
        scores_mat=self.cosine_similarity(self.content_embedding,self.product_embeddings)
        prod_names=list(self.products.values())
        for i,prod in enumerate(self.product_embeddings) :
            scores[prod_names[i]]=scores_mat[i]
        return dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))    