"""A collection of transformers for data preprocessing."""

# general data handling and computation
import pandas as pd
import numpy as np
# python standard modules
from typing import Union, Tuple
import re
# scikit learn
from sklearn.base import BaseEstimator, TransformerMixin
# Tensorflow / Keras
from tensorflow.keras.preprocessing import text, sequence
# modules for this package
from tox_block.config import config



class HyperlinkUsernameIPStopwordRemover(BaseEstimator, TransformerMixin):
    """A transformer that removes Hyperlinks, Usernames, 
    IP addresses from comment text"""
    def __init__(self, re_stopword_string: str = config.RE_STOPWORD_STRING):
        self.re_stopword_string = re_stopword_string


    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """Does nothing"""
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> pd.DataFrame:
        """Removes Hyperlinks, Usernames, IP addresses with regular expressions"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        def remove(x: str) -> str:
            x = x.lower()
            # remove username, ip, hyperlinks
            x = re.sub("\\n"," ",x)
            x = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",x)
            x = re.sub("http://.*\..*","",x)
            x = re.sub("\[\[User.*\|","",x)
            # remove stop words and strip white spaces
            x = re.sub(self.re_stopword_string,"",x,flags=re.IGNORECASE)
            x = re.sub("\s+"," ",x)
            return x
        
        # apply the remover
        X = X.iloc[:,0].apply(remove).fillna("nanana").to_frame()
        return X

class TokenSequencer(BaseEstimator, TransformerMixin):
    """A transformer that fits a tokenizer on training text with a
    max vocabulary, converts texts to numerical sequences accordingly,
    and pads the sequences with zeroes up to max length."""

    def __init__(self, 
                 max_features: int = config.MAX_FEATURES, 
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH):
        self.max_features = max_features
        self.max_sequence_length = max_sequence_length
        self.tokenizer = text.Tokenizer(num_words=self.max_features)
    
    def fit(self, X: pd.DataFrame, y = None):
        """Fit a tokenizer on the training texts"""
        self.tokenizer.fit_on_texts(list(X.iloc[:,0].fillna("nanana").values))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Turn texts into numerical sequences and pad them with 
        zeroes to max_sequence_length"""
        
        # turn text into sequences
        sequences = (self.tokenizer
                     .texts_to_sequences(X.iloc[:,0].fillna("nanana").values))

        return sequence.pad_sequences(sequences, 
                                      maxlen=self.max_sequence_length)
