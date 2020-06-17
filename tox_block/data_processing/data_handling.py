"""A collection of functions for data handling."""
# general data handling and computation
import pandas as pd
import numpy as np
# python standard modules
from typing import Union, Tuple, List
import re
# modules for this package
from tox_block.config import config
from tox_block.data_processing import preprocessors as pp

def load_data_binary_labels(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data from CSV file and returns features (X) and 
    only binary labels meaning (any kind of) toxic or not"""
    df = pd.read_csv(path)
    X = df.comment_text.to_frame()
    y = df[config.LIST_CLASSES].max(axis=1).to_frame(name="toxic")
    return X, y

def load_data_multi_labels(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data from CSV file and returns features (X) and all 6 labels (y)"""
    df = pd.read_csv(path)
    X = df.comment_text.to_frame()
    y = df[config.LIST_CLASSES]
    return X, y

def get_embedding_matrix(token_sequencer: pp.TokenSequencer, 
                         embedding_file: str = config.EMBEDDING_FILE,
                         max_features: int = config.MAX_FEATURES,
                         embedding_size: int = config.EMBEDDING_SIZE) -> np.ndarray:
    """Reads word embeddings from a file and applies them to a tokenizer's word index"""
    
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    # get embeddings form file
    embeddings_index = dict(get_coefs(*str(o,encoding="utf-8").strip().split()) 
                            for o in open(embedding_file, "rb"))
    all_embeddings = np.stack(embeddings_index.values())
    word_index = token_sequencer.tokenizer.word_index
    num_words = min(max_features, len(word_index))
    
    # words not contained in the GloVe embedding will be initialized randomly
    # with mean and standard deviation matching the embedding vectors
    embeddings_mean, embeddings_std = all_embeddings.mean(), all_embeddings.std()
    embedding_matrix = np.random.normal(embeddings_mean, 
                                        embeddings_std, 
                                        (num_words, embedding_size))
    
    # for the words present in the embedding, set their vectors in the embeding matrix
    for word, i in word_index.items():
        if i >= max_features: 
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
            
def validate_input_data(input: Union[str, List[str]]) -> Union[str, List[str]]:
    """Validates if the input is a string, list of strings and not empty"""
    
    # check if input is list
    if isinstance(input,list):
        # check if all list items are non-empty strings
        for i, item in enumerate(input):
            if not isinstance(item,str):
                raise ValueError(f"The list item at position {i} is not a string.")
            if item == "":
                raise ValueError(f"The list item at position {i} is an empty string.")
        return input
    # check if input is non-empty string
    elif isinstance(input,str):
        if input == "":
            raise ValueError("Passed an empty string.")
        return input
    else:
        raise ValueError("The passed object is neither a string nor a list of strings.")