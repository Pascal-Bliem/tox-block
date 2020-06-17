"""This module provides functions for prediction."""

# general data handling and computation
import pandas as pd
import numpy as np
# python standard modules
from typing import List, Dict, Union
import logging
# TensorFlow
from tensorflow.keras.models import load_model
# ignore TensorFlow's deprecation warnings
import warnings
warnings.simplefilter("ignore")
# saving / loading python objects
import joblib
# import modules from this package
from tox_block.config import config
from tox_block.data_processing.data_handling import validate_input_data
from tox_block import __version__ as _version

_logger = logging.getLogger(__name__)

_model = load_model(config.MODEL_PATH)
_encoder = joblib.load(config.ENCODER_PATH)

def _rescale_proba(p: Union[float, pd.Series], 
                   min: float, 
                   max: float) -> Union[float, pd.Series]:
    """Performs min-max-scaling on a probability value"""
    return np.abs((p - min) / (max - min))


def make_predictions(input_texts: List[str], rescale: bool = True) -> Dict:
    """Get probability predictions for toxicity categories on texts.
    
    Args:
        input_texts: List of strings representing the texts for which toxicity
                     probabilities should be predicted.
            rescale: Bool stating weather or not the predicted probabilities for the
                     individual categories should be rescaled to be on similar scales.
          
    Returns:
        predictions: Dict of dicts with the original text and predicted 
                     probabilities for the 6 toxicity categories.
    """
    
    # validate input data
    input_texts = validate_input_data(input_texts)

    # encode the text and make predictions
    texts = pd.DataFrame(input_texts, columns=["text"])
    encoded_texts = _encoder.transform(texts)
    predictions = pd.DataFrame(_model.predict(encoded_texts), 
                               columns=config.LIST_CLASSES)
    
    # rescale the predicted probabilities
    if rescale:
        rsp = config.RESCALE_PROBA
        for col in predictions.columns:
            predictions[col] = _rescale_proba(p=predictions[col], 
                                              min=rsp[col][0], 
                                              max=rsp[col][1])
    
    # log the prediction
    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {input_texts} "
        f"Predictions: {predictions}"
    )

    predictions = pd.concat([texts.text,predictions], axis=1)
    
    return predictions.T.to_dict()

def make_single_prediction(input_text: str, rescale: bool = True) -> Dict:
    """Get probability predictions for toxicity categories on a single text.
    
    Args:
        input_text: A string representing the text for which toxicity
                    probabilities should be predicted.
           rescale: Bool stating weather or not the predicted probabilities for the
                    individual categories should be rescaled to be on similar scales.

    Returns:
        prediction: Dict with the original text and predicted 
                    probabilities for the 6 toxicity categories.
    """
    return make_predictions(input_texts=[input_text], 
                            rescale=rescale)[0]