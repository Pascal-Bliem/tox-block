"""This is the training pipeline"""

# import modules from this package
from tox_block.config import config
from tox_block.data_processing import preprocessors as pp
from tox_block.data_processing import data_handling as dh
from tox_block.model import lstm_multi_label
from tox_block import __version__ as _version
# saving python objects
import joblib

import logging
_logger = logging.getLogger(__name__)

def run_training(save_result: bool = True, apply_remover: bool = False):
    """Train a LSTM Neural Network."""

    # load data
    X, y = dh.load_data_multi_labels(config.MERGED_DATA_FILE)
    
    if apply_remover:
        # remove undesired content and stop words
        X = HyperlinkUsernameIPStopwordRemover().transform(X)
    
    # tokenize text and turn it into numerical sequences
    ts = pp.TokenSequencer()
    X = ts.fit_transform(X)
    
    # get the GloVe embedding matrix
    em = dh.get_embedding_matrix(ts)

    # train the model
    model = lstm_multi_label.get_model(embedding_matrix=em)
    history = model.fit(X, y,
                        batch_size=config.BATCH_SIZE,
                        validation_split=config.VALIDATION_SPLIT,
                        epochs=config.EPOCHS,
                        verbose=1,
                        callbacks=lstm_multi_label.callbacks_list)
    
    # save the pipeline components
    if save_result:
        _logger.info(f"saving model version: {_version}")
        joblib.dump(ts, config.ENCODER_PATH)
        model.save(config.MODEL_PATH)
        joblib.dump(history.history, config.TRAINING_HISTORY_PATH)


if __name__ == '__main__':
    # run the training
    run_training(save_result=True)
