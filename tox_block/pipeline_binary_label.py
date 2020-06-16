
# modules from this package
from tox_block.config import config
from tox_block.data_processing import preprocessors as pp
from tox_block.data_processing import data_handling as dh
from tox_block.model import lstm_binary_label

def run_training(save_result: bool = True):
    """Train a LSTM Neural Network."""

    # load data
    X, y = dh.load_data_binary_labels(config.TRAINING_DATA_FILE)
    
    # remove undesired content and stop words
    X = HyperlinkUsernameIPStopwordRemover().transform(X)
    
    # tokenize text and turn it into numerical sequences
    ts = pp.TokenSequencer()
    X = ts.fit_transform(X)
    
    # train the model
    model = lstm_binary_label.get_model()
    history = model.fit(X, y,
                        batch_size=config.BATCH_SIZE,
                        validation_split=config.VALIDATION_SPLIT,
                        class_weight = config.CLASS_WEIGHT,
                        epochs=config.EPOCHS,
                        verbose=1,
                        callbacks=lstm_binary_label.callbacks_list)

    if save_result:
        joblib.dump(ts, config.ENCODER_PATH)
        model.save(config.MODEL_PATH)


if __name__ == '__main__':
    run_training(save_result=True)
