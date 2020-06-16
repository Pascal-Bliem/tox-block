"""A bidirectional LSTM model with multi labels (6 types of toxicity)"""

# general data handling and computation
import pandas as pd
import numpy as np
# TensorFlow / Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# modules for this package
from tox_block.config import config


def get_model(embedding_matrix: np.ndarray = None,
              embedding_size: int = config.EMBEDDING_SIZE,
              max_sequence_length: int = config.MAX_SEQUENCE_LENGTH,
              max_features: int = config.MAX_FEATURES,
              dropout: float = config.DROPOUT,
              num_lstm_units: int = config.NUM_LSTM_UNITS,
              num_dense_units: int = config.NUM_DENSE_UNITS,
              learning_rate: float = config.LEARNING_RATE):
    """Returns a bidirectional LSTM model"""
    
    inp = Input(shape=(max_sequence_length, ))
    if not embedding_matrix is None:
        x = Embedding(max_features, 
                      embedding_size, 
                      weights=[embedding_matrix])(inp)
    else:
        x = Embedding(max_features, 
                      embedding_size)(inp)
    x = Bidirectional(LSTM(num_lstm_units, 
                           return_sequences=True, 
                           dropout=dropout, 
                           recurrent_dropout=dropout))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense_units, activation="relu")(x)
    x = Dropout(rate=dropout)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(Adam(lr=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model

# callbacks for training
checkpoint = ModelCheckpoint(config.TRAINED_MODEL_DIR + "/checkpoint.h5", 
                             monitor="val_loss", 
                             verbose=1, 
                             save_best_only=True, 
                             mode="min")

early_stop = EarlyStopping(monitor="val_loss", 
                           mode="min", 
                           patience=2,
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                              factor=0.5,
                              patience=1,
                              verbose=1,
                              mode="min",
                              min_lr=0.00001)

callbacks_list = [checkpoint, early_stop, reduce_lr]

if __name__ == '__main__':
    model = get_model()
    model.summary()