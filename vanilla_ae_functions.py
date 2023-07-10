from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow import keras  
from keras.models import Model
from keras import Input
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.engine.functional import Functional
import os


def load_input(dataset_path: str) -> np.ndarray:
    """
    Input:
        dataset_path: Path to the dataset file.
    Purpose:
        Load input data from the specified dataset path.
    Returns:
        Array containing the loaded input data.
    """
    model = KeyedVectors.load_word2vec_format(dataset_path, binary=True)
    all_vectors = model.vectors
    return all_vectors


def normalise_pos_input(unnormalised_dataset: np.ndarray) -> np.ndarray:
    """
    Input:
        unnormalised_dataset: Array containing the unnormalized input dataset.
    Purpose:
        Normalise and make input dataset positive.
    Returns:
        Array containing the normalised, positive input dataset.
    """
    # Figure out the min value, so to make all vectors positive
    min_x_unormalised = min([min(i) for i in unnormalised_dataset])
    pos_x_train = unnormalised_dataset+abs(min_x_unormalised)
    
    # Normalise the data by dividing by the max of the positive data
    max_pos_x = max([max(i) for i in pos_x_train])
    x_train = pos_x_train/max_pos_x
    return x_train


def vanilla_ae(x_train: np.ndarray, original_dim: int = 300, hl_dim: int = 2, epochs: int = 25, batch_size: int = 128) -> Functional:
    """
    Args:
        input: Keras input tensor.
        x_train: Input data for training the autoencoder.
        original_dim: Dimension of the input data.
        hl_dim: Dimension of the hidden layer in the autoencoder.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    Purpose: 
        Train and build a vanilla autoencoder.
    Returns:
        Trained vanilla autoencoder model.
    """
    input = Input(shape=(original_dim,), name='Encoder_Input_Layer')

    encoded = Dense(hl_dim, activation='relu')(input)
    decoded = Dense(original_dim, activation='relu')(encoded)
    
    autoencoder = Model(input, decoded)

    autoencoder.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

    return autoencoder