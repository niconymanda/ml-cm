from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
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


def split_training_validation_test(dataset: np.ndarray):
    """
    Input:
        dataset: The input dataset as a NumPy array. In our case the vectors of the word2vec dataset.
    Purpose:
        Splits the given dataset into training, validation, and test sets.
    Returns:
        A tuple containing the training, validation, and test sets.

    """
    x_train, x_remaining = train_test_split(dataset, test_size=0.4, random_state=42)
    x_val, x_test = train_test_split(x_remaining, test_size=0.5, random_state=42)
    return x_train, x_val, x_test


def normalise_pos_input(unnormalised_dataset: np.ndarray) -> np.ndarray:
    """
    Input:
        unnormalised_dataset: Array containing the unnormalised input dataset.
    Purpose:
        Normalise and make input dataset positive.
    Returns:
        Array containing the normalised, positive input dataset.
    """
    scaler = MinMaxScaler()
    scaler.fit(unnormalised_dataset)
    normalised_data = scaler.transform(unnormalised_dataset)
    return normalised_data


def vanilla_ae(x_train: np.ndarray, activation_ecoder = 'relu', activation_decoder = 'sigmoid', original_dim: int = 300, hl_dim: int = 2, epochs: int = 25, batch_size: int = 128) -> Functional:
    """
    Input:
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

    encoded = Dense(hl_dim, activation=activation_ecoder)(input)
    decoded = Dense(original_dim, activation=activation_decoder)(encoded)
    
    autoencoder = Model(input, decoded)

    autoencoder.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

    return autoencoder

def calculate_mse(x_original: np.ndarray, x_pred: np.ndarray) -> int:
    """
    Input:
        x_original: The original data
        x_pred: The predicted data (i.e. reconstructed data from the autoencoder) 
    Purpose:
        Calculates the Mean Squared Error (MSE) between the original data and the predicted data.
    Returns:
        the MSE
    """
    mse = np.mean(np.square(x_original - x_pred))
    return mse


def perform_pca(n_comps: int, data_to_reduce: np.ndarray) -> np.ndarray:
    """
    Input:
        n_comps: The number of components to keep in the reduced data.
        data_to_reduce: The input data to be reduced in dimensionality.
    Purpose:
        Perform Principal Component Analysis (PCA) on the input data.
    Returns:
        The data transformed to the PCA space.
    """
    pca = PCA(n_components=n_comps)
    pca.fit(data_to_reduce)
    pca_data = pca.transform(data_to_reduce)

    return pca_data