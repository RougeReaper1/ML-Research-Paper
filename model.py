from tensorflow.keras.saving import register_keras_serializable, load_model
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# -------------------------------------
# 🔹 EEG Data Preprocessing Functions 🔹
# -------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=-1)

def preprocess_eeg_data(eeg_data, fs=256):  # Assume 256 Hz sampling rate
    # Apply bandpass filtering to each signal
    LOWCUT = 0.5
    HIGHCUT = 50.0
    filtered_data = np.apply_along_axis(bandpass_filter, -1, eeg_data, lowcut=LOWCUT, highcut=HIGHCUT, fs=fs)
    return filtered_data

def normalize_eeg_data(eeg_data):
    mean = np.mean(eeg_data, axis=-1, keepdims=True)  # Mean along the last axis (channel)
    std = np.std(eeg_data, axis=-1, keepdims=True)  # Standard deviation along the last axis (channel)
    normalized_data = (eeg_data - mean) / (std + 1e-7)  # Avoid division by zero by adding small value
    return normalized_data

def interpolate_missing_data(eeg_data):
    # Interpolate missing data (NaNs) using linear interpolation
    for i in range(eeg_data.shape[0]):
        nan_indices = np.isnan(eeg_data[i])
        if np.any(nan_indices):
            non_nan_indices = np.where(nan_indices)[0]
            interpolator = interp1d(non_nan_indices, eeg_data[i, non_nan_indices], kind='linear', fill_value="extrapolate")
            eeg_data[i, nan_indices] = interpolator(np.where(nan_indices)[0])
    return eeg_data

def preprocess_eeg_data_pipeline(eeg_data, fs=256):
    # Bandpass Filtering
    filtered_data = preprocess_eeg_data(eeg_data, fs)

    # Interpolate Missing Data
    clean_data = interpolate_missing_data(filtered_data)

    # Normalize Data
    normalized_data = normalize_eeg_data(clean_data)

    return normalized_data
# -------------------------------------
# 🔹 Improved Attention Mechanism 🔹
# -------------------------------------
class ScaledDotProductAttention(layers.Layer):
    def call(self, query, key, value):
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, value)

class MultiHeadAttention(layers.Layer):
    def __init__(self, channels, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.heads = num_heads
        self.channels = channels
        self.q_dense = layers.Dense(channels)
        self.k_dense = layers.Dense(channels)
        self.v_dense = layers.Dense(channels)
        self.output_dense = layers.Dense(channels)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.q_dense(inputs)
        key = self.k_dense(inputs)
        value = self.v_dense(inputs)

        query = tf.reshape(query, (batch_size, -1, self.heads, self.channels // self.heads))
        key = tf.reshape(key, (batch_size, -1, self.heads, self.channels // self.heads))
        value = tf.reshape(value, (batch_size, -1, self.heads, self.channels // self.heads))

        attention = ScaledDotProductAttention()
        out = attention(query, key, value)

        out = tf.reshape(out, (batch_size, -1, self.channels))
        return self.output_dense(out)

# -------------------------------------
# 🔹 Improved CNN Model 🔹
# -------------------------------------
def create_cnn_model(input_shape=(23, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Simplified CNN layers
    x = layers.Conv2D(64, (5,5), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Removed one convolutional layer to simplify
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Flatten and pass through attention mechanism
    x = layers.Reshape((-1, 256))(x)
    x = MultiHeadAttention(channels=256)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers with dropout for regularization
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)  # Adjusted dropout to help regularization
    x = layers.Dense(256, activation='relu')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model
# -------------------------------------
# 🔹 Improved BiLSTM Model 🔹
# -------------------------------------
def create_bilstm_model(input_shape=(23, 256)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)  # Added layer
    x = layers.Dense(128, activation='relu')(x)
    model = models.Model(inputs, x)
    return model

# -------------------------------------
# 🔹 Improved Graph Attention Network (GAT) 🔹
# -------------------------------------
class MultiLayerGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * 8, out_channels, heads=8, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)

# -------------------------------------
# 🔹 EEG Graph Construction 🔹
# -------------------------------------
def build_graph_from_eeg_data(eeg_data_np, threshold=0.5):
    if eeg_data_np.ndim != 2:
        eeg_data_np = eeg_data_np.reshape(-1, eeg_data_np.shape[-1])
    correlation_matrix = np.corrcoef(eeg_data_np)
    edges = []
    num_channels = eeg_data_np.shape[0]
    for i in range(num_channels):
        for j in range(i+1, num_channels):
            if np.abs(correlation_matrix[i, j]) > threshold:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(eeg_data_np, dtype=torch.float32)
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

# -------------------------------------
# 🔹 Hybrid Model: CNN + BiLSTM + GAT + Attention 🔹
# -------------------------------------
def create_hybrid_model(input_shape=(23, 256, 1), eeg_data_sample=None):
    if eeg_data_sample is None:
        raise ValueError("eeg_data_sample must be provided with EEG data!")

    # CNN branch
    cnn_inputs = layers.Input(shape=input_shape)
    cnn_model = create_cnn_model(input_shape)
    cnn_features = cnn_model(cnn_inputs)  # Expected shape: (batch_size, 256)

    # BiLSTM branch
    bilstm_inputs = layers.Input(shape=(23, 256))
    bilstm_model = create_bilstm_model((23, 256))
    bilstm_output = bilstm_model(bilstm_inputs)
    bilstm_features = layers.GlobalAveragePooling1D()(bilstm_output)  # Expected shape: (batch_size, 128)

    # Concatenate the features from CNN and BiLSTM
    combined = layers.concatenate([cnn_features, bilstm_features], axis=-1)  # Shape: (batch_size, 384)

    # Graph attention branch (unchanged)
    graph_data = build_graph_from_eeg_data(eeg_data_sample)
    gat_model = MultiLayerGAT(in_channels=256, hidden_channels=128, out_channels=64)
    with torch.no_grad():
        gat_output = gat_model(graph_data.x, graph_data.edge_index)
    gat_output_tf = tf.convert_to_tensor(gat_output.numpy(), dtype=tf.float32)
    graph_features = tf.reduce_mean(gat_output_tf, axis=0, keepdims=True)  # Shape: (1, 64)

    # Tile graph_features to match the batch size using a Lambda layer
    graph_features_tiled = layers.Lambda(
        lambda args: tf.tile(args[1], [tf.shape(args[0])[0], 1])
    )([combined, graph_features])  # Now shape: (batch_size, 64)

    # Final concatenation with graph features
    final_combined = layers.concatenate([combined, graph_features_tiled], axis=-1)  # Shape: (batch_size, 448)
    x = layers.Dense(256, activation='relu')(final_combined)
    x = layers.Dense(128, activation='relu')(x)
    final_output = layers.Dense(3, activation='softmax')(x)

    hybrid_model = models.Model(inputs=[cnn_inputs, bilstm_inputs], outputs=final_output)
    return hybrid_model

# -------------------------------------
# 🔹 Load and Preprocess EEG Data 🔹
# -------------------------------------
def load_eeg_data():
    train_data = np.load('/content/drive/MyDrive/dataset.zip (Unzipped Files)/eeg-seizure_train.npz')
    val_data = np.load('/content/drive/MyDrive/dataset.zip (Unzipped Files)/eeg-predictive_val_balanced.npz')
    test_data = np.load('/content/drive/MyDrive/dataset.zip (Unzipped Files)/eeg-seizure_test.npz')

    X_train, y_train = train_data['train_signals'], train_data['train_labels']
    X_val, y_val = val_data['val_signals'], val_data['val_labels']
    X_test = test_data['test_signals']

    X_train = X_train.astype('float32') / np.max(X_train)
    X_val = X_val.astype('float32') / np.max(X_val)
    X_test = X_test.astype('float32') / np.max(X_test)

    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=3)
    y_val = to_categorical(y_val, num_classes=3)

    return X_train, y_train, X_val, y_val, X_test

# -------------------------------------
# 🔹 Main: Training the Model 🔹
# -------------------------------------
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def save_model(hybrid_model, filename="seizure_prediction_model.keras"):
    hybrid_model.save(filename)
    print(f"Model saved successfully as {filename}")

def load_and_evaluate_model(model_path, X_test):
    model = load_model(model_path)
    print("Model loaded successfully. Evaluating on test set...")

    test_loss, test_accuracy = model.evaluate([X_test, X_test.squeeze()])
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    return test_loss, test_accuracy

def main():
 import matplotlib.pyplot as plt

def main():
    # Load and preprocess EEG data
    X_train, y_train, X_val, y_val, X_test = load_eeg_data()
    eeg_sample = X_train[0].squeeze()  # Select a sample for graph construction

    hybrid_model = create_hybrid_model(input_shape=(23, 256, 1), eeg_data_sample=eeg_sample)

    print("No pre-trained model found, training from scratch...")

    hybrid_model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
  
    from tensorflow.keras.optimizers.schedules import ExponentialDecay

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    hybrid_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = hybrid_model.fit(
        [X_train, X_train.squeeze()], y_train,
        validation_data=([X_val, X_val.squeeze()], y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stop]
    )
l
    save_model(hybrid_model)

    # Plot validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()
    load_and_evaluate_model("seizure_prediction_model.keras", X_test)
if __name__ == "__main__":
    main()
