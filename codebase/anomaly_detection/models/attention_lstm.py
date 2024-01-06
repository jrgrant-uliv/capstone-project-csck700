import math

import numpy as np
from codebase.anomaly_detection.models.base_classifier import Classifier
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import (LSTM, Attention, Concatenate, Dense, Dropout,
                          GaussianNoise, GlobalAveragePooling1D, Input,
                          Masking)
from keras.models import Model


class LSTMAttentionClassifier(Classifier):
    """
    A classifier that uses LSTM for anomaly detection.

    Attributes:
        feature_dim (int): The number of features in the input data.
        sequence_length (int): The length of the input sequence.
        num_classes (int): The number of classes in the output.
        attention_units (int): The number of units in the attention layer.
        lstm_units (int): The number of units in the LSTM layer.
        dropout_rate (float): The dropout rate to use.
        model (tf.keras.Model): The LSTM model.
    """

    def __init__(self, model_name, feature_dim, sequence_length, vocab_size, lstm_units=64, model_artefact_dir="artefacts/models/"):
        """
        Initializes the LSTM model.

        Args:
        - feature_dim (int): The number of features in the input data.
        - sequence_length (int): The length of the input sequence.
        - num_classes (int): The number of classes in the output.
        - attention_units (int): The number of units in the attention layer.
        - lstm_units (int): The number of units in the LSTM layer.
        """
        super().__init__(model_name=model_name, model_artefact_dir=model_artefact_dir)
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.dropout_rate = 0.2
        self.recurrent_dropout = 0.2
        self.build_model_summary()

    def build_model(self):
        """
        Builds a Keras model for anomaly detection using LSTM.

        Returns:
            A Keras model for anomaly detection using LSTM.
        """

        output_dim = 1  # binary classification
        inputs = Input(
            shape=(self.sequence_length, self.feature_dim), name="sequence_input"
        )
        inputs = Masking(mask_value=float(0))(inputs)
        inputs = GaussianNoise(0.1)(inputs)
        lstm_out = LSTM(units=self.lstm_units, return_sequences=True)(inputs)
        attention = Attention()([lstm_out, lstm_out])
        attended_lstm = Concatenate(axis=-1)([lstm_out, attention])
        flat = GlobalAveragePooling1D()(attended_lstm)
        flat = Dropout(self.dropout_rate)(flat)
        output = Dense(units=output_dim, activation='sigmoid')(flat)
        model = Model(inputs=inputs, outputs=output)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=5, batch_size=64, patience=2):
        """
        Trains the LSTM model on the given training data.

        Args:
            x_train (numpy.ndarray): Input data for the model.
            y_train (numpy.ndarray): Target data for the model.
            s_train (numpy.ndarray): Additional input data for the model.
            epochs (int): Number of epochs to train the model for.
            batch_size (int): Batch size to use during training.

        Returns:
            A `History` object containing training metrics.
        """

        lr_scheduler = LearningRateScheduler(self._lr_schedule)

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=patience,
                                       verbose=1,
                                       mode='max',
                                       restore_best_weights=True)

        class_counts = np.unique(y_train, return_counts=True)[1]
        ratio = class_counts[0]/class_counts[1]

        class_weights = {0: 1, 1: ratio}

        history = self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, class_weight=class_weights,
                                 batch_size=batch_size, callbacks=[early_stopping, lr_scheduler])
        self.save_training_plots(self.model_name, history)
        return history

    def _lr_schedule(self, epoch):
        """
        Returns the learning rate for a given epoch based on a predefined schedule.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The learning rate for the given epoch.
        """
        initial_lr = 0.01
        drop = 0.50
        epochs_drop = 10
        lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr
