import math

import numpy as np
from codebase.anomaly_detection.models.base_classifier import Classifier
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import (Attention, Bidirectional, Concatenate, Dense,
                          Dropout, Embedding, GaussianNoise,
                          GlobalAveragePooling1D, Input, Masking)
from keras.models import Model
from tcn import TCN


class TCNSentimentclassifier(Classifier):
    """
    TCN Sentiment Classifier.

    This class represents a sentiment classifier based on Temporal Convolutional Networks (TCN).
    It inherits from the `Classifier` base class.

    Args:
        model_name (str): The name of the model.
        input_size (int): The size of the input.
        input_dim (int): The dimension of the input.
        positional_embedding_matrix (ndarray): The positional embedding matrix.
        word_embedding_matrix (ndarray): The word embedding matrix.
        tcn_units (int, optional): The number of units in the TCN layer. Defaults to 64.
        embedding_output_dim (int, optional): The output dimension of the embeddings. Defaults to 64.
        tcn_filters (int, optional): The number of filters in the TCN layer. Defaults to 64.
        tcn_kernel_size (int, optional): The kernel size of the TCN layer. Defaults to 7.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.5.
        num_classes (int, optional): The number of classes. Defaults to 1.
        prediction_threshold (float, optional): The prediction threshold. Defaults to 0.5.
        model_artefact_dir (str, optional): The directory for model artifacts. Defaults to "artefacts/models/".
    """

    def __init__(self, model_name, input_size, input_dim, positional_embedding_matrix, word_embedding_matrix, tcn_units=64, embedding_output_dim=64, tcn_filters=64, tcn_kernel_size=7, dropout_rate=0.5, num_classes=1, prediction_threshold=0.5, model_artefact_dir="artefacts/models/"):

        super().__init__(model_name=model_name, prediction_threshold=prediction_threshold,
                         model_artefact_dir=model_artefact_dir)
        self.input_size = input_size
        self.input_dim = input_dim
        self.positional_encoding_matrix = positional_embedding_matrix
        self.word_embedding_matrix = word_embedding_matrix
        self.tcn_units = tcn_units
        self.embedding_output_dim = embedding_output_dim
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.build_model_summary()

    def build_model(self):
        """
        Build the TCN sentiment classification model.

        Returns:
            keras.Model: The built model.
        """

        inputs = Input(shape=(self.input_size,))
        inputs = Masking(mask_value=float(0))(inputs)
        inputs = GaussianNoise(0.1)(inputs)
        word_embeddings = Embedding(input_dim=self.input_dim,
                                    weights=[self.word_embedding_matrix],
                                    trainable=False,
                                    output_dim=self.embedding_output_dim)(inputs)
        positional_encodings = Embedding(input_dim=self.input_dim,
                                         weights=[
                                             self.positional_encoding_matrix],
                                         output_dim=self.embedding_output_dim)(inputs)
        embeddings = Concatenate(
            axis=-1)([word_embeddings, positional_encodings])

        tcn_out = TCN(self.tcn_units, return_sequences=True,
                      kernel_size=self.tcn_kernel_size)
        tcn_out = Bidirectional(tcn_out)(embeddings)
        attention = Attention()([tcn_out, tcn_out])

        attention = Concatenate(axis=-1)([tcn_out, attention])
        outputs = GlobalAveragePooling1D()(attention)
        outputs = Dropout(self.dropout_rate)(outputs)
        outputs = Dense(
            self.num_classes, activation='sigmoid')(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, train_labels, epochs=50, batch_size=64, patience=5):
        """
        Train the TCN sentiment classifier.

        Args:
            train_data (ndarray): The training data.
            train_labels (ndarray): The training labels.
            epochs (int, optional): The number of epochs. Defaults to 50.
            batch_size (int, optional): The batch size. Defaults to 64.
            patience (int, optional): The patience for early stopping. Defaults to 5.

        Returns:
            keras.History: The training history.
        """

        lr_scheduler = LearningRateScheduler(self._lr_schedule)

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       patience=patience,
                                       verbose=1,
                                       mode='max',
                                       restore_best_weights=True)

        class_counts = np.unique(train_labels, return_counts=True)[1]
        ratio = class_counts[0]/class_counts[1]

        class_weights = {0: 1, 1: ratio}

        history = self.model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2, class_weight=class_weights,
                                 batch_size=batch_size, callbacks=[early_stopping, lr_scheduler])
        self.save_training_plots(self.model_name, history)
        return history

    def _lr_schedule(self, epoch):
        """
        Learning rate schedule.

        Args:
            epoch (int): The current epoch.

        Returns:
            float: The learning rate.
        """

        initial_lr = 0.001
        drop = 0.25
        epochs_drop = 10
        lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr
