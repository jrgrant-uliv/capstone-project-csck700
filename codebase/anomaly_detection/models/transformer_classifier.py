import numpy as np
import tensorflow as tf
from codebase.anomaly_detection.models.base_classifier import Classifier
from focal_loss import BinaryFocalLoss
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.initializers import GlorotNormal
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from transformers import DistilBertConfig, TFDistilBertModel

# from transformers import BertConfig, TFAutoModel


class TransformerClassifier(Classifier):
    """
    Initialize the TransformerClassifier.

    Args:
        model_name (str): The name of the model.
        num_classes (int): The number of classes.
        seq_len (int): The sequence length.
        vocab_size (int): The vocabulary size.
        model_type (str, optional): The type of the model. Defaults to 'Intel/dynamic_tinybert'.
        model_artefact_dir (str, optional): The directory for model artifacts. Defaults to "artefacts/models/".

    Returns:
        None
    """
    
    def __init__(self, model_name, num_classes, seq_len, vocab_size, model_type='Intel/dynamic_tinybert', model_artefact_dir="artefacts/models/"):

        super().__init__(model_name=model_name, model_artefact_dir=model_artefact_dir)

        self.model_type = model_type
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.batch_size = 32
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.distilbert_dropout = 0.2
        self.distilbert_attention_dropout = 0.2
        self.random_state = 42
        self.learning_rate = 2e-5  # 5e-5  # the recommendation of all the BeRT wisperers
        self.fine_tuning_learning_rate = 5e-5
        self.build_model_summary()

    def build_model(self):
        """
        Build and return the transformer classifier model.

        Returns:
            tf.keras.Model: The built transformer classifier model.
        """

        layer_dropout = 0.2
        distilbert_dropout = 0.1
        distilbert_attention_dropout = 0.1
        vocab_size = self.vocab_size
        model_type = self.model_type
        random_state = self.random_state
        seq_len = self.seq_len
        fine_tuning_learning_rate = self.fine_tuning_learning_rate
        learning_rate = self.learning_rate

        max_pos_embeddings = 512

        config = DistilBertConfig(dropout=distilbert_dropout,
                                    max_position_embeddings=max_pos_embeddings,
                                    attention_dropout=distilbert_attention_dropout,
                                    vocab_size=vocab_size,
                                    output_attentions=True,
                                    output_hidden_states=True)
        transformer = TFDistilBertModel.from_pretrained(
            model_type, config=config
        )
        self.distilbert_layer = transformer
        weight_initializer = GlorotNormal(
            seed=random_state)
        input_ids_layer = Input(shape=(None,),
                                                name='input_ids',
                                                dtype='int32')
        input_attention_layer = Input(shape=(None,),
                                                        name='input_attention',
                                                        dtype='int32')

        last_hidden_state = transformer(
            [input_ids_layer, input_attention_layer])[0]

        cls_token = last_hidden_state[:, 0, :]

        dropout1 = Dropout(layer_dropout, seed=random_state)(cls_token)

        # Define a single node that makes up the output layer (for binary classification)
        output = Dense(1,
                    activation='sigmoid',
                    kernel_initializer=weight_initializer,  # CONSIDER USING CONSTRAINT
                    bias_initializer='zeros'
                    )(dropout1)
        # Define the model
        model = Model(
            [input_ids_layer, input_attention_layer], output)

        self.compile_model(fine_tuning_learning_rate, model)
        return model

    def compile_model(self, fine_tuning_learning_rate, model):
        """
        Compiles the model with the specified learning rate and loss function.

        Args:
            fine_tuning_learning_rate (float): The learning rate for fine-tuning.
            model (tf.keras.Model): The model to compile.

        Returns:
            None
        """
        model.compile(Adam(learning_rate=fine_tuning_learning_rate),
                      loss=BinaryFocalLoss(gamma=0.25),
                      metrics=['accuracy'])

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=5, batch_size=32, patience=3):
        """
        Trains the transformer classifier model.

        Args:
            x_train (tuple): Tuple containing the input_ids and input_attention for the training data.
            y_train (numpy.ndarray): Labels for the training data.
            x_val (tuple, optional): Tuple containing the input_ids and input_attention for the validation data. Defaults to None.
            y_val (numpy.ndarray, optional): Labels for the validation data. Defaults to None.
            epochs (int, optional): Number of epochs to train the model. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 3.
            transfer_learning (bool, optional): Whether to use transfer learning. Defaults to False.

        Returns:
            tensorflow.python.keras.callbacks.History: Training history.
        """
        input_ids = x_train[0]
        input_attention = x_train[1]
        input_labels = y_train
        val_input_ids = x_val[0]
        val_input_attention = x_val[1]
        val_labels = y_val
        class_counts = np.unique(y_train, return_counts=True)[1]
        ratio = class_counts[0]/class_counts[1]

        class_weights = {0: 1, 1: ratio}

        early_stopping = EarlyStopping(monitor='val_accuracy',
                                        patience=patience,
                                        verbose=1,
                                        mode='max',
                                        restore_best_weights=True)

        history = self.model.fit([input_ids, input_attention], input_labels, class_weight=class_weights,
                                    validation_data=(
                                        [val_input_ids, val_input_attention], val_labels), epochs=epochs,
                                    batch_size=batch_size, callbacks=[early_stopping], verbose=1)
        self.save_training_plots(self.model_name, history)
        return history

    def unfreeze_layers(self):
        """
        Unfreezes the layers of the model and recompiles it with a lower learning rate.

        Returns:
            The recompiled model.
        """
        for layer in self.distilbert_layer.layers:
            layer.trainable = True

        # Lower the learning rate to prevent destruction of pre-trained weights
        optimizer = Adam(learning_rate=self.learning_rate)

        # Recompile model after unfreezing
        self.model.compile(optimizer=optimizer,
                            loss=BinaryFocalLoss(gamma=2.),
                            metrics=['accuracy'])
        return self.model

    def cross_validate(self, train_data, train_labels, num_splits=5, num_epochs=10, prediction_threshold=0.9, batch_size=16):
        """
        Perform cross-validation on the given training data and labels.

        Parameters:
        - train_data (tuple): A tuple containing the training data, where the first element is the sequences and the second element is the embeddings.
        - train_labels (numpy.ndarray): The labels corresponding to the training data.
        - num_splits (int): The number of splits for cross-validation. Default is 5.
        - num_epochs (int): The number of epochs to train the model. Default is 10.
        - prediction_threshold (float): The threshold for classifying predictions. Default is 0.9.

        Returns:
        - accuracies (list): A list of accuracy scores for each cross-validation iteration.
        - precisions (list): A list of precision scores for each cross-validation iteration.
        - recalls (list): A list of recall scores for each cross-validation iteration.
        - fscores (list): A list of F1 scores for each cross-validation iteration.
        - aucs (list): A list of AUC scores for each cross-validation iteration.
        - conf_matrices (list): A list of confusion matrices for each cross-validation iteration.
        - predictions (list): A list of predictions for each cross-validation iteration.
        """
        accuracies = []
        precisions = []
        recalls = []
        fscores = []
        aucs = []
        conf_matrices = []
        roc_curves = []

        cv_count = 0

        cv = StratifiedKFold(n_splits=num_splits,
                                shuffle=True, random_state=42)
        
        sequences = train_data[0].numpy()
        embeddings = train_data[1].numpy()

        for train_index, test_index in cv.split(sequences, train_labels):
            cv_count = cv_count + 1
            print ("===========================================================================")
            print (f"Model {self.model_name} - Cross Validation Iteration: {cv_count}")
            print ("===========================================================================")

            # Calculate class weights
            class_counts = np.unique(train_labels[train_index], return_counts=True)[1]
            ratio = class_counts[0] / class_counts[1]
            class_weights = {0: 1, 1: ratio}

            # Create TensorFlow Datasets
            train_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_ids': sequences[train_index], 'input_attention': embeddings[train_index]}, train_labels[train_index]))
            test_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_ids': sequences[test_index], 'input_attention': embeddings[test_index]}, train_labels[test_index]))

            # Batch and prefetch the datasets
            train_dataset = train_dataset.shuffle(len(train_index)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            # Build and train the model
            model = self.build_model()

            print ("===========================================================================")
            print (f"Model {self.model_name} - Transfer Learning")
            print ("===========================================================================")
            model.fit(train_dataset, epochs=num_epochs, verbose=1, class_weight=class_weights)

            # Unfreeze layers and continue training
            model = self.unfreeze_layers()

            print ("===========================================================================")
            print (f"Model {self.model_name} - Training")
            print ("===========================================================================")
            model.fit(train_dataset, epochs=num_epochs, verbose=1, class_weight=class_weights)

                # Evaluate the model on the test set
            test_predictions = model.predict(test_dataset)
            test_predictions = (test_predictions > prediction_threshold).astype(int)

            # Calculate metrics
            acc = accuracy_score(train_labels[test_index], test_predictions)
            prec = precision_score(train_labels[test_index], test_predictions)
            rec = recall_score(train_labels[test_index], test_predictions)
            f1 = f1_score(train_labels[test_index], test_predictions)
            auc_score = roc_auc_score(train_labels[test_index], test_predictions)
            cm = confusion_matrix(train_labels[test_index], test_predictions)
            roc = roc_curve(train_labels[test_index], test_predictions)

            # Append metrics to lists
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            fscores.append(f1)
            aucs.append(auc_score)
            conf_matrices.append(cm)
            roc_curves.append(roc)
        return accuracies, precisions, recalls, fscores, aucs, conf_matrices, roc_curves
