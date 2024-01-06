"""
This module defines a base classifier class and a metrics callback class for evaluating the performance of a Keras model.
The Classifier class is an abstract class that defines the basic structure of a classifier model and the methods that should be implemented by any subclass.
The MetricsCallback class is a Keras callback that records the accuracy, precision, recall, and F1-score of a model on a validation set at the end of each epoch.
"""
import os
from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
from keras.utils import plot_model
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold


class Classifier(ABC):
    """
    Abstract base class for all classifiers.

    Attributes:
        model_name (str): The name of the classifier model.
        prediction_threshold (float): The threshold for binary classification predictions.
        model_evaluation_type (str): The type of model evaluation.
        model_artefact_dir (str): The directory to save model artefacts.

    Methods:
        build_model(): Abstract method to build the classifier model.
        train(x_train, y_train, epochs=5, batch_size=64): Abstract method to train the classifier model.
        evaluate(x_test, y_test, s_test=[]): Method to evaluate the classifier model on test data.
        save_model_file(): Save the trained model to a file.
        save_model_summary(): Save the summary of the model to a text file.
        plot_model(to_file="model.png"): Plot the model architecture and save it to a file.
        save_training_plots(model_name, history): Save the training history plots.
        save_model_plots(true, predicted): Save the ROC curve and confusion matrix plots.
        cross_validate(train_data, train_labels, num_splits=5, num_epochs=10, prediction_threshold=0.9): Perform cross-validation on the classifier model.

    """

    def __init__(self, model_name, prediction_threshold=0.75, model_evaluation_type='time_series_classification', model_artefact_dir="artefacts/models/"):
        self.model_name = model_name
        self.model_evaluation_type = model_evaluation_type
        self.model_artefact_dir = os.path.join(model_artefact_dir, model_name)
        self.prediction_threshold = prediction_threshold

        # ensure directory exists
        os.makedirs(self.model_artefact_dir, exist_ok=True)

    @abstractmethod
    def build_model(self):
        """
        Builds the model for the classifier.

        This method should be implemented by subclasses to define the architecture
        and configuration of the model.
        """
        pass

    def build_model_summary(self):
        """
        Build the model and save its summary.
        """
        self.model = self.build_model()
        self.after_build_model()

    def after_build_model(self):
        """
        Perform actions after building the model.
        """
        assert self.model is not None, "You must build the model first!"
        self.save_model_summary()
        self.plot_model()
        self.model.save(os.path.join(self.model_artefact_dir, 'new_model.h5'))

    @abstractmethod
    def train(self, x_train, y_train, epochs=5, batch_size=64):
        """
        Train the classifier model.

        Args:
            x_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        pass

    def evaluate(self, x_test, y_test, s_test=[]):
        """
        Evaluate the classifier model on test data.

        Args:
            x_test (numpy.ndarray): Test data features.
            y_test (numpy.ndarray): Test data labels.
            s_test (numpy.ndarray, optional): Test data auxiliary features. Defaults to [].

        Returns:
            tuple: A tuple containing the evaluation metrics (accuracy, precision, recall, f1-score).
        """
        if self.model_evaluation_type == 'time_series_classification':
            y_true = y_test
            y_pred = self.model.predict(x_test)
            y_pred_binary = (y_pred > self.prediction_threshold).astype(int)
            samples = y_pred.shape[0]
            if samples > 1000:
                samples = 1000
            else:
                samples = int(samples * 0.5)

            accuracy = accuracy_score(y_true, y_pred_binary)
            precision = precision_score(y_true, y_pred_binary)
            recall = recall_score(y_true, y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary)
            self.save_model_plots(y_true, y_pred)
        else:
            if (len(s_test) > 0):
                y_pred = self.model.predict([s_test, x_test])
            else:
                y_pred = self.model.predict(x_test)
            print(len(y_pred))
            # Existing implementation for evaluation
            y_pred_binary = (y_pred > self.prediction_threshold).astype(int)

            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)
        print('====== Evaluation summary ======')
        print('Accuracy: {:.3f}, Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(
            accuracy, precision, recall, f1))

        return accuracy, precision, recall, f1, y_pred_binary

    def save_model_file(self):
        """
        Save the trained model to a file.
        """
        model_file = os.path.join(self.model_artefact_dir, 'model.h5')
        self.model.save(model_file)

    def save_model_summary(self):
        """
        Save the summary of the model to a text file.
        """
        self.model.summary()
        model_report = os.path.join(
            self.model_artefact_dir, 'model_report.txt')
        with open(model_report, 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(
                line_length=120, print_fn=lambda x: fh.write(x + '\n'))

    def plot_model(
        self,
        to_file="model.png"
    ):
        """
        Plot the model architecture and save it to a file.

        Args:
            to_file (str, optional): The filename to save the plot. Defaults to "model.png".
        """
        to_file = os.path.join(self.model_artefact_dir, to_file)
        plot_model(self.model, to_file, show_shapes=True)

    def save_training_plots(self, model_name, history):
        """
        Save the training history plots.

        Args:
            model_name (str): The name of the model.
            history (keras.callbacks.History): The training history object.
        """
        plot_file_name = 'training_history.png'
        plot_file = os.path.join(self.model_artefact_dir, plot_file_name)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(y=history.history['val_loss'], name="val_loss"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(y=history.history['loss'], name="loss"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(y=history.history['val_accuracy'], name="val accuracy"),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(y=history.history['accuracy'], name="accuracy"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Model Training Loss/Accuracy"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Epoch")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>primary</b> Loss", secondary_y=False)
        fig.update_yaxes(
            title_text="<b>secondary</b> Accuracy", secondary_y=True)

        fig.write_image(plot_file)

        fig.show()

    def save_model_plots(self, true, predicted):
        """
        Save the ROC curve and confusion matrix plots.

        Args:
            true (numpy.ndarray): True labels.
            predicted (numpy.ndarray): Predicted labels.
        """
        # Calculate AUC
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(true, predicted)
        roc_auc = auc(fpr, tpr)

        plot_file = os.path.join(self.model_artefact_dir, 'roc_curve.png')
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(plot_file)
        plt.show()

        # Plot Confusion Matrix
        plot_file = os.path.join(
            self.model_artefact_dir, 'confusion_matrix.png')
        threshold = self.prediction_threshold
        predicted_binary = (predicted > threshold).astype(int)
        cm = confusion_matrix(true, predicted_binary)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Customize x and y-axis labels
        tick_labels = ['Negative', 'Positive']
        plt.xticks([0, 1], tick_labels)
        plt.yticks([0, 1], tick_labels)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center',
                         va='center', color='red', fontsize=12)

        plt.savefig(plot_file)
        plt.show()

    def cross_validate(self, train_data, train_labels, num_splits=5, num_epochs=10, prediction_threshold=0.9):
        """
        Perform cross-validation on the classifier model.

        Args:
            train_data (numpy.ndarray): Training data features.
            train_labels (numpy.ndarray): Training data labels.
            num_splits (int, optional): Number of cross-validation splits. Defaults to 5.
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
            prediction_threshold (float, optional): The threshold for binary classification predictions. Defaults to 0.9.

        Returns:
            tuple: A tuple containing the evaluation metrics (accuracies, precisions, recalls, fscores, aucs, conf_matrices, predictions).
        """
        accuracies = []
        precisions = []
        recalls = []
        fscores = []
        aucs = []
        conf_matrices = []
        roc_curves = []

        cv_count = 0

        def build_fn():
            model = self.build_model()
            return model

        model = KerasClassifier(build_fn=build_fn, verbose=0)
        cv = StratifiedKFold(n_splits=num_splits,
                             shuffle=True, random_state=42)

        for train_index, test_index in cv.split(train_data, train_labels):
            cv_count = cv_count + 1
            print ("===========================================================================")
            print (f"Model {self.model_name} - Cross Validation Iteration: {cv_count}")
            print ("===========================================================================")
            # calculate class weights
            class_counts = np.unique(
                train_labels[train_index], return_counts=True)[1]
            ratio = class_counts[0]/class_counts[1]
            class_weights = {0: 1, 1: ratio}

            print("Cross Validation Iteration: ", cv_count)
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            model.fit(X_train, y_train, epochs=num_epochs, class_weight=class_weights,
                      batch_size=32, verbose=1)

            y_pred = model.predict(X_test)
            y_pred = (y_pred > prediction_threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            roc = roc_curve(y_test, y_pred)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            fscores.append(f1)
            aucs.append(auc_score)
            conf_matrices.append(cm)
            roc_curves.append(roc)
        return accuracies, precisions, recalls, fscores, aucs, conf_matrices, roc_curves
