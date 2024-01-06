"""
The implementation of the SVM model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. Failure Prediction 
        in IBM BlueGene/L Event Logs. IEEE International Conference on Data Mining
        (ICDM), 2007.

"""

import os

import pandas as pd
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn import svm
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from ..utils import metrics


class SVMClassifier():
    """
    SVMClassifier is a class that implements the Invariants Mining model for anomaly detection using Support Vector Machines (SVM).

    Parameters
    ----------
    penalty : str, default='l1'
        The penalty term used in the SVM model.
    tol : float, default=0.1
        The tolerance for stopping criteria.
    C : float, default=1
        The regularization parameter.
    dual : bool, default=False
        Whether to solve the dual or primal optimization problem.
    class_weight : dict or 'balanced', default=None
        The weights associated with classes in the SVM model.
    prediction_threshold : float, default=0.75
        The threshold for classifying instances as anomalies.
    max_iter : int, default=100
        The maximum number of iterations for the SVM model.
    model_artefact_dir : str, default="artefacts/models/"
        The directory to store the model artifacts.
    model_name : str, default="svm"
        The name of the SVM model.

    Attributes
    ----------
    classifier : object
        The classifier for anomaly detection.

    Methods
    -------
    build_model()
        Build the SVM model.
    train(X, y)
        Train the SVM model.
    predict(X)
        Predict anomalies with the SVM model.
    evaluate(X, y_true, save_results=False)
        Evaluate the performance of the SVM model.
    save_roc_curve(true, predicted)
        Save the ROC curve plot.
    save_cofusion_matrix(true, predicted)
        Save the confusion matrix plot.
    save_classification_report(true, predicted)
        Save the classification report.
    cross_validate(train_data, train_labels, num_splits=5, num_epochs=10, prediction_threshold=0.9)
        Perform cross-validation on the SVM model.

    """

    def __init__(self, penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, prediction_threshold=0.75,
                 max_iter=100, model_artefact_dir="artefacts/models/", model_name="svm"):
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.dual = dual
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.model_artefact_dir = model_artefact_dir
        self.prediction_threshold = prediction_threshold
        self.model_name = model_name
        self.model_artefact_dir = os.path.join(model_artefact_dir, model_name)
        os.makedirs(self.model_artefact_dir, exist_ok=True)
        self.model = self.build_model()

    def build_model(self):
        """
        Build the SVM model.

        Returns
        -------
        model : object
            The SVM model.

        """
        return svm.LinearSVC(penalty=self.penalty, tol=self.tol, C=self.C, dual=self.dual,
                             class_weight=self.class_weight, max_iter=self.max_iter)

    def train(self, X, y):
        """
        Train the SVM model.

        Parameters
        ----------
        X : ndarray
            The event count matrix of shape num_instances-by-num_events.
        y : ndarray
            The label vector of shape (num_instances,).

        """
        print('====== Model summary ======')
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict anomalies with the SVM model.

        Parameters
        ----------
        X : ndarray
            The input event count matrix.

        Returns
        -------
        y_pred : ndarray
            The predicted label vector of shape (num_instances,).

        """
        y_pred = self.model.predict(X)
        return y_pred

    def evaluate(self, X, y_true, save_results=False):
        """
        Evaluate the performance of the SVM model.

        Parameters
        ----------
        X : ndarray
            The input event count matrix.
        y_true : ndarray
            The true label vector of shape (num_instances,).
        save_results : bool, default=False
            Whether to save the evaluation results.

        Returns
        -------
        accuracy : float
            The accuracy of the SVM model.
        precision : float
            The precision of the SVM model.
        recall : float
            The recall of the SVM model.
        f1 : float
            The F1-measure of the SVM model.

        """
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        if save_results:
            self.save_cofusion_matrix(y_true, y_pred)
            self.save_classification_report(y_true, y_pred)
            self.save_roc_curve(y_true, y_pred)
        precision, recall, f1 = metrics(y_pred, y_true)
        y_pred_binary = (y_pred > self.prediction_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)
        print(
            'Accuracy: {:.3f}, Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(accuracy,precision, recall, f1))
        return accuracy, precision, recall, f1

    def save_roc_curve(self, true, predicted):
        """
        Save the ROC curve plot.

        Parameters
        ----------
        true : ndarray
            The true label vector of shape (num_instances,).
        predicted : ndarray
            The predicted label vector of shape (num_instances,).

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

    def save_cofusion_matrix(self, true, predicted):
        """
        Save the confusion matrix plot.

        Parameters
        ----------
        true : ndarray
            The true label vector of shape (num_instances,).
        predicted : ndarray
            The predicted label vector of shape (num_instances,).

        """
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

        # save confusion matrix as csv
        df = pd.DataFrame(cm)
        df.to_csv(os.path.join(self.model_artefact_dir, 'confusion_matrix.csv'))

    def save_classification_report(self, true, predicted):
        """
        Save the classification report.

        Parameters
        ----------
        true : ndarray
            The true label vector of shape (num_instances,).
        predicted : ndarray
            The predicted label vector of shape (num_instances,).

        Returns
        -------
        report : str
            The classification report.

        """
        plot_file = os.path.join(
            self.model_artefact_dir, 'classification_report.txt')
        threshold = self.prediction_threshold
        predicted_binary = (predicted > threshold).astype(int)
        report = classification_report(true, predicted_binary)
        with open(plot_file, 'w') as f:
            f.write(report)
        return report

    def cross_validate(self,train_data, train_labels, num_splits=5, num_epochs=10, prediction_threshold=0.9):
        """
        Perform cross-validation on the SVM model.

        Parameters
        ----------
        train_data : ndarray
            The training data.
        train_labels : ndarray
            The training labels.
        num_splits : int, default=5
            The number of splits for cross-validation.
        num_epochs : int, default=10
            The number of epochs for training.
        prediction_threshold : float, default=0.9
            The threshold for classifying instances as anomalies.

        Returns
        -------
        accuracies : list
            The list of accuracies for each fold.
        precisions : list
            The list of precisions for each fold.
        recalls : list
            The list of recalls for each fold.
        fscores : list
            The list of F1-scores for each fold.
        aucs : list
            The list of AUC scores for each fold.
        conf_matrices : list
            The list of confusion matrices for each fold.

        """
        accuracies = []
        precisions = []
        recalls = []
        fscores = []
        aucs = []
        conf_matrices = []

        def build_fn():
            model = self.build_model()
            return model
        
        model = KerasClassifier(build_fn=build_fn, verbose=0)
        cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        for train_index, test_index in cv.split(train_data, train_labels):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=1)
            
            y_pred = model.predict(X_test)
            y_pred = (y_pred > prediction_threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            accuracies.append(acc)
            precisions.append(prec) 
            recalls.append(rec)
            fscores.append(f1)
            aucs.append(auc)
            conf_matrices.append(cm)
        return accuracies, precisions, recalls, fscores, aucs, conf_matrices