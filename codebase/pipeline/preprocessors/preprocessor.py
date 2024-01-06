import pickle
from string import punctuation

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sgt import SGT
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer

# @staticmethod
# @np.vectorize
# def strip_punctuation(s):
#     return ''.join(c for c in s if c not in punctuation)


class SGTVectorizer(object):
    """
    SGTVectorizer is a class that performs sequence vectorization using the SGT algorithm.

    Args:
        num_dims (int): The number of dimensions for the vector representation (default: 64).
        processors (int): The number of processors to use for parallelization (default: None).

    Methods:
        fit(corpus): Fit the SGTVectorizer on the given corpus.
        fit_transform(corpus): Fit the SGTVectorizer on the given corpus and transform it.
        transform(x): Transform the given data using the fitted SGTVectorizer.

    """

    def __init__(self, num_dims=64, processors=None):
        self.num_dims = num_dims
        self.processors = processors

    def fit(self, corpus):
        """
        Fit the SGTVectorizer on the given corpus.

        Args:
            corpus (DataFrame): The corpus containing the sequence data.

        """
        all_tokens = [token for sublist in corpus['sequence']
                      for token in sublist]
        unique_sequence_ids = list(set(all_tokens))

        self.sgt = SGT(alphabets=unique_sequence_ids, kappa=1,
                       lengthsensitive=False)

    def fit_transform(self, corpus):
        """
        Fit the SGTVectorizer on the given corpus and transform it.

        Args:
            corpus (DataFrame): The corpus containing the sequence data.

        Returns:
            ndarray: The transformed data.

        """
        self.fit(corpus)
        return self.transform(corpus)

    def transform(self, x):
        """
        Transform the given data using the fitted SGTVectorizer.

        Args:
            x (DataFrame): The data to be transformed.

        Returns:
            ndarray: The transformed data.

        """
        x = self._prepare_corpus(x)
        x = self._vectorize(x)
        x = self._reduce_dim(x)
        x = np.array(x).astype(np.float32)
        return x

    def _vectorize(self, x):
        x = self.sgt.fit_transform(x)
        x = x.set_index('id')
        return x

    def _reduce_dim(self, x):
        pca = PCA(n_components=self.num_dims)
        pca.fit(x)
        x = pca.transform(x)
        return x

    def _prepare_corpus(self, sequence_data):
        corpus_df = pd.DataFrame(sequence_data, columns=[
                                 'sequence'])
        corpus_df.index = pd.RangeIndex(len(corpus_df.index))
        corpus_df['id'] = corpus_df.index
        return corpus_df


class SequenceVectorizer(object):
    """
    A class for vectorizing sequences of text.

    Parameters:
    - num_words (int): The maximum number of words to keep based on word frequency. Default is 1000.
    - mode (str): The mode for vectorization. Options are 'binary', 'count', 'tfidf', and 'freq'. Default is 'binary'.

    Methods:
    - fit(x): Fit the tokenizer on the given text data.
    - fit_transform(x): Fit the tokenizer on the given text data and transform it into a matrix.
    - transform(x): Transform the given text data into a matrix using the fitted tokenizer.
    - save_tokenizer(tokenizer_file): Save the fitted tokenizer to a file.
    """

    def __init__(self, num_words=1000, mode='binary'):
        self.num_words = num_words
        self.matrix_mode = mode

    def fit(self, x):
        """
        Fit the tokenizer on the given text data.

        Parameters:
        - x (list): The list of text data to fit the tokenizer on.
        """
        self.tokenizer = Tokenizer(
            num_words=self.num_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(x)
        self.word_index = self.tokenizer.word_index

    def fit_transform(self, x):
        """
        Fit the tokenizer on the given text data and transform it into a matrix.

        Parameters:
        - x (list): The list of text data to fit the tokenizer on and transform.

        Returns:
        - x (numpy.ndarray): The transformed matrix representation of the text data.
        """
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        """
        Transform the given text data into a matrix using the fitted tokenizer.

        Parameters:
        - x (list): The list of text data to transform.

        Returns:
        - x (numpy.ndarray): The transformed matrix representation of the text data.
        """
        x = self.tokenizer.texts_to_matrix(x, self.matrix_mode)
        x = np.array(x).astype(np.float32)
        return x

    def save_tokenizer(self, tokenizer_file):
        """
        Save the fitted tokenizer to a file.

        Parameters:
        - tokenizer_file (str): The file path to save the tokenizer to.
        """
        with open(tokenizer_file, 'wb') as handle:
            pickle.dump(self.tokenizer, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


class TFIDFVectorizer(object):
    """
    TFIDFVectorizer is a class that performs TF-IDF vectorization on text data.

    Parameters:
    - max_features (int): The maximum number of features to keep. Default is 1000.

    Methods:
    - fit(x): Fit the vectorizer on the training data.
    - fit_transform(x_train, y_train): Fit the vectorizer on the training data and transform it.
    - transform(x, y): Transform the input data using the fitted vectorizer.

    Attributes:
    - vectorizer: The fitted TfidfVectorizer object.
    - features: The feature names extracted from the vectorizer.
    """

    def __init__(self, max_features=1000):
        self.max_features = max_features

    def fit(self, x):
        """
        Fit the vectorizer on the training data.

        Parameters:
        - x (list): The input data to fit the vectorizer on.
        """
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.vectorizer.fit(x)
        self.features = self.vectorizer.get_feature_names_out()

    def fit_transform(self, x_train, y_train):
        """
        Fit the vectorizer on the training data and transform it.

        Parameters:
        - x_train (list): The input training data.
        - y_train (list): The target training data.

        Returns:
        - x_transformed (array): The transformed input data.
        - y_train (array): The target training data.
        """
        self.fit(x_train)
        return self.transform(x_train, y_train)

    def transform(self, x, y):
        """
        Transform the input data using the fitted vectorizer.

        Parameters:
        - x (list): The input data to transform.
        - y (array): The target data.

        Returns:
        - x_transformed (array): The transformed input data.
        - y (array): The target data.
        """
        # x is a list of lists, need to convert the inner lists to strings
        x = self.vectorizer.transform(x)
        x = x.toarray()
        y = y.reshape(y.shape[0])
        return x, y


class HDFSVectorizer(object):
    """
    Vectorizer class that transforms input data into numerical sequences.
    """

    def fit_transform(self, x_train, y_train):
        """
        Fits the vectorizer to the training data and transforms it into numerical sequences.

        Args:
            x_train (list): List of input sequences.
            y_train (list): List of corresponding labels.

        Returns:
            tuple: A tuple containing the transformed input data and corresponding labels.
        """
        self.max_seq_len = max([len(entry) for entry in x_train])
        unique_sequences = []
        for entry in x_train:
            entry = np.array(entry)
            for seq in entry:
                if seq not in unique_sequences:
                    unique_sequences.append(seq)
        self.label_mapping = {seq: idx for idx,
                              seq in enumerate(unique_sequences, 2)}
        self.label_mapping["#OOV"] = 1
        self.label_mapping["#Pad"] = 0
        self.num_labels = len(self.label_mapping)
        self.scaled = False
        return self.transform(x_train, y_train)

    def transform(self, x, y):
        """
        Transforms the input data into numerical sequences.

        Args:
            x (list): List of input sequences.
            y (list): List of corresponding labels.

        Returns:
            tuple: A tuple containing the transformed input data and corresponding labels.
        """
        x_padded = []
        # pad x to max_seq_len with #Pad
        for entry in x:
            entry = np.array(entry)
            # pad with #Pad
            entry = np.pad(entry, (0, self.max_seq_len -
                           len(entry)), constant_values="#Pad")
            x_padded.append(entry)
        x = np.array(x_padded)
        x_seq = []
        for entry in x:
            entry = np.array([self.label_mapping.get(item, 0)
                             for item in entry])
            x_seq.append(entry)
        x = np.array(x_seq).astype(np.float32)
        x = x.reshape(x.shape[0], x.shape[1])
        y = y.reshape(y.shape[0])
        return x, y


# class ThunderbirdVectorizer:
#     """
#     Vectorizes Thunderbird email data for machine learning models.

#     Attributes:
#     -----------
#     window_size : int
#         The size of the sliding window used to slice the sequences.
#     sequence_length : int
#         The maximum length of the sequences.
#     tokenizer : keras.preprocessing.text.Tokenizer
#         The tokenizer used to convert text to sequences.
#     num_words : int
#         The number of words in the tokenizer's word index plus one.

#     Methods:
#     --------
#     fit_transform(x, y_train, window_size)
#         Fits the tokenizer on the input data and returns the transformed data.
#     transform(x, y)
#         Transforms the input data using the fitted tokenizer.
#     slice_tbird(y, sequences, window_size)
#         Slices the sequences into windows of size window_size.
#     strip_punctuation(s)
#         Removes punctuation from a string.
#     """

#     def fit_transform(self, x, y, window_size=0):
#         """
#         Fit the tokenizer on the input data and transform the input data into sequences of tokens.

#         Args:
#             x (array-like): Input data to be transformed.
#             y (array-like): Target data.
#             window_size (int): Size of the sliding window used to generate sequences.

#         Returns:
#             array-like: Transformed input data.
#         """
#         df_x = pd.DataFrame(x)
#         self.window_size = window_size
#         self.sequence_length = 0
#         self.tokenizer = Tokenizer(
#             num_words=1000, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
#         self.tokenizer.fit_on_texts(df_x[0])
#         self.num_words = len(self.tokenizer.word_index) + 1
#         return self.transform(df_x[0].values, y)

#     def transform(self, x, y):
#         """
#         Transforms the input data `x` and `y` using the preprocessor's settings.

#         Args:
#             x (list): List of input data.
#             y (list): List of target data.

#         Returns:
#             tuple: A tuple containing the transformed input data `x`, target data `y`,
#             and slice data `s` (if `window_size` > 1).
#         """
#         window_size = self.window_size
#         if window_size > 1:
#             s, x, y = self.slice_tbird(y, x, window_size)
#             x = self.tokenizer.texts_to_sequences(x)
#             x = np.array(x).astype(np.float32)
#             x = x.reshape(x.shape[0], x.shape[1], 1)
#             y = np.array(y).astype(np.float32).reshape(y.shape[0], 1)
#             s = np.array(s).astype(np.float32).reshape(s.shape[0], 1, 1)

#             return x, y, s
#         else:

#             x = self.tokenizer.texts_to_sequences(x)
#             if self.sequence_length == 0:
#                 self.sequence_length = max([len(entry) for entry in x])
#             x = pad_sequences(x, maxlen=self.sequence_length,
#                               padding="post", truncating="post")
#             x = np.array(x).astype(np.float32)
#             y = np.array(y).astype(np.float32)
#             return x, y

#     def slice_tbird(self, y, sequences, window_size):
#         """
#         Slices the input sequences into smaller windows of size `window_size` and returns a DataFrame with the sliced sequences,
#         their corresponding labels, and their session IDs.

#         Args:
#             y (list): A list of labels for each sequence.
#             sequences (list): A list of sequences to be sliced.
#             window_size (int): The size of the window to slice the sequences into.

#         Returns:
#             tuple: A tuple containing three arrays: the session IDs, the sliced sequences, and their corresponding labels.
#         """
#         results_data = []
#         for idx, sequence in enumerate(sequences):
#             seqlen = len(sequence)
#             i = 0
#             while (i + window_size) < seqlen:
#                 slice = sequence[i: i + window_size]
#                 results_data.append([idx, slice, y[idx]])
#                 i += 1
#             else:
#                 slice = sequence[i: i + window_size]
#                 slice += ["#Pad"] * (window_size - len(slice))
#                 results_data.append([idx, slice, y[idx]])
#         results_df = pd.DataFrame(results_data, columns=[
#                                   "SessionId", "SequenceText", "Label"])
#         return results_df["SessionId"].values, results_df["SequenceText"].values, results_df["Label"].values


class BertEventTokenizer(object):
    def __init__(self, model_type='distilbert-base-uncased', max_length=128, batch_size=32):
        """
        Initializes a BertEventTokenizer object.

        Parameters:
        - model_type: The type of pre-trained model to use for tokenization. Default is 'distilbert-base-uncased'.
        - max_length: The maximum length for tokenized sequences. Default is 128.
        - batch_size: The batch size for tokenization. Default is 32.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_type)
        self.max_length = max_length
        self.batch_size = batch_size
        self.vocab_size = 0

    def transform(self, texts):
        """
        Tokenizes the input texts using the pre-fitted tokenizer and returns the tokenized sequences
        along with attention masks.

        Parameters:
        - texts: List of input texts to be tokenized.

        Returns:
        - input_ids: Tokenized input sequences.
        - attention_masks: Attention masks indicating the presence of tokens.
        """
        input_ids = []
        attention_mask = []
        if isinstance(texts[0], list):
            texts = [" ".join(text) for text in texts]

        inputs = self.tokenizer.batch_encode_plus(texts,
                                                  max_length=self.max_length,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False
                                                  )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
        vocab_size = len(self.tokenizer.get_vocab())
        if vocab_size > self.vocab_size:
            self.vocab_size = vocab_size
        return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)
