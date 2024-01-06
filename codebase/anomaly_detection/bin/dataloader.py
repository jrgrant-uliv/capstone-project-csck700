# Import necessary libraries
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# Function to split data into training and testing sets
def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    # Splitting based on 'uniform' or 'sequential' strategy
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        
        # Calculate the number of samples for training and testing
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        
        # Combine positive and negative samples for training and testing
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        # Split data sequentially
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    
    # Randomly shuffle the data
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    
    return (x_train, y_train), (x_test, y_test)

# Function to load HDFS log data
def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0):
    """
    Load and preprocess HDFS log data from a CSV or NumPy file.

    Args:
        log_file (str): Path to the HDFS log file.
        label_file (str, optional): Path to the labeled data file. Defaults to None.
        window (str, optional): Type of window to use for data slicing. Defaults to 'session'.
        train_ratio (float, optional): Ratio of training data to total data. Defaults to 0.5.
        split_type (str, optional): Type of data split to use. Defaults to 'sequential'.
        save_csv (bool, optional): Whether to save the preprocessed data as a CSV file. Defaults to False.
        window_size (int, optional): Size of the window to use for data slicing. Defaults to 0.

    Returns:
        tuple: A tuple of training and testing data, each containing a tuple of event sequences, windowed event sequences (if window_size > 0), and labels (if label_file is provided).
    """
    if log_file.endswith('.npz'):
        # Load data from a preprocessed NumPy file
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)
    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        # Read log data from a CSV file and preprocess it
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)

        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            # Extract block IDs from log content
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if blk_Id not in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        
        #shuffle the rows
        data_df = data_df.sample(frac=1.0, random_state=42) 
        
        if label_file:
            # If labeled data is provided, split it into training and testing sets
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, data_df['Label'].values, train_ratio, split_type)
        
        if save_csv:
            # Save the preprocessed data as a CSV file
            data_df.to_csv('data_instances.csv', index=False)
        
        if window_size > 0:
            # If a window size is specified, slice the data into windows
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)
        
        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only supports csv and npz files!')
    
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos
    
    return (x_train, y_train), (x_test, y_test)

# Function to slice HDFS log data into windows
def slice_hdfs(x, y, window_size):
    """
    Slices the input sequences into windows of size `window_size` and returns a DataFrame with the sliced sequences,
    their corresponding labels, and the session labels.

    Args:
        x (list): List of input sequences.
        y (list): List of labels corresponding to the input sequences.
        window_size (int): Size of the window to slice the sequences into.

    Returns:
        tuple: A tuple containing:
            - DataFrame: A DataFrame with the sliced sequences and their corresponding session IDs.
            - Series: A Series with the labels corresponding to the sliced sequences.
            - Series: A Series with the session labels.
    """
    results_data = []

    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])

    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]

# The load_BGL function and bgl_preprocess_data function are not commented here as they seem to be incomplete or require specific knowledge about the data and task they are performing.
