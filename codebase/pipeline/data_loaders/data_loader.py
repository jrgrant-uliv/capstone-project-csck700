import os
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict

class HDFSLoader:
    def __init__(self, log_file, label_file, template_file, data_dir=None, window='session', train_ratio=0.7, split_type='sequential'):
        self.log_file = log_file
        self.label_file = label_file
        self.template_file = template_file
        self.window = window
        self.train_ratio = train_ratio
        self.split_type = split_type
        self.data_dir = data_dir

    
    def _split_data(self, x_data, y_data=None, train_ratio=0, split_type='uniform'):
        if split_type == 'uniform' and y_data is not None:
            pos_idx = y_data > 0
            x_pos = x_data[pos_idx]
            y_pos = y_data[pos_idx]
            x_neg = x_data[~pos_idx]
            y_neg = y_data[~pos_idx]
            train_pos = int(train_ratio * x_pos.shape[0])
            train_neg = int(train_ratio * x_neg.shape[0])
            x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
            y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
            x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
            y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        elif split_type == 'sequential':
            num_train = int(train_ratio * x_data.shape[0])
            x_train = x_data[0:num_train]
            x_test = x_data[num_train:]
            if y_data is None:
                y_train = None
                y_test = None
            else:
                y_train = y_data[0:num_train]
                y_test = y_data[num_train:]
        # Random shuffle
        indexes = shuffle(np.arange(x_train.shape[0]))
        x_train = x_train[indexes]
        if y_train is not None:
            y_train = y_train[indexes]
        return (x_train, y_train), (x_test, y_test)


    def to_idx(self, x, template_file):
        ## convert template to index by EventID
        vocab2idx = dict()
        template_file = pd.read_csv(template_file, engine='c', na_filter=False, memory_map=True)
        for idx, template_id in enumerate(template_file['EventId'], start=len(vocab2idx)):
            vocab2idx[template_id] = idx + 1

        max_len = 0
        x_idx = []
        for i in range(x.shape[0]):
            if len(x[i]) > max_len:
                max_len = len(x[i])

        for i in range(x.shape[0]):
            temp = []
            for j in range(len(x[i])):
                temp.append(vocab2idx[x[i][j]])
            temp += [0]*(max_len-len(x[i]))
            x_idx.append(temp)

        return np.array(x_idx,dtype=float)

    def load_HDFS(self):
        struct_log = pd.read_csv(self.log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if blk_Id not in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        label_data = pd.read_csv(self.label_file, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

        (x_train, y_train), (x_test, y_test) = self._split_data(data_df['EventSequence'].values, data_df['Label'].values, self.train_ratio, self.split_type)

        x_train = self.to_idx(x_train, self.template_file)
        x_test = self.to_idx(x_test, self.template_file)

        if self.data_dir is not None:
            data_file = self.data_dir + 'data_instances.csv'
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            data_df.to_csv(data_file, index=False)

        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    log_file = '../../logs/HDFS/brain/HDFS_2K.log_structured.csv'
    label_file = '../../logs/HDFS/brain/anomaly_label.csv'
    template_file = '../../logs/HDFS/brain/HDFS_2K.log_templates.csv'
    data_dir = '../../data/processed/HDFS_2K/'
    loader = HDFSLoader(log_file, label_file, template_file, data_dir)
    x_train, y_train, x_test, y_test = loader.load_HDFS()


