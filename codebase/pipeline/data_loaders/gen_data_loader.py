import numpy as np
import pandas as pd


from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from .data_balancer import DataBalancer


class ThunderbirdDataLoader:
    def __init__(self, log_file="", event_col_name="",balance_classes=True, random_state=88):
        self.log_file = log_file
        self.balance_classes = balance_classes
        self.random_state = random_state
        self.event_col_name = event_col_name

    def load_data(self):

        df = pd.read_csv(self.log_file, engine='c', na_filter=False, memory_map=True)
        
        #Update Label '-' is 0, rest is 1
        df['Label'] = df['Label'].apply(lambda x: 0 if x == '-' else 1)
         
        if self.balance_classes:
            print("Balancing classes...")
            balancer = DataBalancer(out_col_name=self.event_col_name, save_classified_data=True, log_file_path='./output', log_file_prefix='balanced_data')
            x_train, y_train = balancer.balance_data(df)
            print(x_train.shape, y_train.shape)
        else:
            x_train = df[self.event_col_name].values
            y_train = df['Label'].values
            assert x_train.shape == y_train.shape
        
        # print("Splitting Test and Train...")
        # x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size=0.8, random_state=42)


        return x_train, y_train #, x_test,y_test

    def suffle_data(self, x_train, y_train):
        rand_train_index = shuffle(np.arange(len(y_train)), random_state=self.random_state)
        x_train = x_train[rand_train_index]
        y_train = y_train[rand_train_index]
        
        train_normal_count = x_train[y_train == 0].shape[0]
        train_abnormal_count = x_train[y_train == 1].shape[0]

        print("Train normal size:", train_normal_count)
        print("Train abnormal size:", train_abnormal_count)

        return x_train,y_train

if __name__ == "__main__":
    # Example usage
    data_dir = "../../logs/Thunderbird/"
    log_file = data_dir + "Thunderbird_20M.log_structured.csv"
    event_col_name = "EventTemplate"
    data_loader = ThunderbirdDataLoader(log_file=log_file, event_col_name = event_col_name, balance_classes=True)
    x_train, y_train = data_loader.load_data()
    print(x_train.shape, y_train.shape)
