import pandas as pd
import os
import re
from collections import OrderedDict

from . import data_balancer


class HDFSLogDataLoader:
    """
    A class for loading and balancing data from HDFS log files.

    Args:
        window (str, optional): The window type for data splitting. Defaults to 'session'.
        train_ratio (float, optional): The ratio of training data to total data. Defaults to 0.5.
        split_type (str, optional): The type of data splitting. Defaults to 'uniform'.
        save_csv (bool, optional): Whether to save the balanced data as CSV files. Defaults to False.
        window_size (int, optional): The size of the window for data splitting. Defaults to 0.

    Attributes:
        log_file (str): Path to the log file.
        label_file (str): Path to the label file.
        log_file_path (str): Directory path of the log file.
        log_file_prefix (str): Prefix of the log file name.

    Methods:
        load_HDFS(log_file, label_file, column_name="EventId", save_classified_data=False):
            Load data from HDFS and balance it.

    Returns:
        tuple: A tuple containing the balanced data and its corresponding labels.
    """

    def __init__(
        self,
        window="session",
        train_ratio=0.5,
        split_type="uniform",
        save_csv=False,
        window_size=0,
    ):
        """
        Initialize the BalancedDataLoader object.

        Parameters:
        - window (str): The type of windowing to be used. Default is 'session'.
        - train_ratio (float): The ratio of training data to be used. Default is 0.5.
        - split_type (str): The type of data split to be used. Default is 'uniform'.
        - save_csv (bool): Whether to save the data as CSV files. Default is False.
        - window_size (int): The size of the window. Default is 0.
        """
        pass

    def load_HDFS(
        self, log_file, label_file, column_name="EventId", save_classified_data=False
    ):
        """
        Load data from HDFS and balance it.

        Args:
            log_file (str): Path to the log file.
            label_file (str): Path to the label file.
            column_name (str, optional): Name of the column to use as input. Defaults to "EventId".
            save_classified_data (bool, optional): Whether to save classified data. Defaults to False.

        Returns:
            tuple: A tuple containing the balanced data and its corresponding labels.
        """
        self.log_file = log_file
        self.label_file = label_file
        self.log_file_path = os.path.dirname(log_file)
        self.log_file_prefix = os.path.basename(log_file).split(".")[0]

        if column_name == "EventId":
            out_col_name = "EventSequence"
        else:
            out_col_name = "EventText"

        struct_log = pd.read_csv(
            self.log_file, engine="c", na_filter=False, memory_map=True
        )
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r"(blk_-?\d+)", row["Content"])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if blk_Id not in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row[column_name])
        data_df = pd.DataFrame(
            list(data_dict.items()), columns=["BlockId", out_col_name]
        )

        label_data = pd.read_csv(
            self.label_file, engine="c", na_filter=False, memory_map=True
        )
        label_data = label_data.set_index("BlockId")
        label_dict = label_data["Label"].to_dict()
        data_df["Label"] = data_df["BlockId"].apply(
            lambda x: 1 if label_dict[x] == "Anomaly" else 0
        )

        balancer = data_balancer.DataBalancer(
            out_col_name=out_col_name,
            save_classified_data=save_classified_data,
            log_file_path=self.log_file_path,
            log_file_prefix=self.log_file_prefix,
        )
        x_data, y_data = balancer.balance_data(data_df)

        return x_data, y_data


# Usage example with class balancing enabled:
if __name__ == "__main__":
    log_file = "../../logs/HDFS/brain/HDFS_500K.log_structured.csv"
    label_file = "../../logs/HDFS/brain//anomaly_label.csv"

    hdfs_loader = HDFSLogDataLoader(
        window="session",
        train_ratio=0.5,
        split_type="sequential",
        save_csv=True,
        window_size=0,
    )
    data, labels = hdfs_loader.load_HDFS(
        log_file, label_file, column_name="EventTemplate", save_classified_data=True
    )
    # save x_train to csv
    # x_train_df = pd.DataFrame(data)
    # x_train_df.to_csv('data.csv', index=False)
