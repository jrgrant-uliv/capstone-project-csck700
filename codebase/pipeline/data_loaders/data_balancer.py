import os
import pandas as pd
from sklearn.utils import resample


class DataBalancer:
    """
    A class to balance the data in a given DataFrame by resampling the minority class to match the majority class.

    Attributes:
        out_col_name (str): The name of the output column.
        save_classified_data (bool): Whether to save the classified data or not.
        log_file_path (str): The path to the log file.
        log_file_prefix (str): The prefix for the log file.

    Methods:
        balance_data(data_df): Balances the data in the given DataFrame by resampling the minority class to match the majority class.
    """

    def __init__(self, out_col_name, save_classified_data=False, log_file_path='', log_file_prefix=''):
        """
        Initialize the DataBalancer object.

        Parameters:
        - out_col_name (str): The name of the output column.
        - save_classified_data (bool): Flag indicating whether to save the classified data.
        - log_file_path (str): The path to the log file.
        - log_file_prefix (str): The prefix for the log file.

        Returns:
        None
        """
        self.out_col_name = out_col_name
        self.save_classified_data = save_classified_data
        self.log_file_path = log_file_path
        self.log_file_prefix = log_file_prefix

    def balance_data(self, data_df):
        """
        Balances the data in the given DataFrame by resampling the minority class to match the majority class.

        Args:
            data_df (pandas.DataFrame): The DataFrame containing the data to be balanced.

        Returns:
            tuple: A tuple containing the balanced data and labels as numpy arrays.
        """

        # Step 1: Split the data into two separate DataFrames based on the labels
        class_0 = data_df[data_df['Label'] == 0]
        class_1 = data_df[data_df['Label'] == 1]

        # Step 2: Determine the class with the fewer samples, and resample the other class to match
        balanced_class_0, balanced_class_1 = self.resample(class_0, class_1)

        if 'EventId' in data_df.columns:
            # Step 3: Reduce the occurance imbalance
            balanced_class_1 = self.reduce_occurance_imbalance(
                balanced_class_1)
            balanced_class_0 = self.reduce_occurance_imbalance(
                balanced_class_0)

            # step 4: Resample again to ensure the classes are balanced after reducing the occurance imbalance
            balanced_class_0, balanced_class_1 = self.resample(
                balanced_class_0, balanced_class_1)

        # Step 5: Save balanced data if requested
        if self.save_classified_data:
            # ensure path exists
            if not os.path.exists(self.log_file_path):
                os.makedirs(self.log_file_path)
            f_normal = os.path.join(
                self.log_file_path, self.log_file_prefix + '.normal.csv')
            f_anomalous = os.path.join(
                self.log_file_path, self.log_file_prefix + '.anomalous.csv')
            balanced_class_0.to_csv(f_normal, index=False)
            balanced_class_1.to_csv(f_anomalous, index=False)

        # Step 6: Combine the balanced data back into a single DataFrame
        balanced_data_df = pd.concat([balanced_class_0, balanced_class_1])

        # Step 7: Shuffle the DataFrame to ensure randomness
        balanced_data_df = balanced_data_df.sample(
            frac=1, random_state=42).reset_index(drop=True)

        # Step 8: Extract the balanced data and labels
        x_data = balanced_data_df[self.out_col_name].values
        y_data = balanced_data_df['Label'].values

        return x_data, y_data

    def resample(self, class_0, class_1):
        """
        Resamples the given classes to have an equal number of samples.

        Args:
            class_0 (list): The samples from class 0.
            class_1 (list): The samples from class 1.

        Returns:
            tuple: A tuple containing the resampled samples from class 0 and class 1.
        """
        min_samples = min(len(class_0), len(class_1))

        balanced_class_0 = resample(
            class_0, n_samples=min_samples, random_state=42)
        balanced_class_1 = resample(
            class_1, n_samples=min_samples, random_state=42)
        return balanced_class_0, balanced_class_1

    def reduce_occurance_imbalance(self, balanced_class_1):
        """
        Reduces occurrence imbalance in the given balanced_class_1 DataFrame.

        Args:
            balanced_class_1 (DataFrame): The DataFrame containing the balanced class 1 data.

        Returns:
            DataFrame: The reduced DataFrame with occurrence imbalance addressed.
        """
        event_counts = balanced_class_1['EventId'].value_counts()
        event_counts = event_counts.to_frame()
        event_counts = event_counts.reset_index()
        event_counts.columns = ['EventId', 'count']
        second_most_frequent_class = event_counts['count'].iloc[1]
        most_frequent_class = event_counts['count'].iloc[0]
        if most_frequent_class > (second_most_frequent_class * 1.5):
            print("most frequent class is more than 150% of the second most frequent class")
            print("most frequent class: " + str(most_frequent_class))
            print("second most frequent class: " + str(second_most_frequent_class))
            print("reducing most frequent class to 150% of the second most frequent class")
            most_frequent_class = int(second_most_frequent_class * 1.5)
            print("most frequent class: " + str(most_frequent_class))
            print("second most frequent class: " + str(second_most_frequent_class))
            print("reducing most frequent class to 150% of the second most frequent class")
            balanced_class_1 = balanced_class_1.groupby('EventId').head(most_frequent_class)
            print(balanced_class_1['EventId'].value_counts())
        return balanced_class_1


# Example usage of the DataBalancer class
if __name__ == "__main__":
    data_df = pd.read_csv('data_instances.csv')
    balancer = DataBalancer(out_col_name='EventSequence', save_classified_data=True,
                            log_file_path='./output', log_file_prefix='balanced_data')
    x_data, y_data = balancer.balance_data(data_df)
