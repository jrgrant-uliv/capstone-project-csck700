import os
import pandas as pd
import numpy as np
from logparser import Spell, Drain
from tqdm import tqdm

class ThunderbirdLogProcessor:
    def __init__(self, data_dir, output_dir, raw_log_file, sample_log_file, sample_window_size, sample_step_size,
                 parser_type, window_size, step_size, train_ratio, window_name=''):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.raw_log_file = raw_log_file
        self.sample_log_file = sample_log_file
        self.sample_window_size = sample_window_size
        self.sample_step_size = sample_step_size
        self.parser_type = parser_type
        self.window_size = window_size
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.window_name = window_name

    def count_anomaly(self, log_path):
        """
        Count the number of anomaly log messages in a given log file.
        """
        total_size = 0
        normal_size = 0
        with open(log_path, errors='ignore') as f:
            for line in f:
                total_size += 1
                if line.split('')[0] == '-':
                    normal_size += 1
        print("Total size: {}, abnormal size: {}".format(total_size, total_size - normal_size))

    def deeplog_file_generator(self, filename, df, features):
        """
        Generate a file with log data for the DeepLog model.
        """
        with open(filename, 'w') as f:
            for _, row in df.iterrows():
                for val in zip(*row[features]):
                    f.write(','.join([str(v) for v in val]) + ' ')
                f.write('\n')

    def parse_log(self, input_dir, output_dir, log_file):
        """
        Parse log files using either the "Drain" or "Spell" log parsers.
        """
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
        regex = [
            r'(0x)[0-9a-fA-F]+',  # hexadecimal
            r'\d+\.\d+\.\d+\.\d+',
            r'(?<=Warning: we failed to resolve data source name )[\w\s]+',
            r'\d+'
        ]
        keep_para = False

        if self.parser_type == "drain":
            st = 0.3  # Similarity threshold
            depth = 3  # Depth of all leaf nodes
            parser = Drain.LogParser(log_format,
                                     indir=input_dir,
                                     outdir=output_dir,
                                     depth=depth,
                                     st=st,
                                     rex=regex,
                                     keep_para=keep_para, maxChild=1000)
            parser.parse(log_file)
        elif self.parser_type == "spell":
            tau = 0.35
            parser = Spell.LogParser(indir=input_dir,
                                     outdir=output_dir,
                                     log_format=log_format,
                                     tau=tau,
                                     rex=regex,
                                     keep_para=keep_para)
            parser.parse(log_file)

    def sample_raw_data(self, data_file, output_file):
        """
        Sample raw log data by sliding a window to create a subset for analysis.
        """
        sample_data = []
        labels = []
        idx = 0

        with open(data_file, 'r', errors='ignore') as f:
            for line in f:
                labels.append(line.split()[0] != '-')
                sample_data.append(line)

                if len(labels) == self.sample_window_size:
                    abnormal_rate = sum(np.array(labels)) / len(labels)
                    print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                    break

                idx += 1
                if idx % self.sample_step_size == 0:
                    print(f"Process {round(idx / self.sample_window_size * 100, 4)} % raw data", end='\r')

        with open(output_file, "w") as f:
            f.writelines(sample_data)

        print("Sampling done")

    def preprocess_log_data(self):
        # Count anomalies in the log file (if needed)
        # self.count_anomaly(os.path.join(self.data_dir, self.sample_log_file))

        # Sample raw log data
        self.sample_raw_data(
            os.path.join(self.data_dir, self.raw_log_file),
            os.path.join(self.data_dir, self.sample_log_file)
        )

        # Parse log data
        self.parse_log(self.data_dir, self.output_dir, self.sample_log_file)

        # Additional preprocessing steps can be added here.

if __name__ == "__main__":
    # Example usage of the ThunderbirdLogProcessor class
    data_dir = os.path.expanduser("../../dataset/tbird/")
    output_dir = "../../output/tbird/"
    raw_log_file = "Thunderbird.log"
    sample_log_file = "Thunderbird_20M.log"
    sample_window_size = 2 * 10**7
    sample_step_size = 10**4
    window_name = ''
    parser_type = 'drain'
    window_size = 1
    step_size = 0.5
    train_ratio = 6000

    log_processor = ThunderbirdLogProcessor(
        data_dir, output_dir, raw_log_file, sample_log_file, sample_window_size, sample_step_size,
        parser_type, window_size, step_size, train_ratio, window_name
    )

    log_processor.preprocess_log_data()
