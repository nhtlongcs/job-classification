
import time
import numpy as np
from tqdm import tqdm
import configparser


def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def get_config_section(config, section):
    if config.has_section(section):
        return dict(config.items(section))
    else:
        raise Exception(f"Section {section} not found in the config file")

# Example usage:
# config = read_config('path/to/config.ini')
# db_config = get_config_section(config, 'database')

class TimedTQDM:
    def __init__(self, iterable):
        self.iterable = iterable
        self.times = []

    def __iter__(self):
        with tqdm(self.iterable) as t:
            for i, item in enumerate(t):
                start_time = time.time()
                yield item
                end_time = time.time()
                self.times.append(end_time - start_time)

    def save_times(self, file_path='iteration_times.npy'):
        times_array = np.array(self.times)
        
        # Calculate mean and standard deviation
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        
        # Save times to file
        np.save(file_path, times_array)
        
        # Print analysis
        print(f"Times saved to {file_path}")
        print(f"Mean iteration time: {mean_time:.5f} seconds")
        print(f"Standard deviation of iteration times: {std_time:.5f} seconds")

