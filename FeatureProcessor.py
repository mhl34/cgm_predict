from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy import signal
from datetime import datetime
from utils import dateParser
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import os

class FlirtFeatureDataset(Dataset):
    def __init__(self):
        super(FlirtFeatureDataset, self).__init__()
        self.samples = [str(i).zfill(3) for i in range(1, 11)]

        # directory
        self.data_dir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2"

        # frequencies
        self.freq_dict = {
            'gluc': 1,
            'hr': 1, 
            'acc': 32, 
            'eda': 4
        } 
        self.max_freq = max(self.freq_dict.values())
        self.resample_str = '1000ms'
        self.normalize = True

    def eda(self, sample, resample_str = '1000ms'):
        """
        function: processes eda data given sample into tonic and phasic components, sampled down to a second
        returns: DataFrame of Tonic EDA, Phasic EDA, and time
        """
        filename = f"{self.data_dir}/{sample}/EDA_{sample}.csv"
        
        # get data
        data = pd.read_csv(filename, skiprows = 1, names=['datetime', 'eda'], index_col = False)
        data['datetime'] = pd.date_range(start = data.iloc[0]['datetime'], periods = len(data), freq = str(1 / self.freq_dict['eda'] * 1000) + 'ms')
        data.set_index('datetime', inplace=True)

        # create dataframe
        eda = data['eda']
        eda_features, _ = nk.eda_process(eda, sampling_rate = self.freq_dict['eda'])
        data['EDA_Tonic'] = eda_features['EDA_Tonic'].values
        data['EDA_Phasic'] = eda_features['EDA_Phasic'].values
        data = data.resample(resample_str).mean()

        eda_dict = {
            'time': data.index.to_numpy(),
            'eda_tonic': data['EDA_Tonic'], 
            'eda_phasic': data['EDA_Phasic']
        }

        return pd.DataFrame(eda_dict)

    def hr(self, sample, resample_str = '1000ms'):
        """
        function: get the heart rate and time, sampled down to a second
        returns: DataFrame of hr and time
        """
        filename = f"{self.data_dir}/{sample}/HR_{sample}.csv"
        
        data = pd.read_csv(filename, skiprows = 1, names = ['datetime', 'hr'], index_col = False)
        data['datetime'] = pd.date_range(start = data.iloc[0]['datetime'], periods = len(data), freq = str(1 / self.freq_dict['hr'] * 1000) + 'ms')
        data.set_index('datetime', inplace=True)
        data = data.resample(resample_str).mean()

        time = data.index.to_numpy()
        hr_data = data['hr']
        
        hr_dict = {
            'time': time,
            'hr': hr_data
        }

        return pd.DataFrame(hr_dict)

    def acc(self, sample, resample_str = '1000ms'):
        """
        function: get the mean accelerometry, averaged across the tri-axial measurement
        returns: DataFrame of acc and time
        note: get formula based on https://support.empatica.com/hc/en-us/articles/202028739-How-is-the-acceleration-data-formatted-in-E4-connect
        """
        filename = f"{self.data_dir}/{sample}/ACC_{sample}.csv"

        data = pd.read_csv(filename, skiprows = 1, names = ['datetime', 'acc_x', 'acc_y', 'acc_z'], index_col = False)
        data['datetime'] = pd.date_range(start = data.iloc[0]['datetime'], periods = len(data), freq = str(1 / self.freq_dict['acc'] * 1000) + 'ms')
        acc_x_diff = np.abs(np.diff(data['acc_x'].to_numpy()))
        acc_y_diff = np.abs(np.diff(data['acc_y'].to_numpy()))
        acc_z_diff = np.abs(np.diff(data['acc_z'].to_numpy()))
        acc_max = np.maximum(np.maximum(acc_x_diff, acc_y_diff), acc_z_diff)
        data = data.iloc[1:]
        data['acc_max'] = acc_max

        data.set_index('datetime', inplace=True)
        data = data.resample(resample_str).mean()
        time = data.index.to_numpy()
        acc_data = data['acc_max']
        
        acc_dict = {
            'time': time,
            'acc': acc_data
        }
        
        return pd.DataFrame(acc_dict)

    def food(self, sample, resample_str = '1000ms', window_sizes = None):
        """
        function: gets carb and sugar consumption for the past window_size input (window_size in minutes)
        returns: DataFrame of carb and sugar consumption
        """
        filename = f"{self.data_dir}/{sample}/Food_Log_{sample}.csv"

        data = pd.read_csv(filename, skiprows = 1, names = ['date', 'time', 'time_begin', 'time_end', 'logged_food', 'amount', 'unit', 'searched_food', 'calorie', 'carb', 'fiber', 'sugar', 'protein', 'fat'], index_col = False)
        data['time_begin'] = pd.to_datetime(data['time_begin'])
        # duplicates are augmented slightly in order to allow for the subsequent indexing
        data['time_begin'] += pd.to_timedelta(data.groupby('time_begin').cumcount(), unit='ms')
        data.set_index('time_begin', inplace=True)
        data = data.resample(resample_str).asfreq()
        data = data.fillna(0)

        new_data = {}
        new_data['time'] = data.index.to_numpy()
        for size in window_sizes:
            window_size = f'{size}T'
            new_data[f'carb_{size}'] = data['carb'].rolling(window = window_size, closed = 'right').sum().to_numpy()
            new_data[f'sugar_{size}'] = data['sugar'].rolling(window = window_size, closed = 'right').sum().to_numpy()
        return pd.DataFrame(new_data)

    def gluc(self, sample):
        """
        function: gets the cgm data and outputs the dataframe of the cgm data and time
        returns: DataFrame of cgm readings
        """
        filename = f"{self.data_dir}/{sample}/Dexcom_{sample}.csv"

        df = pd.read_csv(filename).iloc[12:]
        df = df.fillna(0)
        time = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)']).values
        gluc = df['Glucose Value (mg/dL)'].values
       
        gluc_dict = {
            'time': time, 
            'gluc': gluc
        }
        
        return pd.DataFrame(gluc_dict)


    def run(self):
        big_df = pd.DataFrame()
        if not os.path.exists('data/big_df.pkl'):
            for sample in self.samples:
                print(sample)
                eda = self.eda(sample, resample_str = self.resample_str)
                hr = self.hr(sample, resample_str = self.resample_str)
                acc = self.acc(sample, resample_str = self.resample_str)
                food = self.food(sample, resample_str = self.resample_str, window_sizes = [5, 10, 15, 20, 25, 30, 120, 240])
                gluc = self.gluc(sample)

                # # # merges
                df = pd.merge(gluc, hr, on = 'time')
                df = pd.merge(df, eda, on = 'time')
                df = pd.merge(df, acc, on = 'time')
                df = pd.merge(df, food, on = 'time')

                big_df = pd.concat([big_df, df], axis = 0, ignore_index = True)
            big_df.to_pickle('data/big_df.pkl')
        # drop the time
        df = pd.read_pickle('data/big_df.pkl')
        df = df.drop(columns = ['time'])
        data = df.drop(columns = ['gluc'])

    def normalize_df(self, df):
        """
        function: returns a normalized copy of the dataset
        returns: returns a normalized DataFrame
        """
        normalized_df = df.copy()
        for column in df.columns:
                if column != 'time':
                    mean = df[column].mean()
                    std = df[column].std()
                    normalized_df[column] = (df[column] - mean) / std
        return normalized_df

if __name__ == "__main__":
    obj = FlirtFeatureDataset()
    obj.run()