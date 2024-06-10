import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import dateParser
import seaborn as sns
from scipy import signal
from pp5 import pp5_vals
import pickle
import os

sns.set_theme()

class DataProcessor:
    def __init__(self, samples, mainDir = ""):
        self.dexcomFormat = "Dexcom_{0}.csv"
        self.accFormat = "ACC_{0}.csv"
        self.foodLogFormat = "Food_Log_{0}.csv"
        self.ibiFormat = "IBI_{0}.csv"
        self.bvpFormat = "BVP_{0}.csv"
        self.edaFormat = "EDA_{0}.csv"
        self.hrFormat = "HR_{0}.csv"
        self.tempFormat = "TEMP_{0}.csv"
        self.mainDir = mainDir
        self.samples = samples

        # method parameters
        self.food_time = 6

        # alignment parameters
        self.alignment_dict = self.get_alignments(["dexcom", "temp", "eda", "acc", "hr"])

        # get lengths for downsampling
        self.seq_len_dict = self.get_seq_lens(samples)
      
    def loadData(self, samples, fileType):
        data = {}
        pp5 = pp5_vals()
        if fileType == "dexcom":
            mean = 0
            std = 0
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
                lst = pd.to_numeric(df['Glucose Value (mg/dL)']).to_numpy()
                start, end = self.alignment_dict[sample][fileType]
                aligned_lst = lst[start:end]
                mean += np.nanmean(aligned_lst)
                std += np.nanstd(aligned_lst)
                data[sample] = aligned_lst
            data['mean'] = mean / len(samples)
            data['std'] = std / len(samples)
        elif fileType == "temp":
            mean = 0
            std = 0
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.tempFormat.format(sample))
                lst = pd.to_numeric(df[' temp']).to_numpy()
                lst = lst.astype(np.int64)
                start, end = self.alignment_dict[sample][fileType]
                aligned_lst = lst[start:end]
                # downsample for alignment with glucose data (based on pp5)
                downsampled_temp = signal.resample(aligned_lst, self.seq_len_dict[sample])
                mean += np.nanmean(downsampled_temp)
                std += np.nanstd(downsampled_temp)
                data[sample] = downsampled_temp
        elif fileType == "eda":
            mean = 0
            std = 0
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.edaFormat.format(sample))
                lst = pd.to_numeric(df[' eda']).to_numpy()
                start, end = self.alignment_dict[sample][fileType]
                aligned_lst = lst[start:end]
                # downsample for alignment with glucose data (based on pp5)
                downsampled_eda = signal.resample(aligned_lst, self.seq_len_dict[sample])
                mean += np.nanmean(downsampled_eda)
                std += np.nanstd(downsampled_eda)
                data[sample] = downsampled_eda
        elif fileType == "hr":
            mean = 0
            std = 0
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.hrFormat.format(sample))
                lst = pd.to_numeric(df[' hr']).to_numpy()
                start, end = self.alignment_dict[sample][fileType]
                aligned_lst = lst[start:end]
                # downsample for alignment with glucose data (based on pp5)
                downsampled_hr = signal.resample(aligned_lst, self.seq_len_dict[sample])
                mean += np.nanmean(downsampled_hr)
                std += np.nanstd(downsampled_hr)
                data[sample] = downsampled_hr
        elif fileType == "acc":
            mean_x = 0
            mean_y = 0
            mean_z= 0
            std_x = 0
            std_y = 0
            std_z = 0
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.accFormat.format(sample))
                lst_x = pd.to_numeric(df[' acc_x']).to_numpy()
                lst_y = pd.to_numeric(df[' acc_y']).to_numpy()
                lst_z = pd.to_numeric(df[' acc_z']).to_numpy()
                start, end = self.alignment_dict[sample][fileType]
                aligned_lst_x = lst_x[start:end]
                aligned_lst_y = lst_y[start:end]
                aligned_lst_z = lst_z[start:end]
                # downsample for alignment with glucose data (based on pp5)
                downsampled_acc_x = signal.resample(aligned_lst_x, self.seq_len_dict[sample])
                downsampled_acc_y = signal.resample(aligned_lst_y, self.seq_len_dict[sample])
                downsampled_acc_z = signal.resample(aligned_lst_z, self.seq_len_dict[sample])
                mean_x += np.nanmean(downsampled_acc_x)
                mean_y += np.nanmean(downsampled_acc_y)
                mean_z += np.nanmean(downsampled_acc_z)
                std_x += np.nanstd(downsampled_acc_x)
                std_y += np.nanstd(downsampled_acc_y)
                std_z += np.nanstd(downsampled_acc_z)
                data[sample] = (downsampled_acc_x, downsampled_acc_y, downsampled_acc_z)
            data['mean_x'] = mean_x / len(samples)
            data['mean_y'] = mean_y / len(samples)
            data['mean_z'] = mean_z / len(samples)
            data['std_x'] = std_x / len(samples)
            data['std_y'] = std_y / len(samples)
            data['std_z'] = std_z / len(samples)
        elif fileType == "food":
            column_names = ["date", "time", "time_begin", "time_end", "logged_food", "amount", "unit", "searched_food", "calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]
            mean_sugar = 0
            mean_carb = 0
            std_sugar = 0
            std_carb = 0
            for sample in samples:
                food_df = pd.read_csv(self.mainDir + sample + "/" + self.foodLogFormat.format(sample), sep =',', names = column_names)
                dexcom_df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
                ms, mc, ss, sc, data[sample] = self.processFood(food_df, dexcom_df, sample)
                mean_sugar += ms
                mean_carb += mc
                std_sugar += ss
                std_carb += sc
            data['mean_sugar'] = mean_sugar / len(samples)
            data['mean_carb'] = mean_carb / len(samples)
            data['std_sugar'] = std_sugar / len(samples)
            data['std_carb'] = std_carb / len(samples)
        else:
            for sample in samples:
                data[sample] = np.array([])
        if fileType != "dexcom" and fileType != "food" and fileType != "acc":
            data['mean'] = mean / len(samples)
            data['std'] = std / len(samples)
        return data
    
    def persComp(self, value, persHigh, persLow):
        if value > persHigh:
            return "PersHigh"
        elif value < persLow:
            return "PersLow"
        return "PersNorm"
    
    def persValue(self, data, swData):
        persDict = {}
        for key in data.keys():
            persDict[key] = [(data[key][i], self.persComp(data[key][i], swData[key][i][0] + swData[key][i][1], swData[key][i][0] - swData[key][i][1])) for i in range(len(data[key]))]
        return persDict
    
    def processFood(self, food_df, dexcom_df, sample):
        sugar_array = food_df['sugar'].to_numpy()[1:]
        carb_array = food_df['total_carb'].to_numpy()[1:]
        # base it off of time_begin
        food_time_array = np.array(list(map(dateParser, food_df['time_begin'].to_numpy()[1:])))
        # align gluc_time_array
        start, end = self.alignment_dict[sample]['dexcom']
        gluc_time_array = np.array(list(map(dateParser, dexcom_df['Timestamp (YYYY-MM-DDThh:mm:ss)'].to_numpy())))[start: end]
        # gluc_time_array = np.array(list(map(self.getMins, gluc_time_array)))
        # iterate through the gluc_time_array and food_time 
        gluc_idx = 0
        food_idx = 0
        # iterate through gluc and food
        # {gluc_idx1: [], gluc_idx2: [], ...}
        sugar_np_array= np.zeros(len(gluc_time_array))
        carb_np_array = np.zeros(len(gluc_time_array))
        while food_idx != len(food_time_array) - 1 and gluc_idx != len(gluc_time_array) - 1:
            # check if the food_time is within 24 hours of the gluc time
            # 4 cases
            # 1) food_time < gluc_time and > 24 hours --> food_idx += 1
            # 2) food_time < gluc_time and <= 24 hours --> gluc_dict[gluc_idx] += sugar_array[food_idx]; food_idx += 1
            # 3) food_time > gluc_time and <= 24 hours --> 
            # 4) food_time > gluc_time and > 24 hours --> gluc_idx += 1
            food_time = food_time_array[food_idx]
            gluc_time = gluc_time_array[gluc_idx]
            if not isinstance(food_time, datetime):
                food_idx += 1
                continue
            if not isinstance(gluc_time, datetime):
                gluc_idx += 1
                continue
            time_diff = abs(food_time - gluc_time)
            if food_time < gluc_time and time_diff > timedelta(hours = self.food_time):
                food_idx += 1
                continue
            elif food_time < gluc_time and time_diff <= timedelta(hours = self.food_time):
                sugar_np_array[gluc_idx] += float(sugar_array[food_idx])
                carb_np_array[gluc_idx] += float(carb_array[food_idx])
                food_idx += 1
            # elif food_time > gluc_time and time_diff <= timedelta(hours = 24):
            #     sugar_np_array[gluc_idx] += float(sugar_array[food_idx])
            #     carb_np_array[gluc_idx] += float(carb_array[food_idx])
            #     food_idx += 1
            else:
                gluc_idx += 1
                food_idx = 0
                continue
        return np.nanmean(sugar_np_array), np.nanmean(carb_np_array), np.nanstd(sugar_np_array), np.nanstd(carb_np_array), (sugar_np_array, carb_np_array)
    
    def hba1c(self, samples):
        d = {}
        # df = pd.read_csv(self.mainDir + "Demographics.txt", sep='\t')
        df = pd.read_csv(self.mainDir + "Demographics.csv", sep=',')
        for sample in samples:
            hba1c = df.loc[df['ID'] == int(sample)]['HbA1c'].item()
            dexcom_df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
            size = len(dexcom_df)
            d[sample] = np.ones(size) * hba1c
        d['mean'] = hba1c
        d['std'] = 0
        return d

    def minFromMidnight(self, samples):
        data = {}
        mean = 0
        std = 0
        for sample in samples:
            df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
            time_array = np.array(list(map(dateParser, df['Timestamp (YYYY-MM-DDThh:mm:ss)'].to_numpy())))
            min_array = np.array(list(map(self.getMins, time_array)))
            mean += np.nanmean(min_array)
            std += np.nanstd(min_array)
            data[sample] = min_array
        data['mean'] = mean / len(samples)
        data['std'] = std / len(samples)
        return data

    def getMins(self, time):
        try:
            return time.hour * 60 + time.minute + time.second / 60
        except:
            return np.nan

    def get_seq_lens(self, samples):
        data = {}
        for sample in samples:
            start, end = self.alignment_dict[sample]['dexcom']
            data[sample] = end - start
        return data

    def get_alignments(self, file_list):
        """
        get start and end indices to align the sequences by time
        format: {sample: {filename: (start, end) ... }}
        """
        if os.path.exists('data/alignment_dict.pkl'):
            with open('data/alignment_dict.pkl', 'rb') as f:
                alignment_dict = pickle.load(f)
                return alignment_dict
        sample_dict = {}
        for sample in self.samples:
            print(f"sample {sample}")
            dfs = {}
            for fileType in file_list:
                print(f"file {fileType}")
                if fileType == "dexcom":
                    df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
                    time_array = np.array(list(map(dateParser, df['Timestamp (YYYY-MM-DDThh:mm:ss)'].to_numpy())))
                    df['time'] = time_array
                    dfs[fileType] = df
                elif fileType == "temp":
                    df = pd.read_csv(self.mainDir + sample + "/" + self.tempFormat.format(sample))
                    time_array = np.array(list(map(dateParser, df['datetime'].to_numpy())))
                    df['time'] = time_array
                    dfs[fileType] = df
                elif fileType == "eda":
                    df = pd.read_csv(self.mainDir + sample + "/" + self.edaFormat.format(sample))
                    time_array = np.array(list(map(dateParser, df['datetime'].to_numpy())))
                    df['time'] = time_array
                    dfs[fileType] = df
                elif fileType == "hr":
                    df = pd.read_csv(self.mainDir + sample + "/" + self.hrFormat.format(sample))
                    time_array = np.array(list(map(dateParser, df['datetime'].to_numpy())))
                    df['time'] = time_array
                    dfs[fileType] = df
                elif fileType == "acc":
                    df = pd.read_csv(self.mainDir + sample + "/" + self.accFormat.format(sample))
                    time_array = np.array(list(map(dateParser, df['datetime'].to_numpy())))
                    df['time'] = time_array
                    dfs[fileType] = df
                else:
                    raise Exception("Unsupported file type")

            start_times = [df['time'].min() for df in dfs.values()]
            end_times = [df['time'].max() for df in dfs.values()]
            common_start = max(start_times)
            common_end = min(end_times)
            
            indices = {filename: (df[df['time'] >= common_start].index[0], df[df['time'] <= common_end].index[-1]) 
                    for filename, df in dfs.items()}
            
            sample_dict[sample] = indices
        with open('data/alignment_dict.pkl', 'wb') as f:
            pickle.dump(sample_dict, f)
        return sample_dict
