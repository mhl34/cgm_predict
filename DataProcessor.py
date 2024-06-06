import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import dateParser
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

class DataProcessor:
    def __init__(self, mainDir = ""):
        self.dexcomFormat = "Dexcom_{0}.csv"
        self.accFormat = "ACC_{0}.csv"
        self.foodLogFormat = "Food_Log_{0}.csv"
        self.ibiFormat = "IBI_{0}.csv"
        self.bvpFormat = "BVP_{0}.csv"
        self.edaFormat = "EDA_{0}.csv"
        self.hrFormat = "HR_{0}.csv"
        self.tempFormat = "TEMP_{0}.csv"
        self.mainDir = mainDir

        # method parameters
        self.food_time = 6
      
    def loadData(self, samples, fileType):
        data = {}
        if fileType == "dexcom":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
                lst = pd.to_numeric(df['Glucose Value (mg/dL)']).to_numpy()
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "temp":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.tempFormat.format(sample))
                lst = pd.to_numeric(df[' temp']).to_numpy()
                lst = lst.astype(np.int64)
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "eda":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.edaFormat.format(sample))
                lst = pd.to_numeric(df[' eda']).to_numpy()
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "hr":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.hrFormat.format(sample))
                lst = pd.to_numeric(df[' hr']).to_numpy()
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "acc":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.accFormat.format(sample))
                lst_x = pd.to_numeric(df[' acc_x']).to_numpy()
                lst_y = pd.to_numeric(df[' acc_y']).to_numpy()
                lst_z = pd.to_numeric(df[' acc_z']).to_numpy()
                data[sample] = np.sqrt(np.sum([np.square(lst_x), np.square(lst_y), np.square(lst_z)], axis=0))
        elif fileType == "food":
            column_names = ["date", "time", "time_begin", "time_end", "logged_food", "amount", "unit", "searched_food", "calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]
            for sample in samples:
                food_df = pd.read_csv(self.mainDir + sample + "/" + self.foodLogFormat.format(sample), sep =',', names = column_names)
                dexcom_df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
                data[sample] = self.processFood(food_df, dexcom_df)

                # plot for sanity check
                plt.clf()
                plt.plot(np.arange(len(data[sample][0])), data[sample][0])
                plt.plot(np.arange(len(data[sample][1])), data[sample][1])
                plt.savefig(f"plots/{sample}_food_log.png")
        else:
            for sample in samples:
                data[sample] = np.array([])
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
    
    def processFood(self, food_df, dexcom_df):
        sugar_array = food_df['sugar'].to_numpy()[1:]
        carb_array = food_df['total_carb'].to_numpy()[1:]
        # base it off of time_begin
        food_time_array = np.array(list(map(dateParser, food_df['time_begin'].to_numpy()[1:])))
        gluc_time_array = np.array(list(map(dateParser, dexcom_df['Timestamp (YYYY-MM-DDThh:mm:ss)'].to_numpy())))
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
        return (sugar_np_array, carb_np_array)
    
    def hba1c(self, samples):
        d = {}
        # df = pd.read_csv(self.mainDir + "Demographics.txt", sep='\t')
        df = pd.read_csv(self.mainDir + "Demographics.csv", sep=',')
        for sample in samples:
            hba1c = df.loc[df['ID'] == int(sample)]['HbA1c'].item()
            dexcom_df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
            size = len(dexcom_df)
            d[sample] = np.ones(size) * hba1c
        return d

    def minFromMidnight(self, samples):
        data = {}
        for sample in samples:
            df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
            time_array = np.array(list(map(dateParser, df['Timestamp (YYYY-MM-DDThh:mm:ss)'].to_numpy())))
            min_array = np.array(list(map(self.getMins, time_array)))
            data[sample] = min_array
        return data

    def getMins(self, time):
        try:
            return time.hour * 60 + time.minute + time.second / 60
        except:
            return np.nan
