from torch.utils.data import Dataset
import random
from pp5 import pp5
import numpy as np
import torch
from utils import createGlucStats

class FeatureDataset(Dataset):
    def __init__(self, samples, glucose, eda, hr, temp, acc, food, minutes, hba1c, metric = "mean", dtype = torch.float64, seq_length = 28, transforms = None):
        self.samples = samples
        self.glucose = glucose
        self.eda = eda
        self.hr = hr
        self.temp = temp
        self.acc = acc
        self.food = food
        self.minutes = minutes
        self.hba1c = hba1c
        self.seq_length = seq_length
        self.pp5vals = pp5()
        self.metric = metric
        self.dtype = dtype
        self.transforms = transforms

    def __len__(self):
        return self.glucose[self.samples[0]].__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        # the features that we want are minutes, sugar, carbs, hba1c, acc
        sample = random.choice(self.samples)
        # get the different samples
        glucoseSample = self.glucose[sample]
        sugarSample, carbSample = self.food[sample]
        # edaSample = self.eda[sample]
        # hrSample = self.hr[sample]
        # tempSample = self.temp[sample]
        # accSample = self.acc[sample]
        minSample = self.minutes[sample]
        hba1cSample = self.hba1c[sample]

        # sugar, carb, min, glucose, and hba1c (constant) are sampled at the same rate
        # acc needs a factor
        
        # get the start indices for different sequences
        glucStart = sugarStart = carbStart = minStart = hba1cStart = index
        # accStart = glucStart * self.pp5vals.acc
        
        glucTruth = self.truthCheck(glucoseSample, glucStart, "glucose")
        sugarTruth = self.truthCheck(sugarSample, glucStart, "gluc_other")
        carbTruth = self.truthCheck(carbSample, carbStart, "gluc_other")
        minTruth = self.truthCheck(minSample, minStart, "gluc_other")
        hba1cTruth = self.truthCheck(hba1cSample, hba1cStart, "gluc_other")
        # accTruth = self.truthCheck(accSample, accStart, "acc")
        
        # while glucTruth or sugarTruth or carbTruth or minTruth or hba1cTruth or accTruth:
        while glucTruth or sugarTruth or carbTruth or minTruth or hba1cTruth:
            idx = random.randint(0,len(glucoseSample) - 2 * self.seq_length - 1)
            glucStart = sugarStart = carbStart = minStart = hba1cStart = idx
            accStart = glucStart * self.pp5vals.acc
            glucTruth = self.truthCheck(glucoseSample, glucStart, "glucose")
            sugarTruth = self.truthCheck(sugarSample, glucStart, "gluc_other")
            carbTruth = self.truthCheck(carbSample, carbStart, "gluc_other")
            minTruth = self.truthCheck(minSample, minStart, "gluc_other")
            hba1cTruth = self.truthCheck(hba1cSample, hba1cStart, "gluc_other")
            # accTruth = self.truthCheck(accSample, accStart, "acc")
            
        glucosePastSec = glucoseSample[glucStart: glucStart + self.seq_length]
        glucoseSec = glucoseSample[glucStart + self.seq_length + 1: glucStart + 2 * self.seq_length + 1]
        sugarSec = sugarSample[sugarStart: sugarStart + self.seq_length]
        carbSec = carbSample[carbStart: carbStart + self.seq_length]
        minSec = minSample[minStart: minStart + self.seq_length]
        hba1cSec = hba1cSample[hba1cStart: hba1cStart + self.seq_length]
        # accSec = accSample[accStart: accStart + self.seq_length * self.pp5vals.acc]
        # glucStats = createGlucStats(glucoseSec)
        
        # create averages across sequence length
        # accMean = np.array(list(map(np.mean, np.array_split(accSec, self.seq_length))))
        sugarMean = torch.Tensor(np.array(list(map(np.mean, np.array_split(sugarSec, self.seq_length)))))
        carbMean = torch.Tensor(np.array(list(map(np.mean, np.array_split(carbSec, self.seq_length)))))
        minMean = torch.Tensor(np.array(list(map(np.mean, np.array_split(minSec, self.seq_length)))))
        hba1cMean = torch.Tensor(np.array(list(map(np.mean, np.array_split(hba1cSec, self.seq_length)))))
        glucPastMean = torch.Tensor(np.array(list(map(np.mean, np.array_split(glucosePastSec, self.seq_length)))))
        glucMean = torch.Tensor(np.array(list(map(np.mean, np.array_split(glucoseSec, self.seq_length)))))

        # normalize if needed
        output = torch.stack((sugarMean, carbMean, minMean, hba1cMean, glucPastMean, glucMean)).unsqueeze(1)
        if self.transforms != None:
            # return (sample, self.normalizeFn(accMean), self.normalizeFn(sugarMean), self.normalizeFn(carbMean), self.normalizeFn(minMean), self.normalizeFn(hba1cMean), self.normalizeFn(glucPastMean), self.normalizeFn(glucMean))
            output = self.transforms(output)
        
        #return non-normalized outputs
        # return (sample, accMean, sugarMean, carbMean, minMean, hba1cMean, glucPastMean, glucMean)
        return output.to(self.dtype)
    
    def normalizeFn(self, data, eps = 1e-5):
        data = data[~np.isnan(data)]
        data_min = np.min(data)
        data_max = np.max(data)
        scaled_data = (data - data_min) / (data_max - data_min + eps)
        return scaled_data
    
    def truthCheck(self, sample_array, sample_start, sample_type):
        if sample_type == "glucose":
            return True in np.isnan(sample_array[sample_start: sample_start + 2 * self.seq_length + 1]) or len(sample_array[sample_start: sample_start  + 2 * self.seq_length + 1]) != 2 * self.seq_length + 1
        pp5val_dict = {"eda": self.pp5vals.eda, "hr": self.pp5vals.hr, "temp": self.pp5vals.temp, "acc": self.pp5vals.acc, "gluc_other": 1}
        return True in np.isnan(sample_array[sample_start: sample_start + pp5val_dict[sample_type] * self.seq_length + 1]) or len(sample_array[sample_start: sample_start + pp5val_dict[sample_type] * self.seq_length + 1]) != pp5val_dict[sample_type] * self.seq_length + 1
