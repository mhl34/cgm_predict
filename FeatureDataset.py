from torch.utils.data import Dataset
import random
from pp5 import pp5_vals
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
        self.pp5vals = pp5_vals()
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
        edaSample = self.eda[sample]
        hrSample = self.hr[sample]
        tempSample = self.temp[sample]
        acc_xSample, acc_ySample, acc_zSample = self.acc[sample]
        minSample = self.minutes[sample]
        hba1cSample = self.hba1c[sample]
        
        # print("Length of samples:")
        # print(f"Glucose Sample: {len(glucoseSample)}")
        # print(f"Sugar Sample: {len(sugarSample)}")
        # print(f"Carb Sample: {len(carbSample)}")
        # print(f"EDA Sample: {len(edaSample)}")
        # print(f"HR Sample: {len(hrSample)}")
        # print(f"Temp Sample: {len(tempSample)}")
        # print(f"ACC Sample: {len(acc_xSample)}")
        # print(f"Minutes Sample: {len(minSample)}")
        # print(f"HbA1c Sample: {len(hba1cSample)}")

        idx = index
        
        glucTruth = self.truthCheck(glucoseSample, idx, "glucose")
        sugarTruth = self.truthCheck(sugarSample, idx, "gluc_other")
        carbTruth = self.truthCheck(carbSample, idx, "gluc_other")
        minTruth = self.truthCheck(minSample, idx, "gluc_other")
        hba1cTruth = self.truthCheck(hba1cSample, idx, "gluc_other")
        edaTruth = self.truthCheck(edaSample, idx, "eda")
        hrTruth = self.truthCheck(hrSample, idx, "hr")
        tempTruth = self.truthCheck(tempSample, idx, "temp")
        acc_xTruth = self.truthCheck(acc_xSample, idx, "acc")
        acc_yTruth = self.truthCheck(acc_ySample, idx, "acc")
        acc_zTruth = self.truthCheck(acc_zSample, idx, "acc")
        
        # while glucTruth or sugarTruth or carbTruth or minTruth or hba1cTruth or accTruth:
        while glucTruth or sugarTruth or carbTruth or minTruth or hba1cTruth or edaTruth or hrTruth or tempTruth or acc_xTruth or acc_yTruth or acc_zTruth:
            idx = random.randint(0,len(glucoseSample) - 2 * self.seq_length - 1)
            glucTruth = self.truthCheck(glucoseSample, idx, "glucose")
            sugarTruth = self.truthCheck(sugarSample, idx, "gluc_other")
            carbTruth = self.truthCheck(carbSample, idx, "gluc_other")
            minTruth = self.truthCheck(minSample, idx, "gluc_other")
            hba1cTruth = self.truthCheck(hba1cSample, idx, "gluc_other")
            edaTruth = self.truthCheck(edaSample, idx, "eda")
            hrTruth = self.truthCheck(hrSample, idx, "hr")
            tempTruth = self.truthCheck(tempSample, idx, "temp")
            acc_xTruth = self.truthCheck(acc_xSample, idx, "acc")
            acc_yTruth = self.truthCheck(acc_ySample, idx, "acc")
            acc_zTruth = self.truthCheck(acc_zSample, idx, "acc")
            
        glucosePastSec = glucoseSample[idx: idx + self.seq_length]
        glucoseSec = glucoseSample[idx + self.seq_length + 1: idx + 2 * self.seq_length + 1]
        sugarSec = sugarSample[idx: idx + self.seq_length]
        carbSec = carbSample[idx: idx + self.seq_length]
        minSec = minSample[idx: idx + self.seq_length]
        hba1cSec = hba1cSample[idx: idx + self.seq_length]
        edaSec = edaSample[idx: idx + self.seq_length]
        hrSec = hrSample[idx: idx + self.seq_length]
        tempSec = tempSample[idx: idx + self.seq_length]
        acc_xSec = acc_xSample[idx: idx + self.seq_length]
        acc_ySec = acc_ySample[idx: idx + self.seq_length]
        acc_zSec = acc_zSample[idx: idx + self.seq_length]
        
        # create averages across sequence length
        # accMean = np.array(list(map(np.mean, np.array_split(accSec, self.seq_length))))
        sugarTensor = torch.Tensor(np.array(sugarSec))
        carbTensor = torch.Tensor(np.array(carbSec))
        minTensor = torch.Tensor(np.array(minSec))
        hba1cTensor = torch.Tensor(np.array(hba1cSec))
        glucPastTensor = torch.Tensor(np.array(glucosePastSec))
        glucTensor = torch.Tensor(np.array(glucoseSec))
        edaTensor = torch.Tensor(np.array(edaSec))
        hrTensor = torch.Tensor(np.array(hrSec))
        tempTensor = torch.Tensor(np.array(tempSec))
        acc_xTensor = torch.Tensor(np.array(acc_xSec))
        acc_yTensor = torch.Tensor(np.array(acc_ySec))
        acc_zTensor = torch.Tensor(np.array(acc_zSec))

        # normalize if needed
        output = torch.stack((sugarTensor, carbTensor, minTensor, hba1cTensor, edaTensor, hrTensor, tempTensor, acc_xTensor, acc_yTensor, acc_zTensor, glucPastTensor, glucTensor)).unsqueeze(1)
        if self.transforms != None:
            # return (sample, self.normalizeFn(accMean), self.normalizeFn(sugarMean), self.normalizeFn(carbMean), self.normalizeFn(minMean), self.normalizeFn(hba1cMean), self.normalizeFn(glucPastMean), self.normalizeFn(glucMean))
            output = self.transforms(output)
        
        #return non-normalized outputs
        # return (sample, accMean, sugarMean, carbMean, minMean, hba1cMean, glucPastMean, glucMean)
        return output.to(self.dtype)
    
    def truthCheck(self, sample_array, sample_start, sample_type):
        if sample_type == "glucose":
            return True in np.isnan(sample_array[sample_start: sample_start + 2 * self.seq_length + 1]) or len(sample_array[sample_start: sample_start  + 2 * self.seq_length + 1]) != 2 * self.seq_length + 1
        return True in np.isnan(sample_array[sample_start: sample_start + self.seq_length + 1]) or len(sample_array[sample_start: sample_start + self.seq_length + 1]) != self.seq_length + 1
