import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import argparse
from tqdm import tqdm
from DataProcessor import DataProcessor
from FeatureDataset import FeatureDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from models.Conv1DModel import Conv1DModel
from models.LstmModel import LstmModel
from models.TransformerModel import TransformerModel
from models.LstmEnhancedModel import LstmEnhancedModel
from models.UNet import UNet
from sklearn.model_selection import KFold

sns.set_theme()

class runModel:
    def __init__(self, mainDir):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--modelType", dest="modelType", help="input the type of model you want to use")
        parser.add_argument("-e", "--epochs", default=100, dest="num_epochs", help="input the number of epochs to run")
        parser.add_argument("-n", "--normalize", action='store_true', dest="normalize", help="input whether or not to normalize the input sequence")
        parser.add_argument("-k", "--k_fold", default=-1, dest="kfold", help="input whether or not to run k-fold validation")
        parser.add_argument("-l", "--lopocv", action='store_true', dest="lopocv", help="input whether or not to perform leave-one-person-out cross validation")
        parser.add_argument("-s", "--seq_len", default=28, dest="seq_len", help="input the sequence length to analyze")
        parser.add_argument("-ng", "--no_glucose", action='store_true', dest="no_gluc", help="input whether or not to remove the glucose sample")
        args = parser.parse_args()
        self.modelType = args.modelType
        self.dtype = torch.double if self.modelType == "conv1d" else torch.float64
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = int(args.seq_len)
        self.num_epochs = int(args.num_epochs)
        self.normalize = args.normalize
        self.no_gluc = args.no_gluc
        self.kfold = int(args.kfold)
        self.lopocv = args.lopocv

        # model parameters
        self.dropout_p = 0.5
        self.domain_lambda = 0.01
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.num_features = 11
        self.num_features = self.num_features - 1 if self.no_gluc else self.num_features
        self.lr = 1e-3
        self.weight_decay = 1e-8

        # normalization
        samples = [str(i).zfill(3) for i in range(1, 17)]
        dataProcessor = DataProcessor(samples, self.mainDir)
        glucoseData = dataProcessor.loadData(samples, "dexcom")
        self.train_mean = glucoseData['mean']
        self.train_std = glucoseData['std']
        self.eps = 1e-12

        # lstm parameters 
        self.hidden_size = 128
        self.num_layers = 4
        self.hidden_size_e = self.seq_length
        # self.hidden_size_e = 8
        self.num_layers_e = 8

        # transformer parameters
        self.dim_model = 1024
        self.num_head = 128

        # early stopping parameter
        self.patience = 10
        self.min_delta = 0.01

        # direction
        self.checkpoint_folder = "saved_models/"
        self.check_dir(self.checkpoint_folder)
        self.data_folder = "data/"
        self.check_dir(self.data_folder)
        self.model_folder = "model_arch/"
        self.check_dir(self.model_folder)
        self.performance_folder = "performance/"
        self.check_dir(self.performance_folder)
        self.plots_folder = "plots/"
        self.check_dir(self.plots_folder)

    def check_dir(self, folder_path):
        """
        Ensures all of the directories are created locally for saving
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Directory '{folder_path}' created.")
        else:
            print(f"Directory '{folder_path}' already exists.")

    def modelChooser(self, modelType):
        """
        Selects the model based on specified args
        """
        if modelType == "conv1d":
            print(f"model {modelType}")
            return Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "lstm":
            print(f"model {modelType}")
            return LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype, bidirectional = True)
        elif modelType == "lstm_e":
            print(f"model {modelType}")
            return LstmEnhancedModel(hidden_size = self.hidden_size_e, num_layers = self.num_layers_e, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc, batch_first = True, bidirectional = True)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc)
        elif modelType == "unet":
            print(f"model {modelType}")
            return UNet(self.num_features, normalize = False, seq_len = self.seq_length, dropout = self.dropout_p)
        else:
            raise Exception("Invalid model type")
        
    def modelLoader(self, modelType):
        """
        Loads models from saved .pth files
        """
        no_gluc_flag = "_no_gluc" if self.no_gluc else ""
        file = f"{modelType}{no_gluc_flag}.pth"
        if modelType == "conv1d":
            print(f"model {modelType}")
            model = Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "lstm":
            print(f"model {modelType}")
            model = LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        elif modelType == "transformer":
            print(f"model {modelType}")
            model = TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc)
        elif modelType == "unet":
            print(f"model {modelType}")
            model = UNet(self.num_features, normalize = False, seq_len = self.seq_length, dropout = self.dropout_p)
        else:
            raise Exception("Invalid model type")
        model.load_state_dict(torch.load(self.checkpoint_folder + file)['state_dict'])
        return model

    def train(self, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion):
        """
        Runs the training regiment, returns the outputs of loss and accuracy for plotting if it is not running K-Fold Cross Validation
        """
        model.train()
        best_loss = float('inf')
        early_stopping_counter = 0
        global_loss_lst = []
        global_acc_lst = []

        for epoch in range(self.num_epochs):
            model.train()
            np.random.shuffle(train_dataloader)
            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')

            lossLst = []
            accLst = []

            for _, data in progress_bar:
                data = torch.Tensor(data).to(self.dtype).to(self.device)
                data = data.squeeze(2)
                if self.no_gluc:
                    input = data[:, :-2, :].to(self.dtype)
                else:
                    input = data[:, :-1, :].to(self.dtype)

                target = data[:, -1, :]

                if self.modelType in ["conv1d", "lstm", "lstm_e", "unet"]:
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                lossLst.append(loss.item())

                output_arr = ((output * self.train_std) + self.train_mean)
                target_arr = ((target * self.train_std) + self.train_mean)

                acc_val = 1 - self.mape(output_arr, target_arr)
                accLst.append(acc_val)

            scheduler.step()

            avg_loss = np.mean(lossLst)
            avg_acc = np.mean(accLst)

            global_loss_lst.append(avg_loss)
            global_acc_lst.append(avg_acc)

            print(f"epoch {epoch + 1} training loss: {avg_loss} learning rate: {scheduler.get_last_lr()} training accuracy: {avg_acc}")

            val_loss, _ = self.evaluate(model, val_dataloader, criterion)

            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
                if not os.path.exists(self.checkpoint_folder):
                    os.makedirs(self.checkpoint_folder)
                print("Saving ...")
                state = {'state_dict': model.state_dict(), 'epoch': epoch, 'lr': self.lr}
                torch.save(state, os.path.join(self.checkpoint_folder, f'{self.modelType}.pth' if not self.no_gluc else f'{self.modelType}_no_gluc.pth'))
            elif val_loss > best_loss + self.min_delta:
                early_stopping_counter += 1

            if early_stopping_counter >= self.patience:
                print("Early stopping triggered.")
                break

        if self.kfold == -1 and not self.lopocv:
            file_path_loss = f"{self.performance_folder}{self.modelType}_loss"
            file_path_acc = f"{self.performance_folder}{self.modelType}_acc"
            np.savez(file_path_acc, arr=np.array(global_acc_lst))
            np.savez(file_path_loss, arr=np.array(global_loss_lst))

        
    def evaluate(self, model, val_dataloader, criterion):
        """
        Model evaluation method, plots the outputs if it's not performing K-Fold, returns the average loss and accuracy (100 - MAPE)
        """
        model.eval()
        with torch.no_grad():
            epoch = 1
            lossLst = []
            accLst = []

            for _, data in enumerate(val_dataloader):
                data = torch.Tensor(data).to(self.dtype).to(self.device)
                # stack the inputs and feed as 3 channel input
                data = data.squeeze(2)
                if self.no_gluc:
                    input = data[:, :-2, :].to(self.dtype)
                else:
                    input = data[:, :-1, :].to(self.dtype)

                target = data[:, -1, :]
                
                if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "lstm_e" or self.modelType == "unet":
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()
                
                # loss is only calculated from the main task
                loss = criterion(output, target)

                lossLst.append(loss.item())
                output_arr = ((output * self.train_std) + self.train_mean)
                target_arr = ((target * self.train_std) + self.train_mean)
                accLst.append(1 - self.mape(output_arr, target_arr))

            print(f"val loss: {np.mean(lossLst)} val accuracy: {np.mean(accLst)}")

            if self.kfold == -1 and not self.lopocv:
                # create example output plot with the epoch
                plt.clf()
                plt.grid(True)
                plt.figure(figsize=(8, 6))

                # Plot the target array
                plt.plot(target_arr.cpu().detach().numpy()[-1], label='Target')

                # Plot the output arrays (first and second arrays in the tuple)
                plt.plot(output_arr.cpu().detach().numpy()[-1], label='Output')

                # Add labels and legend
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title(f'Target vs. Output (model: {self.modelType})', fontdict={'fontsize': 16, 'fontweight': 'bold'})
                plt.legend()

                # Save the plot as a PNG file
                if self.no_gluc:
                    plt.savefig(f'{self.plots_folder}{self.modelType}_output_no_gluc.png')
                else:
                    plt.savefig(f'{self.plots_folder}{self.modelType}_output.png')

                plt.close()
        return np.mean(lossLst), np.mean(accLst)           

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()

    def run(self):
        """
        Chooses which one to run based on flags
        """
        if self.kfold == -1 and not self.lopocv:
            self.run_train_test()
        elif self.kfold == -1 and self.lopocv:
            self.run_lopocv()
        else:
            self.run_k_fold(self.kfold)

    def run_lopocv(self):
        """
        Runs Leave-One-Person-Out Cross Validation
        """
        samples = [str(i).zfill(3) for i in range(1, 17)]

        fold_losses = []
        fold_accs = []

        for lopo_sample in samples:
            print(f"getting data for sample {lopo_sample}")
            self.getData(self.data_folder, [sample for sample in samples if sample != lopo_sample], f"{lopo_sample}_left_out_data.npz")
            self.getData(self.data_folder, [lopo_sample], f"{lopo_sample}_data.npz")
        for sample in samples:
            print(f"Sample {sample}")
            model = self.modelChooser(self.modelType).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
            criterion = nn.MSELoss()

            train_dataloader = np.load(self.data_folder + f"{lopo_sample}_left_out_data.npz")['arr']
            val_dataloader = np.load(self.data_folder + f"{lopo_sample}_data.npz")['arr']

            # Train the model
            self.train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion)
            
            print("============================")
            print("Evaluating ...")
            print("============================")

            # Evaluate the model
            val_loss, val_acc = self.evaluate(model, val_dataloader, criterion)
            print(f'Sample {sample} - Validation Loss: {val_loss} - Validation Accuracy: {val_acc}')

            fold_losses.append(val_loss)
            fold_accs.append(val_acc)

        loss_arr = np.array(fold_losses)
        accs_arr = np.array(fold_accs)

        np.savez(f"{self.performance_folder}{self.modelType}_lopocv_loss.npz", arr = loss_arr)
        np.savez(f"{self.performance_folder}{self.modelType}_lopocv_accs.npz", arr = accs_arr)

        print(f'Average Validation Loss: {np.mean(fold_losses)} - Average Validation Accuracy: {np.mean(fold_accs)}')

    def run_k_fold(self, folds):
        """
        Runs K-Fold Cross Validation
        """
        samples = [str(i).zfill(3) for i in range(1, 17)]

        self.getData(self.data_folder, samples, "full_data.npz")
        full_dataloader = np.load(self.data_folder + "full_data.npz")['arr']

        k = folds # Number of folds
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        fold = 0
        fold_losses = []
        fold_accs = []

        for train_index, val_index in kf.split(full_dataloader):
            fold += 1
            print(f'Fold {fold}')

            train_dataloader, val_dataloader = full_dataloader[train_index], full_dataloader[val_index]
            model = self.modelChooser(self.modelType).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
            criterion = nn.MSELoss()

            # Train the model
            self.train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion)
            
            print("============================")
            print("Evaluating Fold...")
            print("============================")

            # Evaluate the model
            val_loss, val_acc = self.evaluate(model, val_dataloader, criterion)
            print(f'Fold {fold} - Validation Loss: {val_loss} - Validation Accuracy: {val_acc}')

            fold_losses.append(val_loss)
            fold_accs.append(val_acc)
        
        loss_arr = np.array(fold_losses)
        accs_arr = np.array(fold_accs)

        np.savez(f"{self.performance_folder}{self.modelType}_k_fold_loss.npz", arr = loss_arr)
        np.savez(f"{self.performance_folder}{self.modelType}_k_fold_accs.npz", arr = accs_arr)

        print(f'Average Validation Loss: {np.mean(fold_losses)} - Average Validation Accuracy: {np.mean(fold_accs)}')

    def run_train_test(self):
        """
        Runs with a simple train test split of 75-25
        """
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-4]
        valSamples = samples[-4:]

        model = self.modelChooser(self.modelType).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        criterion = nn.MSELoss()

        self.getData(self.data_folder, trainSamples, "train_data_aligned.npz")
        train_dataloader = np.load(self.data_folder + "train_data_aligned.npz")['arr']
        self.getData(self.data_folder, valSamples, "val_data_aligned.npz")
        val_dataloader = np.load(self.data_folder + "val_data_aligned.npz")['arr']
        
        self.train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion)

        print("============================")
        print("Evaluating...")
        print("============================")

        _, _ = self.evaluate(model, val_dataloader, criterion)
    
    def getData(self, save_dir, samples, file_name):
        """
        Gets the full data to perform K-Fold Cross Validation or Train-Test and stores it in a .npz file
        """
        data_list = []
        file_path = save_dir + file_name

        if os.path.exists(file_path):
            print(f"{file_name} Found!")
            return
        else:
            print(f"Creating {file_name} File")

        # load in classes
        dataProcessor = DataProcessor(samples, mainDir = self.mainDir)

        foodData = dataProcessor.loadData(samples, "food")
        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")
        accData = dataProcessor.loadData(samples, "acc")
        hba1c = dataProcessor.hba1c(samples)
        minData = dataProcessor.minFromMidnight(samples)

       # sugarTensor, carbTensor, minTensor, hba1cTensor, edaTensor, hrTensor, tempTensor, accTensor, glucPastTensor, glucTensor
        # means
        sugar_mean = foodData['mean_sugar']
        carb_mean = foodData['mean_carb']
        min_mean = minData['mean']
        hba1c_mean = hba1c['mean']
        eda_mean = edaData['mean']
        hr_mean = hrData['mean']
        temp_mean = tempData['mean']
        acc_x_mean = accData['mean_x']
        acc_y_mean = accData['mean_y']
        acc_z_mean = accData['mean_z']
        gluc_mean = glucoseData['mean']

        # stds
        sugar_std = foodData['std_sugar']
        carb_std = foodData['std_carb']
        min_std = minData['std']
        hba1c_std = hba1c['std']
        eda_std = edaData['std']
        hr_std = hrData['std']
        temp_std = tempData['std']
        acc_x_std = accData['std_x']
        acc_y_std = accData['std_y']
        acc_z_std = accData['std_z']
        gluc_std = glucoseData['std']

        mean_list = [sugar_mean, carb_mean, min_mean, hba1c_mean, eda_mean, hr_mean, temp_mean, acc_x_mean, acc_y_mean, acc_z_mean, gluc_mean, gluc_mean]
        std_list = [sugar_std, carb_std, min_std, hba1c_std, eda_std, hr_std, temp_std, acc_x_std, acc_y_std, acc_z_std, gluc_std, gluc_std]
        std_list = [std + self.eps if std > self.eps else 1 for std in std_list]
        # Step 2: Define a custom transform to normalize the data
        custom_transform = transforms.Compose([
            # transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean = mean_list, std = std_list)  # Normalize using mean and std
        ])

        dataset = FeatureDataset(samples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, dtype = self.dtype, seq_length = self.seq_length, transforms = custom_transform)
        # returns eda, hr, temp, then hba1c
        dataloader = DataLoader(dataset, batch_size = self.val_batch_size, shuffle = False)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), unit='batch')

        for _, data in progress_bar:
            # stack the inputs and feed as 3 channel input
            if data.shape[0] < self.train_batch_size:
                continue
            data_list.append(data.cpu().detach().numpy())

        data_np_arr = np.array(data_list)

        np.savez(file_path, arr = data_np_arr)

        print(f"Array saved to {file_path} successfully.")

if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    # mainDir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    # obj.run_train_test()
    obj.run()
