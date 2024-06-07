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
from models.UNet import UNet

sns.set_theme()

class runModel:
    def __init__(self, mainDir):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--modelType", dest="modelType", help="input the type of model you want to use")
        parser.add_argument("-gm", "--glucMetric", default = "mean", dest="glucMetric", help="input the type of glucose metric you want to regress for")
        parser.add_argument("-e", "--epochs", default=100, dest="num_epochs", help="input the number of epochs to run")
        parser.add_argument("-n", "--normalize", action='store_true', dest="normalize", help="input whether or not to normalize the input sequence")
        parser.add_argument("-s", "--seq_len", default=28, dest="seq_len", help="input the sequence length to analyze")
        parser.add_argument("-ng", "--no_glucose", action='store_true', dest="no_gluc", help="input whether or not to remove the glucose sample")
        args = parser.parse_args()
        self.modelType = args.modelType
        self.glucMetric = args.glucMetric
        self.dtype = torch.double if self.modelType == "conv1d" else torch.float64
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = int(args.seq_len)
        self.num_epochs = int(args.num_epochs)
        self.normalize = args.normalize
        self.no_gluc = args.no_gluc

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
        print(self.train_mean, self.train_std)
        self.eps = 1e-12

        # lstm parameters 
        self.hidden_size = 32
        self.num_layers = 2

        # transformer parameters
        self.dim_model = 1024
        self.num_head = 128

        # direction
        self.checkpoint_folder = "saved_models/"
        self.check_dir(self.checkpoint_folder)
        self.data_folder = "data/"
        self.check_dir(self.data_folder)
        self.model_folder = "model_arch/"
        self.check_dir(self.model_folder)
        self.performance_folder = "performance/"
        self.check_dir(self.performance_folder)

    def check_dir(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Directory '{folder_path}' created.")
        else:
            print(f"Directory '{folder_path}' already exists.")

    def modelChooser(self, modelType, samples):
        if modelType == "conv1d":
            print(f"model {modelType}")
            return Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "lstm":
            print(f"model {modelType}")
            return LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc)
            # return TransformerModel(num_features = 1024, num_head = 256, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
        elif modelType == "unet":
            print(f"model {modelType}")
            return UNet(self.num_features, normalize = False, seq_len = self.seq_length)
        else:
            raise Exception("Invalid model type")


    def train(self, model, train_dataloader, optimizer, scheduler, criterion):
        best_acc = -float('inf')
        global_loss_lst = []
        global_acc_lst = []
        for epoch in range(self.num_epochs):

            np.random.shuffle(train_dataloader)

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')

            lossLst = []
            accLst = []
            
            # for batch_idx, (sample, acc, sugar, carb, mins, hba1c, glucPast, glucPres) in progress_bar:
            for _, data in progress_bar:
                data = torch.Tensor(data).to(self.dtype)
                # stack the inputs and feed as 3 channel input
                data = data.squeeze(2)
                if self.no_gluc:
                    input = data[:, :-2, :].to(self.dtype)
                else:
                    input = data[:, :-1, :].to(self.dtype)

                target = data[:, -1, :]

                if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
                optimizer.step()

                lossLst.append(loss.item())

                output_arr = ((output * self.train_std) + self.train_mean)
                target_arr = ((target * self.train_std) + self.train_mean)

                acc_val = 1 - self.mape(output_arr, target_arr)
                
                accLst.append(acc_val)
            scheduler.step()

            for outVal, targetVal in zip(output_arr.detach().numpy()[-1][:-3], target_arr.detach().numpy()[-1][:-3]):
                print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {(outVal.item() - targetVal.item())}")
       
            avg_loss = sum(lossLst)/len(lossLst)
            avg_acc  = sum(accLst)/len(accLst)

            global_loss_lst.append(avg_loss)
            global_acc_lst.append(avg_acc)

            print(f"epoch {epoch + 1} training loss: {avg_loss} learning rate: {scheduler.get_last_lr()} training accuracy: {avg_acc}")

            if avg_acc > best_acc:
                    best_acc = acc_val
                    if not os.path.exists(self.checkpoint_folder):
                        os.makedirs(self.checkpoint_folder)
                    print("Saving ...")
                    state = {'state_dict': model.state_dict(),
                            'epoch': epoch,
                            'lr': self.lr}
                    torch.save(state, os.path.join(self.checkpoint_folder, f'{self.modelType}.pth' if not self.no_gluc else f'{self.modelType}_no_gluc.pth'))
        
        file_path_loss = f"{self.performance_folder}{self.modelType}_loss"
        file_path_acc = f"{self.performance_folder}{self.modelType}_acc"
        np.savez(file_path_acc, arr = np.array(global_acc_lst))
        np.savez(file_path_loss, arr = np.array(global_loss_lst))

            

    def evaluate(self, model, val_dataloader, criterion):
        with torch.no_grad():
            epoch = 1
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
                
            lossLst = []
            accLst = []

            for _, data in progress_bar:
                data = torch.Tensor(data).to(self.dtype)
                # stack the inputs and feed as 3 channel input
                data = data.squeeze(2)
                if self.no_gluc:
                    input = data[:, :-2, :].to(self.dtype)
                else:
                    input = data[:, :-1, :].to(self.dtype)

                target = data[:, -1, :]
                
                if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()
                
                # loss is only calculated from the main task
                loss = criterion(output, target)

                lossLst.append(loss.item())
                output_arr = ((output * self.train_std) + self.train_mean)
                target_arr = ((target * self.train_std) + self.train_mean)
                accLst.append(1 - self.mape(output_arr, target_arr))

            print(f"val loss: {sum(lossLst)/len(lossLst)} val accuracy: {sum(accLst)/len(accLst)}")

            # create example output plot with the epoch
            plt.clf()
            plt.grid(True)
            plt.figure(figsize=(8, 6))

            # Plot the target array
            plt.plot(target_arr.detach().numpy()[-1], label='Target')

            # Plot the output arrays (first and second arrays in the tuple)
            plt.plot(output_arr.detach().numpy()[-1], label='Output')

            # Add labels and legend
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Target vs. Output (model: {self.modelType})', fontdict={'fontsize': 16, 'fontweight': 'bold'})
            plt.legend()

            # Save the plot as a PNG file
            if self.no_gluc:
                plt.savefig(f'plots/{self.modelType}_output_no_gluc.png')
            else:
                plt.savefig(f'plots/{self.modelType}_output.png')            

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()

    def run(self):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-5]
        valSamples = samples[-5:]

        model = self.modelChooser(self.modelType, samples)
        self.getTrainData(self.data_folder)

        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        criterion = nn.MSELoss()

        train_dataloader = np.load(self.data_folder + "train_data.npz")['arr']
        
        self.train(model, train_dataloader, optimizer, scheduler, criterion)

        print("============================")
        print("Evaluating...")
        print("============================")
        model.eval()

        # drop auxiliary networks
        if self.modelType == "ssl":
            model.decoder = nn.Identity()
        if self.modelType == "dann":
            model.adversary = nn.Identity()

        self.getEvalData(self.data_folder)

        val_dataloader = np.load(self.data_folder + "val_data.npz")['arr']

        self.evaluate(model, val_dataloader, criterion)

    def getTrainData(self, save_dir):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-5]

        file_name = "train_data.npz"
        file_path = save_dir + file_name

        if os.path.exists(file_path):
            print("Train Data Found!")
            return
        else:
            print("Creating Train Data File")

        # load in classes
        dataProcessor = DataProcessor(trainSamples, mainDir = self.mainDir)

        foodData = dataProcessor.loadData(trainSamples, "food")
        glucoseData = dataProcessor.loadData(trainSamples, "dexcom")
        edaData = dataProcessor.loadData(trainSamples, "eda")
        tempData = dataProcessor.loadData(trainSamples, "temp")
        hrData = dataProcessor.loadData(trainSamples, "hr")
        accData = dataProcessor.loadData(trainSamples, "acc")
        hba1c = dataProcessor.hba1c(trainSamples)
        minData = dataProcessor.minFromMidnight(trainSamples)

        self.train_mean = glucoseData['mean']
        self.train_std = glucoseData['std']

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
        print(mean_list)
        print(std_list)
        std_list = [std + self.eps if std > self.eps else 1 for std in std_list]

        # Step 2: Define a custom transform to normalize the data
        custom_transform = transforms.Compose([
            # transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean = mean_list, std = std_list)  # Normalize using mean and std
        ])

        train_dataset = FeatureDataset(trainSamples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, transforms = custom_transform)
        # returns eda, hr, temp, then hba1c
        train_dataloader = DataLoader(train_dataset, batch_size = self.train_batch_size, shuffle = True)

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch')

        # where to store the data
        data_list = []

        # for batch_idx, (sample, acc, sugar, carb, mins, hba1c, glucPast, glucPres) in progress_bar:
        for _, data in progress_bar:
            # stack the inputs and feed as 3 channel input
            if data.shape[0] < self.train_batch_size:
                continue
            data_list.append(data.detach().numpy())

        data_np_arr = np.array(data_list)

        np.savez(file_path, arr = data_np_arr)

        print(f"Array saved to {file_path} successfully.")

    def getEvalData(self, save_dir):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        valSamples = samples[-5:]

        data_list = []

        file_name = "val_data.npz"
        file_path = save_dir + file_name

        if os.path.exists(file_path):
            print("Val Data Found!")
            return
        else:
            print("Creating Val Data File")

        # load in classes
        dataProcessor = DataProcessor(valSamples, mainDir = self.mainDir)

        foodData = dataProcessor.loadData(valSamples, "food")
        glucoseData = dataProcessor.loadData(valSamples, "dexcom")
        edaData = dataProcessor.loadData(valSamples, "eda")
        tempData = dataProcessor.loadData(valSamples, "temp")
        hrData = dataProcessor.loadData(valSamples, "hr")
        accData = dataProcessor.loadData(valSamples, "acc")
        hba1c = dataProcessor.hba1c(valSamples)
        minData = dataProcessor.minFromMidnight(valSamples)

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

        val_dataset = FeatureDataset(valSamples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, transforms = custom_transform)
        # returns eda, hr, temp, then hba1c
        val_dataloader = DataLoader(val_dataset, batch_size = self.val_batch_size, shuffle = False)

        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), unit='batch')

        for _, data in progress_bar:
            # stack the inputs and feed as 3 channel input
            if data.shape[0] < self.train_batch_size:
                continue
            data_list.append(data.detach().numpy())

        data_np_arr = np.array(data_list)

        np.savez(file_path, arr = data_np_arr)

        print(f"Array saved to {file_path} successfully.")

    def plot_model(self):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        model = self.modelChooser(self.modelType, samples)

        input = torch.randn(self.train_batch_size, self.num_features, self.seq_length).to(self.dtype)
        target = torch.randn(self.train_batch_size, 1, self.seq_length).to(self.dtype)

        if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
            output = model(input).to(self.dtype).squeeze()
        elif self.modelType == "transformer":
            output = model(target, input).to(self.dtype).squeeze()
        
        # graph = make_dot(output, params = dict(model.named_parameters()))

        # graph.render(filename=f'{self.model_folder}{self.modelType}', format='png')

        # keras_model = visualkeras.utils.convert_to_visualkeras(model, input)

        # # Plot the architecture
        # visualkeras.layered_view(keras_model, scale_xy=1.5, scale_z=1, max_z=10)

        # # Save the plot as a PNG file
        # visualkeras.save(keras_model, f'{self.model_folder}{self.modelType}')

    def plot_output(self):
        # to get normalization calculations
        self.getTrainData(self.data_folder)
        
        samples = [str(i).zfill(3) for i in range(1, 17)]
        conv1d = Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        lstm = LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        transformer = TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc)
        unet = UNet(self.num_features, normalize = False, seq_len = self.seq_length)

        if self.no_gluc:
            conv1d.load_state_dict(torch.load(self.checkpoint_folder + "conv1d_no_gluc.pth")['state_dict'])
            lstm.load_state_dict(torch.load(self.checkpoint_folder + "lstm_no_gluc.pth")['state_dict'])
            transformer.load_state_dict(torch.load(self.checkpoint_folder + "transformer_no_gluc.pth")['state_dict'])
            unet.load_state_dict(torch.load(self.checkpoint_folder + "unet_no_gluc.pth")['state_dict'])
        else:
            conv1d.load_state_dict(torch.load(self.checkpoint_folder + "conv1d.pth")['state_dict'])
            lstm.load_state_dict(torch.load(self.checkpoint_folder + "lstm.pth")['state_dict'])
            transformer.load_state_dict(torch.load(self.checkpoint_folder + "transformer.pth")['state_dict'])
            unet.load_state_dict(torch.load(self.checkpoint_folder + "unet.pth")['state_dict'])

        model_dict = {'conv1d': conv1d, 'lstm': lstm, 'transformer': transformer, 'unet': unet}
        output_dict = {'conv1d': None, 'lstm': None, 'transformer': None, 'unet': None, 'target': None}

        val_data = np.load(self.data_folder + "val_data.npz")['arr'][-1]

        for model_name in model_dict.keys():
            model = model_dict[model_name]
            data = torch.Tensor(val_data).to(self.dtype)
            # stack the inputs and feed as 3 channel input
            data = data.squeeze(2)
            if self.no_gluc:
                input = data[:, :-2, :].to(self.dtype)
            else:
                input = data[:, :-1, :].to(self.dtype)

            target = data[:, -1, :]

            if output_dict['target'] == None:
                target_arr = ((target * self.train_std) + self.train_mean)
                output_dict['target'] = target_arr

            if model_name == "conv1d" or model_name == "lstm" or model_name == "unet":
                output = model(input).to(self.dtype).squeeze()
            elif model_name == "transformer":
                output = model(target, input).to(self.dtype).squeeze()

            output_arr = ((output * self.train_std) + self.train_mean)

            output_dict[model_name] = output_arr
        
        minutes = np.arange(0, 28 * 5, 5)

        plt.clf()
        plt.figure(figsize = (8,6))
        plt.gca().set_facecolor('white')
        plt.grid(True, color='grey', linestyle='--')

        color_dict = {
            "conv1d": "red",
            "unet": "green",
            "lstm": "blue", 
            "transformer": "orange",
            "target": "purple"
        }

        linestyle_dict = {
            "conv1d": "-",
            "unet": "-",
            "lstm": "-", 
            "transformer": "--",
            "target": "--"
        }

        linewidth_dict = {
            "conv1d": 1,
            "unet": 1,
            "lstm": 1, 
            "transformer": 4,
            "target": 4
        }

        for key, value in output_dict.items():
            plt.plot(minutes, value.detach().numpy()[-1], label = key, color=color_dict[key], linewidth = linewidth_dict[key], linestyle = linestyle_dict[key])

        plt.xlabel('Time (Minutes)', fontsize=12, fontweight='bold')
        plt.ylabel('Glucose Value (mg/dL)', fontsize=12, fontweight='bold')
        plt.title('Model Outputs', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        if self.no_gluc:
            plt.savefig('plots/output_plot_no_gluc.png')
        else:
            plt.savefig('plots/output_plot.png')

    def plot_performance(self):
        # load in accuracy arrays
        conv1d_acc = np.load(self.performance_folder + "conv1d_acc.npz")['arr']
        unet_acc = np.load(self.performance_folder + "unet_acc.npz")['arr']
        lstm_acc = np.load(self.performance_folder + "lstm_acc.npz")['arr']
        transformer_acc = np.load(self.performance_folder + "transformer_acc.npz")['arr']
        
        conv1d_loss = np.load(self.performance_folder + "conv1d_loss.npz")['arr']
        unet_loss = np.load(self.performance_folder + "unet_loss.npz")['arr']
        lstm_loss = np.load(self.performance_folder + "lstm_loss.npz")['arr']
        transformer_loss = np.load(self.performance_folder + "transformer_loss.npz")['arr']

        performance_dict = {
            "conv1d": {
                "acc": conv1d_acc,
                "loss": conv1d_loss
            },
            "unet": {
                "acc": unet_acc,
                "loss": unet_loss
            },
            "lstm": {
                "acc": lstm_acc,
                "loss": lstm_loss
            },
            "transformer": {
                "acc": transformer_acc,
                "loss": transformer_loss
            }
        }

        color_dict = {
            "conv1d": "red",
            "unet": "green",
            "lstm": "blue", 
            "transformer": "orange",
            "target": "purple"
        }

        linestyle_dict = {
            "conv1d": "-",
            "unet": "-",
            "lstm": "-", 
            "transformer": "--",
            "target": "--"
        }

        linewidth_dict = {
            "conv1d": 1,
            "unet": 1,
            "lstm": 1, 
            "transformer": 4,
            "target": 4
        }

        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('white')
        plt.grid(True, color='grey', linestyle='--')
        for key, val in performance_dict.items():
            plt.plot(np.arange(len(val['acc'])), val['acc'] * 100, label = key, color = color_dict[key], linewidth = linewidth_dict[key], linestyle = linestyle_dict[key])
            print(key)
            print(val['acc'][-1] * 100)
        plt.xlabel("Epochs", fontsize=12, fontweight='bold')
        plt.ylabel("Accuracy (100 - MAPE)", fontsize=12, fontweight='bold')
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.legend(loc='lower center')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/accuracy_plot_no_gluc.png')

        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('white')
        plt.grid(True, color='grey', linestyle='--')
        for key, val in performance_dict.items():
            plt.plot(np.arange(len(val['loss'])), val['loss'], label = key, color = color_dict[key], linewidth = linewidth_dict[key], linestyle = linestyle_dict[key])
            print(val['loss'][-1])
        plt.xlabel("Epochs", fontsize=12, fontweight='bold')
        plt.ylabel("Loss (MSE Loss)", fontsize=12, fontweight='bold')
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        # plt.legend(loc='lower center')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/loss_plot_no_gluc.png')

if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    # mainDir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()
