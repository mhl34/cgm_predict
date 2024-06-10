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
from sklearn.model_selection import KFold

sns.set_theme()

class Analysis:
    def __init__(self, mainDir):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--modelType", dest="modelType", help="input the type of model you want to use")
        parser.add_argument("-e", "--epochs", default=100, dest="num_epochs", help="input the number of epochs to run")
        parser.add_argument("-n", "--normalize", action='store_true', dest="normalize", help="input whether or not to normalize the input sequence")
        parser.add_argument("-k", "--k_fold", default=-1, dest="kfold", help="input whether or not to run k-fold validation")
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
            return LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc, bidirectional = True)
            # return TransformerModel(num_features = 1024, num_head = 256, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
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
            model = LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype, bidirectional = True)
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

    def enable_dropout(self, model):
        """
        Enable just the dropout for the monte carlo dropout simulation
        """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def monte_carlo_dropout(self, runs = 100):
        """
        Uncertainty estimation using Monte Carlo dropout
        """
        mean_dict = {}
        std_dict = {}
        modeltypes = ['conv1d', 'unet', 'lstm', 'transformer']
        color_means = {'lstm': 'purple', 'transformer': 'r', 'conv1d': 'g', 'unet': 'b'}
        color_stds = {'lstm': 'violet', 'transformer': 'lightcoral', 'conv1d': 'lightgreen', 'unet': 'lightblue'}
        names_dict = {'lstm': 'LSTM', 'unet': 'U-Net', 'conv1d': 'CNN', 'transformer': 'Transformer'}
        for modeltype in modeltypes:
            model = self.modelLoader(modeltype)
            model.eval()
            self.enable_dropout(model)

            dataloader = np.load(self.data_folder + "full_data.npz")['arr']
            data = dataloader[-1]

            preds_list = []
            progress_bar = tqdm(range(runs), total=runs, desc='Monte Carlo Dropout')
            for _ in progress_bar:
                data = torch.Tensor(data).to(self.dtype)
                # stack the inputs and feed as 3 channel input
                data = data.squeeze(2)
                if self.no_gluc:
                    input = data[:, :-2, :].to(self.dtype)
                else:
                    input = data[:, :-1, :].to(self.dtype)
            
                target = data[:, -1, :]

                if modeltype == "conv1d" or modeltype == "lstm" or modeltype == "unet":
                    output = model(input).to(self.dtype).squeeze()
                elif modeltype == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()
                output_arr = ((output * self.train_std) + self.train_mean)
                preds_list.append(output_arr[-1].detach().numpy())
            
            target_arr = ((target[-1] *self.train_std) + self.train_mean).detach().numpy()
            predictions = torch.Tensor(np.array(preds_list))
            mean_prediction = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)

            mean_dict[modeltype] = mean_prediction
            std_dict[modeltype] = uncertainty

            mean_arr = mean_prediction.detach().numpy()
            std_arr = uncertainty.detach().numpy()
            lower_arr = mean_arr - std_arr
            upper_arr = mean_arr + std_arr

            time = np.arange(0, predictions.shape[-1], 1) * 5

            if modeltype != 'transformer':
                plt.plot(time, mean_arr, color = color_means[modeltype], label=names_dict[modeltype], linewidth = 1)
                plt.fill_between(time, lower_arr, upper_arr, color=color_stds[modeltype], alpha=0.25)
                continue
            plt.plot(time, mean_arr, color = color_means[modeltype], label=names_dict[modeltype], linewidth = 2)
            plt.fill_between(time, lower_arr, upper_arr, color=color_stds[modeltype], alpha=0.5)

        plt.plot(time, target_arr, color = 'black', label='Target', linewidth = 2, linestyle = '--')

        # Set labels and title
        plt.xlabel("Interval")
        plt.ylabel("Glucose Value (mg/dL)")
        plt.title(f"Monte Carlo Dropout")
        plt.legend(loc = "upper left", prop={'size': 8})

        # Save the plot
        plt.savefig(f'{self.plots_folder}mcd_plot.png')
        
        return mean_prediction, uncertainty, preds_list
 
    def plot_model(self):
        """
        Model Plotting Method
        """
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
        """
        Plotting the outputs of the models onto a saved graph
        """
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

        color_dict = {'lstm': 'purple', 'transformer': 'r', 'conv1d': 'g', 'unet': 'b', 'target': 'black'}

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
            plt.savefig(f'{self.plots_folder}output_plot_no_gluc.png')
        else:
            plt.savefig(f'{self.plots_folder}output_plot.png')

    def plot_performance(self):
        """
        Plots the saved performances from the training regiment of each of the models
        """
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
            'lstm': 'purple', 
            'transformer': 'r', 
            'conv1d': 'g', 
            'unet': 'b', 
            'target': 'black'
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
        plt.savefig(f'{self.plots_folder}accuracy_plot_no_gluc.png')

        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('white')
        plt.grid(True, color='grey', linestyle='--')
        for key, val in performance_dict.items():
            plt.plot(np.arange(len(val['loss'])), val['loss'], label = key, color = color_dict[key], linewidth = linewidth_dict[key], linestyle = linestyle_dict[key])
            print(key)
            print(val['loss'][-1])
        plt.xlabel("Epochs", fontsize=12, fontweight='bold')
        plt.ylabel("Loss (MSE Loss)", fontsize=12, fontweight='bold')
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        # plt.legend(loc='lower center')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.plots_folder}loss_plot_no_gluc.png')

    def plot_lopocv(self):
        # get lopocv loss
        cnn_acc = np.load(self.performance_folder + "conv1d_lopocv_accs.npz")['arr']
        unet_acc = np.load(self.performance_folder + "unet_lopocv_accs.npz")['arr']
        lstm_acc = np.load(self.performance_folder + "lstm_lopocv_accs.npz")['arr']
        transformer_acc = np.load(self.performance_folder + "transformer_lopocv_accs.npz")['arr']

        # Additional plots for better comparison (e.g., box plot)
        accs_lst = [cnn_acc, unet_acc, lstm_acc, transformer_acc]
        box = plt.boxplot(accs_lst, labels=['CNN', 'UNet', 'LSTM', 'Transformer'], 
                          patch_artist = True)
        # Define colors for each box
        color_stds = {'lstm': 'violet', 'transformer': 'lightcoral', 'conv1d': 'lightgreen', 'unet': 'lightblue'}
        colors = [color_stds['conv1d'], color_stds['unet'], color_stds['lstm'], color_stds['transformer']]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        # Set the median line color to black and make it thicker
        for median in box['medians']:
            median.set_color('black')
            median.set_linewidth(2.5)
        print([np.median(i) for i in accs_lst])
        plt.ylabel('Accuracy (100 - MAPE)')
        plt.title('LOPOCV Accuracies')
        plt.grid(True)
        plt.savefig(self.plots_folder + 'lopocv_box_plot.png')



if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    # mainDir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = Analysis(mainDir)
    obj.plot_performance()
    # obj.monte_carlo_dropout()
