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
from run import runModel

sns.set_theme()

class Analysis(runModel):
    def modelChooser(self, modelType):
        """
        Selects the model based on specified args
        """
        if modelType == "conv1d":
            print(f"model {modelType}")
            return Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length).to(self.device)
        elif modelType == "lstm":
            print(f"model {modelType}")
            return LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype).to(self.device)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc, bidirectional = self.bidirectional).to(self.device)
            # return TransformerModel(num_features = 1024, num_head = 256, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
        elif modelType == "unet":
            print(f"model {modelType}")
            return UNet(self.num_features, normalize = False, seq_len = self.seq_length, dropout = self.dropout_p).to(self.device)
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
            model = LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype, bidirectional = self.bidirectional)
        elif modelType == "transformer":
            print(f"model {modelType}")
            model = TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc)
        elif modelType == "unet":
            print(f"model {modelType}")
            model = UNet(self.num_features, normalize = False, seq_len = self.seq_length, dropout = self.dropout_p)
        else:
            raise Exception("Invalid model type")
        model.load_state_dict(torch.load(self.checkpoint_folder + file)['state_dict'])
        return model.to(self.device)

    def enable_dropout(self, model):
        """
        Enable just the dropout for the monte carlo dropout simulation
        """
        for m in model.modules():
            name = m.__class__.__name__
            if name.startswith('Dropout'):
                m.train()

    def monte_carlo_dropout(self, runs = 100):
        """
        Uncertainty estimation using Monte Carlo dropout
        """
        with torch.no_grad():
            mean_dict = {}
            std_dict = {}
            modeltypes = ['conv1d', 'unet', 'lstm', 'transformer']
            color_means = {'lstm': 'purple', 'transformer': 'r', 'conv1d': 'g', 'unet': 'b'}
            color_stds = {'lstm': 'violet', 'transformer': 'lightcoral', 'conv1d': 'lightgreen', 'unet': 'lightblue'}
            names_dict = {'lstm': 'LSTM', 'unet': 'U-Net', 'conv1d': 'CNN', 'transformer': 'Transformer'}
            for modeltype in modeltypes:
                model = self.modelLoader(modeltype)
                model.train()
                # self.enable_dropout(model)

                dataloader = np.load(self.data_folder + "val_data_aligned.npz")['arr']
                data = dataloader[-1]

                preds_list = []
                progress_bar = tqdm(range(runs), total=runs, desc='Monte Carlo Dropout')
                for _ in progress_bar:
                    data = torch.Tensor(data).to(self.dtype).to(self.device)
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
                    preds_list.append(output_arr[-1].cpu().detach().numpy())
                
                target_arr = ((target[-1] *self.train_std) + self.train_mean).cpu().detach().numpy()
                predictions = torch.Tensor(np.array(preds_list))
                mean_prediction = predictions.mean(dim=0)
                uncertainty = predictions.std(dim=0)

                mean_dict[modeltype] = mean_prediction
                std_dict[modeltype] = uncertainty

                mean_arr = mean_prediction.cpu().detach().numpy()
                std_arr = uncertainty.cpu().detach().numpy()
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
            plt.xlabel("Interval", fontsize=12, fontweight='bold')
            plt.ylabel("Glucose Value (mg/dL)", fontsize=12, fontweight='bold')
            plt.title(f"Monte Carlo Dropout", fontsize=14, fontweight='bold')

            plt.gca().set_facecolor('white')
            plt.grid(True, color='grey', linestyle='--')
            plt.legend(loc = "upper left", prop={'size': 12})

            # Save the plot
            plt.savefig(f'{self.plots_folder}mcd_plot.png')
            
            return mean_prediction, uncertainty, preds_list

    def feature_ablation(self, feature_lst, num_trials = 100):
        """
        Run Feature Ablation
        """
        with torch.no_grad():
            model = self.modelLoader(self.modelType)
            criterion = nn.MSELoss()
            val_dataloader = np.load(self.data_folder + "val_data_aligned.npz")['arr']

            val_losses = []
            val_accs = []

            model.eval()

            losses_dict = {feat: 0 for feat in feature_lst}
            accs_dict = {feat: 0 for feat in feature_lst}
            for trial in range(num_trials):
                print("==================================")
                print(f"trial {trial + 1}")
                print("==================================")
                for feat_idx in range(self.num_features):
                    feature = feature_lst[feat_idx]
                    print(f"feature {feature}")
                    indices = torch.randperm(self.seq_length)
                    
                    with torch.no_grad():
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

                            # indices = torch.randperm(input.size(2))

                            mu = 0 
                            sigma = 1
                            size = 1
                            rand_num = np.random.normal(mu, sigma, size)[0]
                            input[:, feat_idx, :] = torch.ones_like(input[:, feat_idx, :]).to(self.dtype).to(self.device) * rand_num
                            # input = torch.zeros_like(input).to(self.dtype).to(self.device)

                            target = data[:, -1, :]
                            # tgt = torch.zeros_like(data[:, -1, :]).to(self.dtype).to(self.device)
                            
                            if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                                output = model(input).to(self.dtype).squeeze()
                            elif self.modelType == "transformer":
                                sos_token = torch.ones(self.train_batch_size, 1).to(self.dtype).to(self.device) * self.sos_token
                                target = torch.cat((sos_token, target[:, :-1]), dim=1)
                                output = model(target, input).to(self.dtype).squeeze()
                            
                            # # loss is only calculated from the main task
                            loss = criterion(output, target)

                            lossLst.append(loss.item())
                            output_arr = ((output * self.train_std) + self.train_mean)
                            target_arr = ((target * self.train_std) + self.train_mean)
                            accLst.append(1 - self.mape(output_arr, target_arr))

                    # print(f"val loss: {np.mean(lossLst)} val accuracy: {np.mean(accLst)}")
                    losses_dict[feature] += np.mean(lossLst)
                    accs_dict[feature] += np.mean(accLst)
                # Create a list of tuples (value, original_index)
                indexed_losses = feature_lst.copy()
                indexed_accs = feature_lst.copy()

                # Sort the list of tuples by the values
                indexed_losses.sort(key=lambda x: losses_dict[x])
                indexed_accs.sort(key=lambda x: accs_dict[x], reverse=True)

            print("Losses Ranking")
            for key in losses_dict.keys():
                print(f"{key}: {losses_dict[key]}")
            print("Accuracy Ranking")
            for key in accs_dict.keys():
                print(f"{key}: {accs_dict[key]}")

            print("Ranked by Losses (Most Important to Least):", [feature for feature in indexed_losses])
            print("Ranked by Accuracy (Most Important to Least):", [feature for feature in indexed_accs])

    def plot_model(self):
        """
        Model Plotting Method
        """
        samples = [str(i).zfill(3) for i in range(1, 17)]
        model = self.modelChooser(self.modelType, samples)

        input = torch.randn(self.train_batch_size, self.num_features, self.seq_length).to(self.dtype).to(self.device)
        target = torch.randn(self.train_batch_size, 1, self.seq_length).to(self.dtype).to(self.device)

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
        # self.getTrainData(self.data_folder)
        
        samples = [str(i).zfill(3) for i in range(1, 17)]
        conv1d = Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        lstm = LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        # lstm = LstmEnhancedModel(hidden_size = self.hidden_size_e, num_layers = self.num_layers_e, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc, batch_first = True, bidirectional = self.bidirectional)
        transformer = TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype, num_seqs = self.num_features, no_gluc = self.no_gluc)
        unet = UNet(self.num_features, normalize = False, seq_len = self.seq_length)

        if self.no_gluc:
            conv1d.load_state_dict(torch.load(self.checkpoint_folder + "conv1d_no_gluc.pth")['state_dict'])
            conv1d = conv1d.to(self.device)
            lstm.load_state_dict(torch.load(self.checkpoint_folder + "lstm_no_gluc.pth")['state_dict'])
            lstm = lstm.to(self.device)
            transformer.load_state_dict(torch.load(self.checkpoint_folder + "transformer_no_gluc.pth")['state_dict'])
            transformer = transformer.to(self.device)
            unet.load_state_dict(torch.load(self.checkpoint_folder + "unet_no_gluc.pth")['state_dict'])
            unet = unet.to(self.device)
        else:
            conv1d.load_state_dict(torch.load(self.checkpoint_folder + "conv1d.pth")['state_dict'])
            lstm.load_state_dict(torch.load(self.checkpoint_folder + "lstm.pth")['state_dict'])
            transformer.load_state_dict(torch.load(self.checkpoint_folder + "transformer.pth")['state_dict'])
            unet.load_state_dict(torch.load(self.checkpoint_folder + "unet.pth")['state_dict'])

        model_dict = {'conv1d': conv1d, 'lstm': lstm, 'transformer': transformer, 'unet': unet}
        output_dict = {'conv1d': None, 'lstm': None, 'transformer': None, 'unet': None, 'target': None}

        val_data = np.load(self.data_folder + "val_data.npz")['arr'][-1]
        with torch.no_grad():
            for model_name in model_dict.keys():
                model = model_dict[model_name]
                data = torch.Tensor(val_data).to(self.dtype).to(self.device)
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
                    sos_token = torch.ones(self.train_batch_size, 1).to(self.dtype).to(self.device) * self.sos_token
                    tgt = target
                    tgt = torch.cat((sos_token, tgt[:, :-1]), dim=1)
                    output = model(tgt, input).to(self.dtype).squeeze()

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
            plt.plot(minutes, value.cpu().detach().numpy()[-1], label = key, color=color_dict[key], linewidth = linewidth_dict[key], linestyle = linestyle_dict[key])

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
        # plt.figure(figsize=(8, 6))
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
        # plt.figure(figsize=(8, 6))
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

        performance_dict = {
            'cnn': {
                'mean': cnn_acc.mean().item(),
                'std': cnn_acc.std().item(),
                'min': cnn_acc.min().item(),
                'max': cnn_acc.max().item()
            },
            'unet': {
                'mean': unet_acc.mean().item(),
                'std': unet_acc.std().item(), 
                'min': unet_acc.min().item(),
                'max': unet_acc.max().item()
            },
            'lstm': {
                'mean': lstm_acc.mean().item(),
                'std': lstm_acc.std().item(),
                'min': lstm_acc.min().item(),
                'max': lstm_acc.max().item()
            }, 
            'transformer': {
                'mean': transformer_acc.mean().item(),
                'std': transformer_acc.std().item(),
                'min': transformer_acc.min().item(),
                'max': transformer_acc.max().item()
            }
        }

        print(performance_dict)

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

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()


if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    # mainDir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    feature_lst = ["sugar", "carb", "min", "hba1c", "eda", "hr", "temp", "acc_x", "acc_y", "acc_z"]
    obj = Analysis(mainDir)
    # obj.plot_lopocv()
    # obj.monte_carlo_dropout()
    # obj.plot_performance()
    # obj.plot_output()
    obj.feature_ablation(feature_lst)
