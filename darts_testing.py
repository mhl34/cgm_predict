from torch.utils.data import Dataset
from torch.nn import HuberLoss, L1Loss
import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy import signal
from datetime import datetime
from utils import dateParser
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import (
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    NHiTSModel,
    DLinearModel,
    NLinearModel,
    TiDEModel,
    TSMixerModel
)
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from pytorch_lightning.callbacks import EarlyStopping
import pickle
import os
import torch

input_chunk_length = 20
output_chunk_length = 20
num_epochs = 100

# train parameters
optimizer_kwargs = {
    'lr': 1e-4,
    'weight_decay':  1e-8,
}

def train(models, train_df, test_df, input_chunk_length, output_chunk_length):
    train_df = train_df.drop(columns = ['time', 'Unnamed: 0'])
    test_df = test_df.drop(columns = ['time', 'Unnamed: 0'])
    train_target = train_df[['gluc']]
    train_data = train_df.drop(columns = ['gluc'])
    test_target = test_df[['gluc']].iloc[: input_chunk_length]
    test_truth = test_df[['gluc']].iloc[input_chunk_length:]
    test_data = test_df.drop(columns = ['gluc']).iloc[: len(test_df) - input_chunk_length]

    train_target, val_target = TimeSeries.from_dataframe(train_target).split_before(0.9)
    train_data, val_data = TimeSeries.from_dataframe(train_data).split_before(0.9)
    test_target = TimeSeries.from_dataframe(test_target).shift(len(train_target) - input_chunk_length)
    test_truth = TimeSeries.from_dataframe(test_truth).shift(len(train_target) - input_chunk_length)
    test_data = TimeSeries.from_dataframe(test_data).shift(len(train_target) - input_chunk_length)

    data_scaler = Scaler()
    target_scaler = Scaler()

    train_target = target_scaler.fit_transform(train_target)
    train_data = data_scaler.fit_transform(train_data)
    val_target = target_scaler.fit_transform(val_target)
    val_data = data_scaler.fit_transform(val_data)
    test_target = target_scaler.fit_transform(test_target)
    test_truth = target_scaler.fit_transform(test_truth)
    test_data = data_scaler.fit_transform(test_data)


    # store the mape of the validation
    mape_dict = {}
    for model_name in models.keys():
        print(f"model_name: {model_name}")
        model = models[model_name]

        stopper = EarlyStopping(
            monitor = 'val_loss',
            patience = 5,
            min_delta = 1e-4,
            mode = 'min'
        )

        model.pl_trainer_kwargs = {"callbacks": [stopper]}

        model.fit(
            series=train_target,
            past_covariates=train_data,
            val_series=val_target,
            val_past_covariates=val_data,
            verbose=True
        )

        predictions = model.predict(
            n=len(test_truth), 
            series = test_target, 
            past_covariates=test_data
        )

        predictions = target_scaler.inverse_transform(predictions)
        test_truth = target_scaler.inverse_transform(test_truth)

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} darts prediction")
        plt.plot(predictions.values(), label = "Predictions")
        plt.plot(test_truth.values(), label = "Target")
        plt.legend()
        plt.savefig(f"plots/dart_{model_name}.png")

        # Evaluate the model
        mape_val = mape(test_truth, predictions)
        print(f'MAPE: {mape_val}')

        mape_dict[model_name] = mape_val

        # redo the inverse transform
        test_truth = target_scaler.fit_transform(test_truth)

        # save model
        model.save(f"saved_models/{model_name}_darts.pkl")

    return mape_dict

# models
def init_transformer():
    return TransformerModel(
        model_name='transformer_model',
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=num_epochs,
        force_reset = True,
        random_state = 42,
        dropout = 0.3,
        optimizer_kwargs = optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_rnn():
    return BlockRNNModel(
        model = "RNN",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        dropout = 0.3,
        optimizer_kwargs = optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_lstm(): 
    return BlockRNNModel(
        model = "LSTM",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        dropout = 0.3,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_nbeats():
    return NBEATSModel(
        model_name = "nbeats_model",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        random_state = 42,
        dropout = 0.3,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_nhits():
    return NHiTSModel(
        model_name = "nhits_model",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        dropout = 0.3,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_dlinear():
    return DLinearModel(
        model_name = "dlinear_model",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs = {
            "T_max": num_epochs
        }
    )

def init_nlinear():
    return NLinearModel(
        model_name = "nlinear_model",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_tide():
    return TiDEModel(
        model_name = "tide_model",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

def init_tsmixer():
    return TSMixerModel(
        model_name = "tsmixer_model",
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs = num_epochs,
        force_reset = True,
        random_state = 42,
        dropout = 0.3,
        optimizer_kwargs =  optimizer_kwargs,
        loss_fn = HuberLoss(),
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

samples = [str(i).zfill(3) for i in range(1, 17)]
# lopocv
for sel_sample in samples:
    if os.path.exists(f"performance/mape_dict_no_{sel_sample}.pickle"):
        continue

    models = {
        "dlinear_model": init_dlinear(),
        "nhits_model": init_nhits(),
        "transformer_model": init_transformer(),
        "rnn_model": init_rnn(),
        "lstm_model": init_lstm(),
        "nbeats_model": init_nbeats(),
        "nlinear_model": init_nlinear(),
        "tide_model": init_tide(),
        "tsmixer_model": init_tsmixer()
    }
    print("=============================")
    print(f"sample {sel_sample} left out")
    print("=============================")
    train_df = pd.DataFrame()
    # create testing data without the sel_sample
    for sample in samples:
        if sample == sel_sample:
            continue
        df = pd.read_csv(f"data/data_{sample}.csv")
        train_df = pd.concat([train_df, df], axis = 0, ignore_index = True)
    # test_data with sel_sample
    test_df = pd.read_csv(f"data/data_{sel_sample}.csv")

    if len(test_df) == 0:
        continue

    mape_dict = train(models, train_df, test_df, input_chunk_length, output_chunk_length)

    with open(f"performance/mape_dict_no_{sel_sample}.pickle", "wb") as file:
        pickle.dump(mape_dict, file)

