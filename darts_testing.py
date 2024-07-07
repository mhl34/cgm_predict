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
import pytorch_lightning as pl
import logging
from tqdm import tqdm

# model initialization methods
# models
def init_transformer():
    return TransformerModel(
        # model_name='transformer_model',
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
        # model = "RNN",
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
        # model = "LSTM",
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
        # model_name = "nbeats_model",
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
        # model_name = "nhits_model",
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
        # model_name = "dlinear_model",
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
        # model_name = "nlinear_model",
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
        # model_name = "tide_model",
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
        # model_name = "tsmixer_model",
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

# Suppress specific log messages
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

input_chunk_length = 20
output_chunk_length = 20
num_epochs = 100

samples = [str(i).zfill(3) for i in range(1, 17)]
optimizer_kwargs = {
    'lr': 1e-4,
    'weight_decay':  1e-8,
}
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


def personalized_train(models, sample, df, input_chunk_length, output_chunk_length):
    """
    function: train for the lopocv loop
    returns: results from training as well as testing from the left out sample
    """
    if 'Unnamed: 0' in df:
        df = df.drop(columns = ['time', 'Unnamed: 0'])
    else:
        df = df.drop(columns = ['time'])

        # Split index
    split_index = int(len(df) * 0.8)

    # Splitting the DataFrame
    train_df = df[:split_index]
    test_df = df[split_index:]

    train_target = train_df[['gluc']]
    train_data = train_df.drop(columns = ['gluc'])

    train_target, val_target = TimeSeries.from_dataframe(train_target).split_before(0.8)
    train_data, val_data = TimeSeries.from_dataframe(train_data).split_before(0.8)

    data_scaler = Scaler()
    target_scaler = Scaler()

    train_target = target_scaler.fit_transform(train_target)
    train_data = data_scaler.fit_transform(train_data)
    val_target = target_scaler.fit_transform(val_target)
    val_data = data_scaler.fit_transform(val_data)


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

        mape_dict[model_name] = 0
        for start_idx in tqdm(range(0, len(test_df) - input_chunk_length - output_chunk_length, output_chunk_length), "testing"):
            test_target = test_df[['gluc']].iloc[start_idx: start_idx + input_chunk_length]
            test_truth = test_df[['gluc']].iloc[start_idx + input_chunk_length: start_idx + input_chunk_length + output_chunk_length]
            test_data = test_df.drop(columns = ['gluc']).iloc[start_idx: start_idx + input_chunk_length]

            test_target = TimeSeries.from_dataframe(test_target).shift(len(train_target) - input_chunk_length)
            test_truth = TimeSeries.from_dataframe(test_truth).shift(len(train_target) - input_chunk_length)
            test_data = TimeSeries.from_dataframe(test_data).shift(len(train_target) - input_chunk_length)

            test_target = target_scaler.fit_transform(test_target)
            test_data = data_scaler.fit_transform(test_data)

            predictions = model.predict(
                n=len(test_truth), 
                series = test_target, 
                past_covariates=test_data,
                verbose=False,
                show_warnings=False,
            )

            predictions = target_scaler.inverse_transform(predictions)

            # Evaluate the model
            mape_val = mape(test_truth, predictions)

            mape_dict[model_name] += mape_val

        mape_dict[model_name] /= (len(test_df) - input_chunk_length - output_chunk_length) // input_chunk_length

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} darts prediction")
        plt.plot(predictions.values(), label = "Predictions")
        plt.plot(test_truth.values(), label = "Target")
        plt.legend()
        plt.savefig(f"plots/dart_{model_name}_personal.png")
        plt.close()

        print(f'MAPE: {mape_dict[model_name]}')

        # save model
        model.save(f"saved_models/{model_name}_darts_personal_{sample}.pkl")

    return mape_dict

def lopocv_train(models, sel_sample, train_df, test_df, input_chunk_length, output_chunk_length):
    """
    function: train for the lopocv loop
    returns: results from training as well as testing from the left out sample
    """
    if 'Unnamed: 0' in train_df:
        train_df = train_df.drop(columns = ['time', 'Unnamed: 0'])
    else:
        train_df = train_df.drop(columns = ['time'])
    if 'Unnamed: 0' in test_df:
        test_df = test_df.drop(columns = ['time', 'Unnamed: 0'])
    else:
        test_df = test_df.drop(columns = ['time'])
    train_target = train_df[['gluc']]
    train_data = train_df.drop(columns = ['gluc'])

    train_target, val_target = TimeSeries.from_dataframe(train_target).split_before(0.9)
    train_data, val_data = TimeSeries.from_dataframe(train_data).split_before(0.9)

    data_scaler = Scaler()
    target_scaler = Scaler()

    train_target = target_scaler.fit_transform(train_target)
    train_data = data_scaler.fit_transform(train_data)
    val_target = target_scaler.fit_transform(val_target)
    val_data = data_scaler.fit_transform(val_data)


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
        mape_dict[model_name] = 0
        for start_idx in tqdm(range(0, len(test_df) - input_chunk_length - output_chunk_length, output_chunk_length), "testing"):
            test_target = test_df[['gluc']].iloc[start_idx: start_idx + input_chunk_length]
            test_truth = test_df[['gluc']].iloc[start_idx + input_chunk_length: start_idx + input_chunk_length + output_chunk_length]
            test_data = test_df.drop(columns = ['gluc']).iloc[start_idx: start_idx + input_chunk_length]

            test_target = TimeSeries.from_dataframe(test_target).shift(len(train_target) - input_chunk_length)
            test_truth = TimeSeries.from_dataframe(test_truth).shift(len(train_target) - input_chunk_length)
            test_data = TimeSeries.from_dataframe(test_data).shift(len(train_target) - input_chunk_length)

            test_target = target_scaler.fit_transform(test_target)
            test_data = data_scaler.fit_transform(test_data)

            predictions = model.predict(
                n=len(test_truth), 
                series = test_target, 
                past_covariates=test_data,
                verbose=False,
                show_warnings=False,
            )

            predictions = target_scaler.inverse_transform(predictions)

            # Evaluate the model
            mape_val = mape(test_truth, predictions)

            mape_dict[model_name] += mape_val

        mape_dict[model_name] /= (len(test_df) - input_chunk_length - output_chunk_length) // input_chunk_length

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} darts prediction")
        plt.plot(predictions.values(), label = "Predictions")
        plt.plot(test_truth.values(), label = "Target")
        plt.legend()
        plt.savefig(f"plots/dart_{model_name}.png")
        plt.close()

        print(f'MAPE: {mape_dict[model_name]}')

        # save model
        model.save(f"saved_models/{model_name}_darts_no_{sel_sample}.pkl")

    return mape_dict

def personalized(samples):
    """
    function: trains on the first half of a person's data and then predicts
    returns: saved models and created files for each of the recorded MAPE values
    """
    for sample in samples:
        if os.path.exists(f"performance/mape_dict_personal_{sample}.pickle"):
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

        print("===================================")
        print(f"sample {sample} personalized model")
        print("===================================")

        # load the dataframe of the person's data
        df = pd.read_csv(f"data/data_{sample}.csv")

        if df.empty:
            continue

        # print(test_df.head())

        mape_dict = personalized_train(models, sample, df, input_chunk_length, output_chunk_length)

        with open(f"performance/mape_dict_personal_{sample}.pickle", "wb") as file:
            pickle.dump(mape_dict, file)

def lopocv(samples):
    """
    function: leave-one-person-out cross validation method with the various models
    returns: saved models and created files for each of the recorded MAPE values for each of the models used
    """
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

        if test_df.empty:
            continue

        # print(test_df.head())

        mape_dict = lopocv_train(models, sel_sample, train_df, test_df, input_chunk_length, output_chunk_length)

        with open(f"performance/mape_dict_no_{sel_sample}.pickle", "wb") as file:
            pickle.dump(mape_dict, file)

# LOPOCV analysis
print("LOPOCV analysis")
models_lst = list(models.keys())
total = 0
lopocv_mape_dict = {model: 0 for model in models_lst}
for sample in samples:
    if not os.path.exists(f"performance/mape_dict_no_{sample}.pickle"):
        continue
    with open(f"performance/mape_dict_no_{sample}.pickle", "rb") as file:
        mape_dict = pickle.load(file)
    min_val = min(mape_dict.values())
    for model in models_lst:
        lopocv_mape_dict[model] += mape_dict[model] / min_val
    total+=1

models_lst.sort(key=lambda x: lopocv_mape_dict[x])
for model in models_lst:
    print(f"model: {model} value: {lopocv_mape_dict[model] / total}")

# Personalized Analysis
print("Personalized analysis")
models_lst = list(models.keys())
total = 0
personal_mape_dict = {model: 0 for model in models_lst}
for sample in samples:
    if not os.path.exists(f"performance/mape_dict_personal_{sample}.pickle"):
        continue
    with open(f"performance/mape_dict_personal_{sample}.pickle", "rb") as file:
        mape_dict = pickle.load(file)
    min_val = min(mape_dict.values())
    for model in models_lst:
        personal_mape_dict[model] += mape_dict[model] / min_val
    total+=1

models_lst.sort(key=lambda x: personal_mape_dict[x])
for model in models_lst:
    print(f"model: {model} value: {personal_mape_dict[model] / total}")