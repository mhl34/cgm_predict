# from pytorch_lightning.callbacks import EarlyStopping
# import ray
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# from ray.tune.schedulers import ASHAScheduler
# from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MetricCollection
# from torch.utils.data import Dataset
# from torch.nn import HuberLoss, L1Loss
# import pandas as pd
# import numpy as np
# import neurokit2 as nk
# from scipy import signal
# from datetime import datetime
# from utils import dateParser
# import matplotlib.pyplot as plt
# from darts import TimeSeries
# from darts.models import (
#     TransformerModel,
#     NBEATSModel,
#     BlockRNNModel,
#     NHiTSModel,
#     DLinearModel,
#     NLinearModel,
#     TiDEModel,
#     TSMixerModel
# )
# from darts.dataprocessing.transformers import Scaler
# from darts.metrics import mape
# from pytorch_lightning.callbacks import EarlyStopping
# import pickle
# import os
# import torch
# import pytorch_lightning as pl
# import logging
# from tqdm import tqdm

# # train function
# def train_model(model_args, callbacks, train_target, train_data, val_target, val_data):
#     # Ensure the past covariates cover the required history
#     input_chunk_length = 20
#     output_chunk_length = 20

#     torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])

#     model = TransformerModel(
#         input_chunk_length=input_chunk_length,
#         output_chunk_length=output_chunk_length,
#         model_name='transformer_model',
#         torch_metrics=torch_metrics,
#         pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
#         **model_args)


#     model.fit(
#         series=train_target,
#         past_covariates=train_data,
#         val_series=val_target,
#         val_past_covariates=val_data,
#         verbose=True
#     )

# df = pd.read_pickle("data/big_df.pkl")
# data = df.drop(columns = ['gluc', 'time'])

# target_ts = TimeSeries.from_dataframe(df[['gluc']])
# data_ts = TimeSeries.from_dataframe(data)

# target_scaler = Scaler()
# data_scaler = Scaler()

# target_ts = target_scaler.fit_transform(target_ts)
# data_ts = data_scaler.fit_transform(data_ts)

# # Adjust the split to ensure sufficient history in validation
# train_target, val_target = target_ts.split_before(0.8)
# train_data, val_data = data_ts.split_before(0.8)

# ray.init(num_cpus=8, num_gpus=1, ignore_reinit_error=True)

# stopper = EarlyStopping(
#     monitor = 'val_MeanAbsolutePercentageError',
#     patience = 5,
#     min_delta = 0.001,
#     mode = 'min'
# )


# tune_callback = TuneReportCallback(
#     {
#         "loss": "val_loss",
#         "MAPE": "val_MeanAbsolutePercentageError",
#     },
#     on="validation_end",
# )

# config = {
#     'batch_size': tune.choice([16, 32, 64, 128, 256]),
#     'nhead': tune.choice([2, 4, 8, 16, 32]),
#     'dim_feedforward': tune.choice([2, 4, 8, 16, 32]),
#     'num_encoder_layers': tune.choice([2, 4, 8, 16]),
#     'num_decoder_layers': tune.choice([2, 4, 8, 16]),
#     'dropout': tune.uniform(0, 0.5)
# }


# resources_per_trial = {
#     "cpu": 8,
#     "gpu": 1
# }


# reporter = CLIReporter(
#     parameter_columns = list(config.keys()),
#     metric_columns = ['loss', 'MAPE', 'training_iteration'],
# )

# num_samples = 10

# scheduler = ASHAScheduler(max_t = 100, grace_period = 3, reduction_factor = 2)

# train_fn_with_parameters = tune.with_parameters(
#     train_model, callbacks = [stopper, tune_callback], train_target = train_target, train_data = train_data, val_target = val_target, val_data = val_data
# )


# analysis = tune.run(
#     train_fn_with_parameters,
#     resources_per_trial = resources_per_trial,
#     metric = 'MAPE',
#     mode = 'min',
#     config = config,
#     num_samples = num_samples,
#     scheduler = scheduler,
#     progress_reporter = reporter,
#     name = 'tune_darts'
# )


# print("Best hyperparameters found were: ", analysis.best_config)


import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
import torch
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MetricCollection
from darts import TimeSeries
from darts.models import TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import os
import logging

# Configure logging
optuna.logging.set_verbosity(optuna.logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load data
df = pd.read_pickle("data/big_df.pkl")
data = df.drop(columns=['gluc', 'time'])

target_ts = TimeSeries.from_dataframe(df[['gluc']])
data_ts = TimeSeries.from_dataframe(data)

target_scaler = Scaler()
data_scaler = Scaler()

target_ts = target_scaler.fit_transform(target_ts)
data_ts = data_scaler.fit_transform(data_ts)

# Adjust the split to ensure sufficient history in validation
train_target, val_target = target_ts.split_before(0.8)
train_data, val_data = data_ts.split_before(0.8)

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    batch_size = trial.suggest_categorical('batch_size', [16, 64, 256])
    nhead = trial.suggest_categorical('nhead', [2, 8, 32])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [2, 8, 32])
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', [2, 16])
    num_decoder_layers = trial.suggest_categorical('num_decoder_layers', [2, 16])
    dropout = trial.suggest_uniform('dropout', 0, 0.5)

    # Ensure the past covariates cover the required history
    input_chunk_length = 30
    output_chunk_length = 30

    torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])

    model = TransformerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        model_name='transformer_model',
        torch_metrics=torch_metrics,
        batch_size=batch_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        pl_trainer_kwargs={"callbacks": [PyTorchLightningPruningCallback(trial, monitor="val_MeanAbsolutePercentageError"), EarlyStopping(monitor='val_MeanAbsolutePercentageError', patience=5, min_delta=0.001, mode='min')], "enable_progress_bar": False},
    )

    model.fit(
        series=train_target,
        past_covariates=train_data,
        val_series=val_target,
        val_past_covariates=val_data,
        verbose=True
    )

    # Validation MAPE
    val_mape = mape(model.predict(n=len(val_target), past_covariates=val_data), val_target)
    
    return val_mape

# Create study and optimize
def tune():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=600, n_jobs = -1)

    # Log the best trial
    trial = study.best_trial
    logging.info(f"Best hyperparameters: {trial.params}")
    logging.info(f"Best value (MAPE): {trial.value}")

    print("Best hyperparameters found were: ", study.best_trial.params)

tune()