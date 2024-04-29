import argparse
import configparser
import glob
from pathlib import Path
import random
from typing import Dict
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn as nn
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from arguments import create_argument_parser
from diffus_ie.data_modules.data_modules import EREDataModule
from diffus_ie.models.trainer import DiffusIEModel

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
TODO: Add EMA to stablize the training process
"""

def run(trial: optuna.trial.Trial = None, args = None):
    seed_everything(1741, workers=True)
    # Args configure
    args_config = configparser.ConfigParser(allow_no_value=False)
    args_config.read(args.config_file)
    job = args.data_name
    assert job in args_config
    if trial != None:
        # Tuning space
        config = {'num_epochs': trial.suggest_categorical('num_epochs', [20, 30, 50]),
                  'encoder_warm_up': trial.suggest_categorical('encoder_warm_up', [1, 3, 5, 10]),
                  'diffusion_train_step': trial.suggest_categorical('diffusion_train_step', [1000]),
                  'diffusion_inference_step': trial.suggest_categorical('diffusion_inference_step', [100]),
                  'block_type': trial.suggest_categorical('block_type', ["in-context", "adaLN-Zero"]), # 
                  'diff_depth': trial.suggest_categorical('diff_depth', [2, 4, 6, 8]),
                  'diff_lr': trial.suggest_categorical('diff_lr', [1e-6, 3e-6, 5e-6, 1e-5]),
                  'encoder_lr': trial.suggest_categorical('encoder_lr', [1e-6, 3e-6, 5e-6, 8e-6, 1e-5, 2e-5]),
                  'head_lr': trial.suggest_categorical('head_lr', [1e-5, 3e-5, 5e-5, 8e-5, 1e-4]),}
        
        print("Hyperparams: {}".format(config))
        config.update(dict(args_config.items(job)))
        args = vars(args)
        for key in config:
            if config[key] in ['True', 'False']:
                config[key] = True if config[key]=='True' else False
            if config[key] == 'None':
                config[key] = None
            if type(config[key]) == str:
                if config[key].isdigit():
                    config[key] = int(config[key])
            args[key] = config[key]
        args = argparse.Namespace(**args)
    else:
        args = dict(vars(args))
        job_config = dict(args_config.items(job))
        for key in job_config:
            if job_config[key] in ['True', 'False']:
                job_config[key] = True if job_config[key]=='True' else False
            if job_config[key] == 'None':
                job_config[key] = None
            if type(job_config[key]) == str:
                if job_config[key].isdigit():
                    job_config[key] = int(job_config[key])
        args.update(job_config)
        args = argparse.Namespace(**args)
    print(args)

    f1s = []
    ps = []
    rs = []
    for i in range(int(args.n_fold)):
        print(f"TRAINING AND TESTING IN FOLD {i}: ")
        
        dm = EREDataModule(params=args, fold=i)
        model = DiffusIEModel(params=args)

        checkpoint_save_dir = os.path.join(
            args.output_dir,
            f'{args.data_name}'
            f'-intra_{args.intra}'
            f'-inter_{args.inter}'
            f'-block_type_{args.block_type}',
            f'fold_{i}')
        checkpoint_save_dir = Path(checkpoint_save_dir)
        checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
                                    dirpath=checkpoint_save_dir,
                                    every_n_epochs=1,
                                    save_top_k=1,
                                    monitor='hp_metric',
                                    mode='max',
                                    filename='{epoch}-{hp_metric:.2f}', # this cannot contain slashes 
                                    )
        lr_logger = LearningRateMonitor(logging_interval='step')
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=checkpoint_save_dir)

        if trial != None:
            optuna_callback = PyTorchLightningPruningCallback(trial, monitor="hp_metric/f1_val")
            callbacks = [lr_logger, checkpoint_callback, optuna_callback,]
        else:
            callbacks = [lr_logger, checkpoint_callback,]

        if args.tuning or args.training:
            trainer = Trainer(
                logger=tb_logger,
                max_epochs=args.num_epochs + args.encoder_warm_up, 
                accelerator="auto" if torch.cuda.is_available() else "cpu", 
                devices=args.devices,
                precision='16-mixed',
                # strategy='ddp_find_unused_parameters_true',
                accumulate_grad_batches=args.gradient_accumulation_steps,
                num_sanity_val_steps=3, 
                check_val_every_n_epoch=1,
                callbacks = [*callbacks],
            )

            print("Training....")
            trainer.fit(model, datamodule=dm)
            p, r, f1 = model.val_result

            if args.testing:
                best_checkpoint_path = checkpoint_callback.best_model_path
                model = DiffusIEModel.load_from_checkpoint(best_checkpoint_path, map_location=torch.device('cpu'))
                trainer.test(model, datamodule=dm)
                p, r, f1 = model.result

        if args.testing and not (args.training or args.tuning):
            assert args.load_checkpoint != None
            checkpoint = args.load_checkpoint
            model = DiffusIEModel.load_from_checkpoint(checkpoint, map_location=torch.device('cpu'))

            trainer = Trainer(
                logger=tb_logger,
                max_epochs=args.num_epochs, 
                accelerator="auto" if torch.cuda.is_available() else "cpu", 
                devices=1,
                precision='16-mixed',
                accumulate_grad_batches=args.gradient_accumulation_steps,
                num_sanity_val_steps=3, 
                check_val_every_n_epoch=2,
                callbacks = [lr_logger,],
            )

            print("Testing .....")
            trainer.test(model, datamodule=dm)
            p, r, f1 = model.result

        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        print(f"RESULT IN FOLD {i}: ")
        print(f"F1: {f1}")
        print(f"P: {p}")
        print(f"R: {r}")
        # shutil.rmtree(f'{output_dir}')
    
    f1 = sum(f1s)/len(f1s)
    p = sum(ps)/len(ps)
    r = sum(rs)/len(rs)
    print(f"F1: {f1} - P: {p} - R: {r}")

    record_file_name = f'result_{args.data_name}_intra_{args.intra}_inter_{args.inter}.log'
    with open(record_file_name, 'a', encoding='utf-8') as f:
        f.write(f"Dataset: {args.data_name} (Intra: {args.intra}, Inter: {args.inter}) \n")
        f.write(f"Random_state: 1741\n")
        if trial != None:
            f.write(f"Hyperparams: \n {trial.params.items()}\n")
        else:
            f.write(f"Hyperparams: \n {args}\n")
        f.write(f"F1: {f1}  \n")
        f.write(f"P: {p} \n")
        f.write(f"R: {r} \n")
        f.write(f"{'--'*10} \n")

    return f1

if __name__ == '__main__':
    # parse arguments
    args = create_argument_parser()
    if args.tuning:
        pruner = optuna.pruners.MedianPruner()
        func = lambda trial: run(trial=trial, args=args)

        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(func, n_trials=50)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        run(args=args)
