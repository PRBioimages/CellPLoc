from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from basic_train import Pseudo_train_baseline, Pseudo_train_HPAv21, Pseudo_train_HPAv23
from scheduler import get_scheduler
from utils import load_matched_state
from torch.utils.tensorboard import SummaryWriter
import torch
try:
    from apex import amp
except:
    pass
import albumentations as A
from dataloaders.transform_loader import get_tfms
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from configs import Config
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print('[ √ ] Landmark!')
    args, cfg = parse_args()
    result_path = prepare_for_result(cfg)
    writer = SummaryWriter(log_dir=result_path)
    cfg.dump_json(result_path / 'config.json')

    # modify for training multiple fold
    # cfg.experiment.run_fold = 1
    if cfg.experiment.run_fold == -1:
        for i in range(cfg.experiment.fold):
            torch.cuda.empty_cache()
            print('[ ! ] Full fold coverage training! for fold: {}'.format(i))
            cfg.experiment.run_fold = i
            train_dl, _, _ = get_dataloader(cfg)(cfg).get_dataloader()
            print('[ i ] The length of train_dl is {}'.format(len(train_dl)))
            model = get_model(cfg).cuda()
            if not cfg.model.from_checkpoint == 'none':
                print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
                load_matched_state(model, torch.load(cfg.model.from_checkpoint))
            loss_func = get_loss(cfg)
            if cfg.train.freeze_backbond:
                print('[ i ] freeze backbone')
                model.model.requires_grad = False
            optimizer = get_optimizer(model, cfg)
            print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name,
                                                                         cfg.optimizer.name))
            if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
                print('[ i ] Call apex\'s initialize')
                model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
            if not cfg.scheduler.name == 'none':
                scheduler = get_scheduler(cfg, optimizer, len(train_dl))
            else:
                scheduler = None
            if len(cfg.basic.GPU) > 1:
                model = torch.nn.DataParallel(model)
            Pseudo_train_HPAv21(cfg, model, train_dl, loss_func, optimizer, result_path, scheduler, writer)
    else:
        train_dl, _, _ = get_dataloader(cfg)(cfg).get_dataloader()
        print('[ i ] The length of train_dl is {}'.format(len(train_dl)))
        model = get_model(cfg).cuda()
        if not cfg.model.from_checkpoint == 'none':
            print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
            load_matched_state(model, torch.load(cfg.model.from_checkpoint, map_location='cpu'))
        loss_func = get_loss(cfg)
        if cfg.train.freeze_backbond:
            print('[ i ] freeze backbone')
            model.model.requires_grad = False
        optimizer = get_optimizer(model, cfg)
        print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))
        if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
            print('[ i ] Call apex\'s initialize')
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
        if not cfg.scheduler.name == 'none':
            scheduler = get_scheduler(cfg, optimizer, len(train_dl))
        else:
            scheduler = None
        if len(cfg.basic.GPU) > 1:
            model = torch.nn.DataParallel(model)
        Pseudo_train_HPAv21(cfg, model, train_dl, loss_func, optimizer, result_path, scheduler, writer)
