from models.DualBranchMIL import *
from configs import Config


def get_model(cfg: Config):
    if cfg.model.name.split('-')[0] in ['max']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a DualBranchMIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return DualBranchMIL(pretrained=cfg.model.param['pretrained'], model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['VMamba']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a VMamba, mdl_name: {cfg.model.name}, pool: {pool}')
        return DualBranchMILVMamba(pretrained=cfg.model.param['pretrained'], pool=pool, patch_size=cfg.train.batch_size,
                                   dropout=cfg.model.param.get('dropout', 0.1))
    if cfg.model.name.split('-')[0] in 'transformer':
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a DualBranchMILTransformer, mdl_name: {cfg.model.name}, pool: {pool}')
        return DualBranchMILTrans(pretrained=cfg.model.param['pretrained'], model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))

