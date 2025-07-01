from path import Path
from utils import Config
from torch.utils.data import DataLoader
from dataloaders.datasets import *
from dataloaders.transform_loader import get_tfms
import os


class RandomKTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        if cfg.experiment.file == 'none':
            csv_file = 'hpa_train_notest.csv'
        else:
            csv_file = cfg.experiment.file
        train = pd.concat([pd.read_csv(path / 'split' / cf) for cf in csv_file.split(';')], axis=0)
        self.train_meta = train[train.fold != cfg.experiment.run_fold]
        # self.train_meta = self.train_meta.sample(frac=0.00005)

        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.00005)

    def get_dataloader(self, train_shuffle=True):
        print('[ √ ] Using transformation: {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)

        train_ds = RANZERDataset(df=self.train_meta, tfms=train_tfms, cfg=self.cfg)
        train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                              num_workers=self.cfg.transform.num_preprocessor,
                              shuffle=train_shuffle, drop_last=True, pin_memory=True)

        return train_dl, None, None


class RandomKTrainTestSplit_Mix_transform_strong:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        if cfg.experiment.file == 'none':
            csv_file = 'hpa_train_notest.csv'
        else:
            csv_file = cfg.experiment.file
        train = pd.concat([pd.read_csv(path / 'split' / cf) for cf in csv_file.split(';')], axis=0)

        self.train_meta = train[train.fold != cfg.experiment.run_fold]
        # self.train_meta = self.train_meta.sample(frac=0.00005)
        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.00005)

    def get_dataloader(self, train_shuffle=True):
        print('[ √ ] Using transformation: {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        train_ds = RANZERDataset_Mix_transform_strong(df=self.train_meta, tfms=train_tfms, cfg=self.cfg)
        train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                              num_workers=self.cfg.transform.num_preprocessor,
                              shuffle=train_shuffle, drop_last=True, pin_memory=True)

        return train_dl, None, None


class RandomKTrainTestSplit_Mix_transform_strong_HPAv23:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        if cfg.experiment.file == 'none':
            csv_file = 'hpa_train_notest.csv'
        else:
            csv_file = cfg.experiment.file
        train = pd.concat([pd.read_csv(path / 'split' / cf) for cf in csv_file.split(';')], axis=0)
        self.train_meta = train[train.fold != cfg.experiment.run_fold]
        # self.train_meta = self.train_meta.sample(frac=0.00005)
        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.00005)

    def get_dataloader(self, train_shuffle=True):
        print('[ √ ] Using transformation: {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        train_ds = HPAv23Dataset_Mix_transform_strong(df=self.train_meta, tfms=train_tfms,
                                                      cfg=self.cfg)
        train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                              num_workers=self.cfg.transform.num_preprocessor,
                              shuffle=train_shuffle, drop_last=True, pin_memory=True)

        return train_dl, None, None

