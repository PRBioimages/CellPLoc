import yaml
import json
import os
import glob


class Element:
    def __repr__(self):
        return ', '.join(['{}: {}'.format(k, v) for k, v in self.__dict__.items()])


class DPP(Element):
    def __init__(self, dict):
        self.nodes = 1
        self.gpus = 4
        self.rank = 0
        self.sb = True
        self.mode = 'train'
        self.checkpoint = None


class Basic(Element):
    def __init__(self, dict):
        self.seed = dict.get('seed', '233')
        # self.GPU = str(dict.get('GPU', '0'))
        self.GPU = dict.get('GPU', '0')
        self.id = dict.get('id', 'unnamed')
        self.debug = dict.get('debug', False)
        self.mode = dict.get('mode', 'train')
        self.search = dict.get('search', False)
        self.amp = dict.get('amp', 'None')
        if len(self.GPU) > 1:
            self.GPU = [str(x) for x in self.GPU]


class Softmatch(Element):
    def __init__(self, dict):
        self.num_classes = dict.get('num_classes', 19)
        self.ema_p = dict.get('ema_p', 0.999)
        self.dist_uniform = dict.get('dist_uniform', True)
        self.n_sigma = dict.get('n_sigma', 2)
        self.per_class = dict.get('per_class', False)


class Experiment(Element):
    def __init__(self, dict):
        self.name = dict.get('name', 'KFold')
        self.mix_transform_strong = dict.get('Mix_transform_strong', False)
        self.use_hard_label = dict.get('use_hard_label', False)
        self.use_hard_mask = dict.get('use_hard_mask', False)
        self.alpha = dict.get('alpha', 1)
        self.random_state = dict.get('random_state', '2333')
        self.fold = dict.get('fold', 5)
        self.run_fold = dict.get('run_fold', 0)
        self.weight = dict.get('weight', False)
        self.method = dict.get('method', 'none')
        self.tile = dict.get('tile', 12)
        self.count = dict.get('count', 16)
        self.regression = dict.get('regression', False)
        self.scale = dict.get('scale', 1)
        self.level = int(dict.get('level', 1))
        self.public = dict.get('public', True)
        self.merge = dict.get('merge', True)
        self.n = dict.get('N', True)
        self.batch_sampler = dict.get('batch_sampler', False)
        # batch sampler
        #   initial_miu: 6
        #   miu_factor: 6
        self.pos_ratio = dict.get('pos_ratio', 16)
        self.externals = dict.get('externals', [])
        self.initial_miu = dict.get('initial_miu', -1)
        self.miu_factor = dict.get('miu_factor', -1)
        self.full = dict.get('full', False)
        self.preprocess = dict.get('preprocess', 'train')
        self.image_only = dict.get('image_only', True)
        self.skip_outlier = dict.get('skip_outlier', False)
        self.outlier = dict.get('outlier', 'train')
        self.outlier_method = dict.get('outlier_method', 'drop')
        self.file = dict.get('csv_file', 'none')
        self.smoothing = dict.get('smoothing', 0)
        self.label = dict.get('label', 'none')
        self.onehotpseudo = dict.get('onehotpseudo', False)
        self.pseudo_dir = dict.get('pseudo_dir', None)
        self.part_pseudo = dict.get('part_pseudo', True)
        self.extraData = dict.get('extraData', None)


class Data(Element):
    def __init__(self, dict):
        self.cell = dict.get('cell', 'none')
        self.name = dict.get('name', 'CouldDataset')
        if os.name == 'nt':
            self.data_root = dict.get('dir_nt', '/')
        else:
            self.data_root = dict.get('dir_sv', '/')
        self.transclass = dict.get('transclass', 'all')
        self.withclassor = dict.get('withclassor', 'all')
        self.onlyclassand = dict.get('onlyclassand', 'all')
        self.celllabel = dict.get('celllabel', 'imagelabel')
        # self.hpa_mean = dict.get('hpa_mean', [0.0994, 0.0466, 0.0606, 0.0879])
        # self.hpa_std = dict.get('hpa_std', [0.1406, 0.0724, 0.1541, 0.1264])
        # for aws,
        # /home/sheep/Bengali/data
        # to any user
        try:
            self.data_root = glob.glob('/' + self.data_root.split('/')[1] + '/*/' + '/'.join(self.data_root.split('/')[3:]))[0]
        except:
            self.data_root = 'REPLACE ME PLZ!'


class Model(Element):
    def __init__(self, dict):
        self.name = dict.get('name', 'resnet50')
        self.param = dict.get('params', {})
        # add default true
        if 'dropout' not in self.param:
            self.param['dropout'] = True
        self.from_checkpoint = dict.get('from_checkpoint', 'none')
        self.out_feature = dict.get('out_feature', 1)
        self.need_fp = dict.get('need_fp', False)


class Train(Element):
    '''
      freeze_backbond: 1
      freeze_top_layer_groups: 0
      freeze_start_epoch: 1


    :param dict:
    '''
    def __init__(self, dict):
        self.dir = dict.get('dir', None)
        if not self.dir:
            raise Exception('Training dir must assigned')
        self.batch_size = dict.get('batch_size', 8)
        self.num_epochs = dict.get('num_epochs', 100)
        self.start_epoch = dict.get('start_epoch', 0)
        self.cutmix = dict.get('cutmix', False)
        self.mixup = dict.get('mixup', False)
        self.beta = dict.get('beta', 1)
        self.cutmix_prob = dict.get('cutmix_prob', 0.5)
        self.cutmix_prob_increase = dict.get('cutmix_prob_increase', 0)
        self.validations_round = dict.get('validations_round', 1)
        self.freeze_backbond = dict.get('freeze_backbond', 0)
        self.freeze_top_layer_groups = dict.get('freeze_top_layer_groups', 0)
        self.freeze_start_epoch = dict.get('freeze_start_epoch', 1)
        self.clip = dict.get('clip_grad', None)
        self.combine_mix = dict.get('combine_mix', False)
        self.combine_list = dict.get('combine_list', [])
        self.combine_p = dict.get('combine_p', [])


class Eval(Element):
    def __init__(self, dict):
        self.batch_size = dict.get('batch_size', 32)


class Loss(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.param = dict.get('params', {})
        # if 'class_balanced' not in self.param:
        #     self.param['class_balanced'] = False
        self.weight_type = dict.get('weight_type', 'None')
        self.weight_value = dict.get('weight_value', None)
        self.cellweight = dict.get('cellweight', 1)
        self.imgweight = dict.get('imgweight', 1)
        self.pos_weight = dict.get('pos_weight', 10)


class Optimizer(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.param = dict.get('params', {})
        self.step = dict.get('step', 1)


class Scheduler(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.param = dict.get('params', {})
        self.warm_up = dict.get('warm_up', False)


class Transform(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.val_name = dict.get('val_name', 'None')
        self.param = dict.get('params', {})
        self.num_preprocessor = dict.get('num_preprocessor', 0)
        self.size = dict.get('size', (137, 236))
        self.half = dict.get('half', False)
        self.tiny = dict.get('tiny', False)
        self.smaller = dict.get('smaller', False)
        self.larger = dict.get('larger', False)
        self.random_scale = dict.get('random_scale', False)
        self.random_margin = dict.get('random_margin', False)
        self.random_choice = dict.get('random_choice', False)
        self.shuffle = dict.get('shuffle', False)
        self.scale = dict.get('scale', [])
        self.gray = dict.get('gray', False)


class Config:
    def __init__(self, dict):
        self.param = dict
        self.basic = Basic(dict.get('basic', {}))
        self.softmatch = Softmatch(dict.get('softmatch', {}))
        self.experiment = Experiment(dict.get('experiment', {}))
        self.data = Data(dict.get('data', {}))
        self.model = Model(dict.get('model', {}))
        self.train = Train(dict.get('train', {}))
        self.eval = Eval(dict.get('eval', {}))
        self.loss = Loss(dict.get('loss', {}))
        self.optimizer = Optimizer(dict.get('optimizer', {}))
        self.scheduler = Scheduler(dict.get('scheduler', {}))
        self.transform = Transform(dict.get('transform', {}))
        self.dpp = DPP({})

    def __repr__(self):
        return '\t\n'.join(['{}: {}'.format(k, v) for k, v in self.__dict__.items()])

    def dump_json(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.param, fp, indent=4)

    def to_flatten_dict(self):
        ft = {}
        for k, v in self.param.items():
            for kk, vv in v.items():
                if type(vv) in [dict, list]:
                    vv = str(vv)
                ft[f'{k}.{kk}'] = vv
        return ft

    @staticmethod
    def load_json(file_path):
        with open(file_path) as fp:
            data = json.load(fp)
        return Config(data)

    @staticmethod
    def load(file_path):
        with open(file_path) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
        return Config(data)


def get_config(name):
    return Config.load(os.path.dirname(os.path.realpath(__file__)) + '/' + name)


if __name__ == '__main__':
    args = get_config('example.yaml')
    ft = args.to_flatten_dict()
    a = 0
