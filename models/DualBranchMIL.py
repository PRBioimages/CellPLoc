import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from VMamba.classification.models.vmamba import VSSM


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class DualBranchMIL(nn.Module):
    def __init__(self, model_name='max-xception', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('-')[-1], pretrained=pretrained, in_chans=4)
        if 'efficient' in model_name.split('-')[-1]:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        else:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        self.model.global_pool = nn.Identity()

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt, need_feat=False):
        features = self.model(x)  # [320, 2048, 5, 5]
        pooled = nn.Flatten()(self.pooling(features))  # [320, 2048]
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])  # [20, 16, 2048]
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]  # [20, 2048]
        # print(viewed_pooled.shape)
        if need_feat:
            return self.dropout(pooled), self.last_linear(self.dropout(pooled))
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled)), \
               self.last_linear2(self.dropout(pooled))

    @property
    def net(self):
        return self.model


class DualBranchMILVMamba(nn.Module):
    def __init__(self, patch_size=16, channels=4, out_features=19, pretrained=False, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        if pretrained is True:
            self.model = VSSM(
            patch_size=4,  # 4
            in_chans=3,  # 3
            num_classes=1000,  # 1000
            depths=[2, 2, 5, 2],  # [2, 2, 5, 2]
            dims=96,  # 96
            # ===================
            ssm_d_state=1,  # 1
            ssm_ratio=2.0,  # 2.0
            ssm_rank_ratio=2.0,  # 2.0
            ssm_dt_rank="auto",  # "auto"
            ssm_act_layer="silu",  # "silu"
            ssm_conv=3,  # 3
            ssm_conv_bias=False,  # False
            ssm_drop_rate=0.0,  # 0.0
            ssm_init="v0",  # "v0"
            forward_type="v05_noz",  # "v05_noz"
            # ===================
            mlp_ratio=4.0,  # 4.0
            mlp_act_layer="gelu",  # "gelu"
            mlp_drop_rate=0.0,  # 0.0
            # ===================
            drop_path_rate=0.2,  # 0.2
            patch_norm=True,  # True
            norm_layer="ln2d",  # "ln2d"
            downsample_version="v3",  # "v3"
            patchembed_version="v2",  # "v2"
            gmlp=False,  # False
            use_checkpoint=True,  # False
            # ===================
            posembed=False,  # False
            imgsize=224,  # 224
        )
            _ckpt = torch.load("/path/to/your/VMamba_pretrained_model/vssm_tiny_0230_ckpt_epoch_262.pth",
                               map_location=torch.device("cpu"))
            self.model.load_state_dict(_ckpt["model"], strict=False)
        else:
            self.model = VSSM(
            patch_size=4,  # 4
            in_chans=channels,  # 3
            num_classes=out_features,  # 1000
            depths=[2, 2, 5, 2],  # [2, 2, 5, 2]
            dims=96,  # 96
            # ===================
            ssm_d_state=1,  # 1
            ssm_ratio=2.0,  # 2.0
            ssm_rank_ratio=2.0,  # 2.0
            ssm_dt_rank="auto",  # "auto"
            ssm_act_layer="silu",  # "silu"
            ssm_conv=3,  # 3
            ssm_conv_bias=False,  # False
            ssm_drop_rate=0.0,  # 0.0
            ssm_init="v0",  # "v0"
            forward_type="v05_noz",  # "v05_noz"
            # ===================
            mlp_ratio=4.0,  # 4.0
            mlp_act_layer="gelu",  # "gelu"
            mlp_drop_rate=0.0,  # 0.0
            # ===================
            drop_path_rate=0.2,  # 0.2
            patch_norm=True,  # True
            norm_layer="ln2d",  # "ln2d"
            downsample_version="v3",  # "v3"
            patchembed_version="v2",  # "v2"
            gmlp=False,  # False
            use_checkpoint=False,  # False
            # ===================
            posembed=False,  # False
            imgsize=224,  # 224
        )
        self.model.patch_embed[0] = nn.Conv2d(channels, 48, kernel_size=(patch_size, patch_size), stride=(2, 2),
                                              padding=(1, 1))
        n_features = self.model.classifier.head.in_features
        self.model.classifier.head = nn.Identity()

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt):
        features = self.model(x.cuda())
        pooled = features
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled)), \
               self.last_linear2(self.dropout(pooled))

    @property
    def net(self):
        return self.model


class DualBranchMILTrans(nn.Module):
    def __init__(self, model_name='transformer-swin_s3_base_224', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('-')[-1], pretrained=pretrained, in_chans=4)
        assert model_name.split('-')[0] in 'transformer', print('using model not transformer model')
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        if 'swin' not in model_name:
            self.model.global_pool = nn.Identity()

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt):
        features = self.model(x)
        pooled = features
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled)), \
               self.last_linear2(self.dropout(pooled))

    @property
    def net(self):
        return self.model

