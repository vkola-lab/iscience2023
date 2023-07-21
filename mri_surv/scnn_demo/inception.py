from collections import namedtuple
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Callable, Any, Optional, Tuple, List


__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


class Inception3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None
    ) -> None:
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv3d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv3d_1a_3x3 = conv_block(1, 32, kernel_size=3, stride=2)
        self.Conv3d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv3d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.Conv3d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv3d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, 'stddev') else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.out = nn.functional.normalize

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 1 x 121 x 145 x 121
        x = self.Conv3d_1a_3x3(x)
        # N x 32 x 60 x 72 x 60
        x = self.Conv3d_2a_3x3(x)
        # N x 32 x 58 x 70 x 58
        x = self.Conv3d_2b_3x3(x)
        # N x 64 x 58 x 70 x 58
        x = self.maxpool1(x)
        # N x 64 x 28 x 34 x 28
        x = self.Conv3d_3b_1x1(x)
        # N x 80 x 28 x 34 x 28
        x = self.Conv3d_4a_3x3(x)
        # N x 192 x 26 x 32 x 26
        x = self.maxpool2(x)
        # N x 192 x 12 x 15 x 12
        x = self.Mixed_5b(x)
        # N x 256 x 12 x 15 x 12
        x = self.Mixed_5c(x)
        # N x 288 x 12 x 15 x 12
        x = self.Mixed_5d(x)
        # N x 288 x 12 x 15 x 12
        x = self.Mixed_6a(x)
        # N x 768 x 5 x 7 x 5
        x = self.Mixed_6b(x)
        # N x 768 x 5 x 7 x 5
        x = self.Mixed_6c(x)
        # N x 768 x 5 x 7 x 5
        x = self.Mixed_6d(x)
        # N x 768 x 5 x 7 x 5
        x = self.Mixed_6e(x)
        # N x 768 x 5 x 7 x 5
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
                # N x 3
        x = self.Mixed_7a(x)
        # N x 1280 x 3 x 5 x 3
        x = self.Mixed_7b(x)
        # N x 1280 x 3 x 5 x 3
        x = self.Mixed_7c(x)
        # N x 1280 x 3 x 5 x 3
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 3 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(self.out(x), self.out(aux))
        else:
            return self.out(x)  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(self.out(x), self.out(aux))
        else:
            return self.eager_outputs(x, aux)


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool3d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1, 1), padding=(3, 0, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        #potential change!!!

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=1) #was 2

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1, 1), padding=(3, 0, 0)) #consider more!
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=1) #was 2

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool3d(x, kernel_size=3, stride=1) #was 2
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3, 1), padding=(0, 1, 0)) #consider more layers!
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=(3, 5, 3))
        # self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 5 x 7 x 5
        x = F.avg_pool3d(x, kernel_size=3, stride=1)
        # x = F.avg_pool3d(x, kernel_size=5, stride=3)
        # N x 768 x 3 x 5 x 3
        x = self.conv0(x)
        # N x 768 x 3 x 5 x 3
        x = self.conv1(x)
        # N x 768 x 1 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        # N x 768 x 1 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 3
        return x


class BasicConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


import sys, os, glob
import numpy as np
from utils import read_json
from dataloader import AE_Cox_Data
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy import interpolate
from sksurv.metrics import concordance_index_censored, integrated_brier_score

def sur_loss(preds, obss, hits, bins=torch.Tensor([[0, 24, 48, 108]])):
    if torch.cuda.is_available():
        bins = bins.cuda()
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(-1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)), bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)
    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]), torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

class Wrapper:
    def __init__(self, config, exp_idx, num_fold, seed):
        self.gpu = 1
        self.config = config
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.seed = seed
        torch.manual_seed(seed)
        self.prepare_dataloader()

        vector_len = 3
        self.targets = list(range(vector_len))

        self.model = Inception3(num_classes=vector_len).cuda()
        # self.model = Inception3(drop_rate=drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()

        if self.gpu != 1:
            self.model = self.model.cpu()
        self.criterion = sur_loss
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['lr'], weight_decay=0.01)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.config['lr'], weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.config['name'], exp_idx)

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def check(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])
        print('loading trained model...')
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])

    def load(self, dir, fixed=False):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        del st['l2.weight']
        del st['l2.bias']
        self.model.load_state_dict(st, strict=False)
        if fixed:
            ps = []
            for n, p in self.model.named_parameters():
                # if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                if n == 'l2.weight' or n == 'l2.bias' :
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def prepare_dataloader(self):
        self.train_data = AE_Cox_Data(self.config['Data_dir'], self.exp_idx, stage='train', seed=self.seed, name=self.config['name'], fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = self.train_data.get_sample_weights()
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=True)
        if self.gpu != 1:
            self.train_dataloader = DataLoader(self.train_data, batch_size=len(train_data), shuffle=True)

        self.valid_data = AE_Cox_Data(self.config['Data_dir'], self.exp_idx, stage='valid', seed=self.seed, name=self.config['name'], fold=self.num_fold)
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=1, shuffle=False)

        self.test_data  = AE_Cox_Data(self.config['Data_dir'], self.exp_idx, stage='test' , seed=self.seed, name=self.config['name'], fold=self.num_fold)
        self.test_dataloader  = DataLoader(self.test_data, batch_size=1, shuffle=False)

        self.all_data  = AE_Cox_Data(self.config['Data_dir'], self.exp_idx, stage='all' , seed=self.seed, name=self.config['name'], fold=self.num_fold)
        self.all_dataloader  = DataLoader(self.all_data, batch_size=len(self.all_data))

        self.ext_data = AE_Cox_Data(self.config['Data_dir_ext'], self.exp_idx, stage='all', seed=self.seed, name=self.config['name'], fold=self.num_fold, external=True)
        # self.ext_dataloader  = DataLoader(self.ext_data, batch_size=1, shuffle=False)

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()
            self.model.zero_grad()
            preds = self.model(inputs)
            loss1 = self.criterion(preds[0], obss, hits)
            loss2 = self.criterion(preds[1], obss, hits)
            loss = loss1 + 0.4*loss2


            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            # clip = 1
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
        return loss

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        cis, bss = self.concord(load=False)
        ci_v, ci_t = cis
        print('initial CI:', 'CI_test vs CI_valid: %.3f : %.3f' % (ci_t[0], ci_v[0]))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):
            train_loss = self.train_model_epoch()

            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                cis, bss = self.concord(load=False)
                ci_t, ci_v = cis

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch validation loss [surv] ='.format(self.epoch), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| CI (test vs valid) = %.3f : %.3f' % (ci_t[0], ci_v[0]), '|| time(s) =', start.elapsed_time(end)//1000)
                # , '|| BS (test vs valid) = %.3f : %.3f' %(bss[0], bss[1]))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.config['name'], self.optimal_epoch))
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        if self.config['metric'] == 'concord':
            cis, bss = self.concord(load=False)
            ci_t, ci_v = cis
            return -ci_t[0]
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                preds_all += [self.model(data).cpu().numpy().squeeze()]

                obss_all += [obss]
                hits_all += [hits]

            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)

            if self.gpu != 1:
                loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))

            else:
                loss = self.criterion(torch.tensor(preds_all).cuda(), torch.tensor(obss_all).cuda(), torch.tensor(hits_all).cuda())

        return loss

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.config['name'], self.optimal_epoch))

    def concord(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.valid_dataloader, self.test_dataloader]
            if all:
                train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
                ext_dl = DataLoader(self.ext_data, batch_size=1, shuffle=False)
                dls = [train_dl, self.valid_dataloader, self.test_dataloader, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
            for dl in dls:
                preds_all, preds_all_brier = [], []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    # preds = nn.functional.normalize(self.model(data.cuda())).cpu()
                    # print(preds)
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                    concordance_time = 24
                    preds = interp(concordance_time) # approx @ 24
                    preds_all_brier += list(interp(bins))
                    preds_all += [preds[0]]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0] == 1]
                # print(preds_all[:10])
                preds_all = [-p for p in preds_all]
                # print(torch.sum(data!=data))
                # print(preds_all)
                # print(hits_all)
                # print(obss_all)
                c_index = concordance_index_censored(hits_all, obss_all, preds_all)
                test_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
                bins[-1] = max(obss_all)-1
                bins[0] = min(obss_all)
                # print('train', len(obss_all), obss_all)
                # sys.exit()
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, bins)
                cis += [c_index]
                bss += [bs]
            # self.c_index.append(c_index)
        return cis, bss


torch.use_deterministic_algorithms(True)

def inc_main(repeat, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    c_te, c_tr, c_ex = [], [], []
    b_te, b_tr, b_ex = [], [], []
    for exp_idx in range(repeat):
        inc = Wrapper(config = config, exp_idx = exp_idx, num_fold = repeat, seed = 1000*exp_idx)
        inc.train(epochs = config['train_epochs'])
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=False)
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=True)
        # cnn.check('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0))
        out = inc.concord(all=True) # [train_dl, self.valid_dataloader, self.test_dataloader, ext_dl]
        c_tr += [out[0][0][0]]
        c_te += [out[0][2][0]]
        c_ex += [out[0][3][0]]
        b_tr += [out[1][0]]
        b_te += [out[1][2]]
        b_ex += [out[1][3]]
        # cnn.shap()
        # cnn.predict_plot()
        # cnn.predict_plot_general()
        # cnn.predict_plot_scatter()
    print('CI test: %.3f+-%.3f' % (np.mean(c_te), np.std(c_te)))
    print('CI train: %.3f+-%.3f' % (np.mean(c_tr), np.std(c_tr)))
    print('CI external: %.3f+-%.3f' % (np.mean(c_ex), np.std(c_ex)))
    print('BS test: %.3f+-%.3f' % (np.mean(b_te), np.std(b_te)))
    print('BS train: %.3f+-%.3f' % (np.mean(b_tr), np.std(b_tr)))
    print('BS external: %.3f+-%.3f' % (np.mean(b_ex), np.std(b_ex)))

def main():
    config = read_json('./inception_config.json')['inception']
    print('Training inception classifiers ...')
    print('-'*100)
    inc_main(3, config, Wrapper)
    print('-'*100)

if __name__ == "__main__":
    main()
