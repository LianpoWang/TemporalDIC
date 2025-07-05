import time
import os
import shutil
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger
import warnings
import scipy as sp
import numpy as np

def fetch_optimizer(model, cfg):
    """ Create the optimizer and learning rate scheduler """
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer)

    return optimizer, scheduler

def build_optimizer(model, config):
    name = config.optimizer
    lr = config.canonical_lr

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.adam_decay, eps=config.epsilon)
    elif name == "adamw":
        if hasattr(config, 'twins_lr_factor'):
            factor = config.twins_lr_factor
            print("[Decrease lr of pre-trained model by factor {}]".format(factor))
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "feat_encoder" not in n and 'context_encoder' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if ("feat_encoder" in n or 'context_encoder' in n) and p.requires_grad],
                    "lr": lr*factor,
                },
            ]
            full = [n for n, _ in model.named_parameters()]
            return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=config.adamw_decay, eps=config.epsilon)
        else:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.adamw_decay, eps=config.epsilon)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
        }
    """
    # scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.scheduler
    lr = config.canonical_lr

    if name == 'OneCycleLR':
        # scheduler = OneCycleLR(optimizer, )
        if hasattr(config, 'twins_lr_factor'):
            factor = config.twins_lr_factor
            scheduler = OneCycleLR(optimizer, [lr, lr*factor], config.num_steps+100,
                pct_start=0.05, cycle_momentum=False, anneal_strategy=config.anneal_strategy)
        else:
            scheduler = OneCycleLR(optimizer, lr, config.num_steps+100,
                pct_start=0.05, cycle_momentum=False, anneal_strategy=config.anneal_strategy)
    else:
        raise NotImplementedError()

    return scheduler


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.cfg.sum_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {}] ".format(self.total_steps + 1, self.scheduler.get_last_lr())
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        loguru_logger.info(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.cfg.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def load_vit(model):

    params_dict = np.load(r'E:\code\imagenet21k_ViT-B_32.npz', allow_pickle=False)  # pylint: disable=unexpected-keyword-arg

    model_dict = model.state_dict()

    for key_value in model_dict:
        if 'layer_dict.' in key_value:
            pretrain_key = key_value.replace('layer_dict.','')
            pretrain_key = pretrain_key.replace('.','/')
            pretrain_key = pretrain_key.replace('weight', 'kernel')
            pretrain_key = pretrain_key.replace('_left', '')
            pretrain_key = pretrain_key.replace('_right', '')
            # print(pretrain_key)
            if 'cls_token' in pretrain_key:
                model_dict[key_value]= torch.Tensor(params_dict['cls'])
            elif 'embedding/'in pretrain_key:
                if 'kernel' in pretrain_key:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key].transpose(3, 2, 0, 1))
                else:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key])
            elif 'MultiHeadDotProductAttention' in pretrain_key:
                if 'kernel' in pretrain_key:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key].reshape((768,768)))
                else:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key].reshape((768)))
            elif 'LayerNorm' in pretrain_key:
                if 'kernel' in pretrain_key:
                    new_pretrain_key=pretrain_key.replace('kernel','scale')
                    model_dict[key_value] = torch.Tensor(params_dict[new_pretrain_key])
                else:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key])
            elif 'MlpBlock' in pretrain_key:
                if 'kernel' in pretrain_key:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key].transpose(1,0))
                else:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key])
            elif 'pos_embedding' in pretrain_key:
                source_weights=params_dict[pretrain_key]
                token, grid = source_weights[0, :1], source_weights[0, 1:]
                sin = int(np.sqrt(grid.shape[0]))
                sout_x = 4
                sout_y = 4
                warnings.warn(
                    "Resizing position embeddings from " f"{sin}, {sin} to {sout_x}, {sout_y}",
                    UserWarning,
                )
                zoom = (sout_y / sin, sout_x / sin, 1)
                grid = sp.ndimage.zoom(grid.reshape(sin, sin, -1), zoom, order=1).reshape(
                    sout_x * sout_y, -1
                )
                source_weights = np.concatenate([token, grid], axis=0)[np.newaxis]
                model_dict[key_value] = torch.Tensor(source_weights)
            elif 'encoder_norm' in pretrain_key:
                if 'kernel' in pretrain_key:
                    new_pretrain_key = pretrain_key.replace('kernel', 'scale')
                    model_dict[key_value] = torch.Tensor(params_dict[new_pretrain_key])
                else:
                    model_dict[key_value] = torch.Tensor(params_dict[pretrain_key])

    model.load_state_dict(model_dict)
    print('Finished load pretrain weight!')
    return model

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


