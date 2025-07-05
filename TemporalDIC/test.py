import argparse

import torch.nn as nn

from dataset import dataset as datasets
from network.temporaldicnet import TemporalDICNet
from utils.some_func import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    B, N, _, H, W = flow_gt.shape

    NAN_flag = False

    mag = torch.sum(flow_gt ** 2, dim=2).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        flow_pre = flow_preds[i]
        i_loss = (flow_pre[:,2:,:,:,:] - flow_gt[:,2:,:,:,:]).abs()

        if torch.isnan(i_loss).any():
            NAN_flag = True

        _valid = valid[:, :, None]
        if cfg.filter_epe:
            loss_mag = torch.sum(i_loss ** 2, dim=2).sqrt()
            mask = loss_mag > 1000
            if torch.any(mask):
                print("[Found extrem epe. Filtered out. Max is {}. Ratio is {}]".format(torch.max(loss_mag),
                                                                                        torch.mean(mask.float())))
                _valid = _valid & (~mask[:, :, None])

        flow_loss += i_weight * (_valid * i_loss).mean()

    epe = torch.sum((flow_preds[-1][:,2:,:,:,:] - flow_gt[:,2:,:,:,:]) ** 2, dim=2).sqrt()
    epe = epe.view(-1)[valid[:,2:,:,:].view(-1)]
    metrics = {
        'epe': epe.float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics, NAN_flag


def test(cfg):
    loss_func = sequence_loss
    model = nn.DataParallel(TemporalDICNet(cfg))
    model.cuda()

    test_loader = datasets.fetch_dataloader(cfg)

    checkpoint = r"./weight/checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    total_loss = 0
    total_epe = 0

    for i_batch, data_blob in enumerate(test_loader):
        images, flows, valids = [x.cuda() for x in data_blob]
        output = {}
        flow_predictions = model(images, output)
        loss, metrics, NAN_flag = loss_func(flow_predictions, flows, valids, cfg)
        total_loss = total_loss + loss.item()
        total_epe = total_epe + metrics["epe"]

    avg_epe = total_epe / (i_batch + 1)
    avg_loss = total_loss / (i_batch + 1)
    print(" Average EPE:", avg_epe, "Average LOSS:", avg_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', default='sintel', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--name', default='temporaldicnet', help="name your experiment")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--max_flow', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sum_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=100000000)
    parser.add_argument('--image_size', type=list, default=[256, 256])
    parser.add_argument('--add_noise', type=bool, default=False)
    parser.add_argument('--use_smoothl1', type=bool, default=False)
    # parser.add_argument('--critical_params', type=str, default=[])
    parser.add_argument('--network', type=str, default='TemporalDICNet')
    parser.add_argument('--restore_ckpt', type=str, default='PATH_TO_FINAL/final')

    parser.add_argument('--mixed_precision', type=bool, default=True)
    parser.add_argument('--input_frames', type=int, default=5)
    parser.add_argument('--filter_epe', type=bool, default=False)

    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--Tfusion', type=str, default='stack')
    parser.add_argument('--cnet', type=str, default='twins')
    parser.add_argument('--fnet', type=str, default='twins')
    parser.add_argument('--down_ratio', type=int, default=8)
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--corr_fn', type=str, default='default')
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--context_3D', type=bool, default=False)
    parser.add_argument('--decoder_depth', type=int, default=1)

    ### TRAINER
    parser.add_argument('--scheduler', type=str, default='OneCycleLR')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--canonical_lr', type=float, default=1e-4)
    parser.add_argument('--adamw_decay', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--anneal_strategy', type=str, default='cos')

    parser.add_argument('--exp_name', type=str, default='0326-1')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(1234)
    np.random.seed(1234)

    test(args)
