import argparse
import numpy as np
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast

from dataset import dataset as datasets
from network.mofnet import MOFNet
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
        i_loss = (flow_pre - flow_gt).abs()

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

    epe = torch.sum((flow_preds[-1][:, 2:, :, :, :] - flow_gt[:, 2:, :, :, :]) ** 2, dim=2).sqrt()
    epe = epe.view(-1)[valid[:, 2:, :, :].view(-1)]

    metrics = {
        'epe': epe.float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics, NAN_flag


def train(cfg):
    loss_func = sequence_loss

    model = nn.DataParallel(MOFNet(cfg))

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    log_dir = (r"F:\record\0718-1\400000001.pt")

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint)
        print('加载 epoch {} 成功！'.format(total_steps))
    else:
        epoch = 0
        print('无保存模型，将从头开始训练！')

    PATH = r"F:\record\\" + cfg.exp_name
    # Save the checkpoints
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    writer = SummaryWriter(PATH)

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            images, flows, valids = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                images = (images + stdv * torch.randn(*images.shape).cuda()).clamp(0.0, 255.0)

            with autocast():
                output = {}
                flow_predictions = model(images, output)
                loss, metrics, NAN_flag = loss_func(flow_predictions, flows, valids, cfg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

            writer.add_scalar("LOSS/Total_Steps", metrics['loss'], total_steps)
            writer.add_scalar("EPE/Total_Steps", metrics['epe'], total_steps)

            scheduler.step()
            metrics.update(output)
            logger.push(metrics)

            if (total_steps + 1) % 1000 == 0:
                save_path = os.path.join(PATH, '{}.pt'.format(total_steps + 1))
                torch.save(model.state_dict(), save_path)
                print("Checkpoint has already been saved to", save_path)

            total_steps += 1

            if total_steps >= cfg.num_steps:
                should_keep_training = False
                break

    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', default='sintel', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--name', default='mofnet', help="name your experiment")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--max_flow', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sum_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=100000000)
    parser.add_argument('--image_size', type=list, default=[256, 256])
    parser.add_argument('--add_noise', type=bool, default=False)
    parser.add_argument('--use_smoothl1', type=bool, default=False)
    parser.add_argument('--network', type=str, default='MOFNetStack')
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
    parser.add_argument('--num_steps', type=int, default=400000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--anneal_strategy', type=str, default='cos')

    parser.add_argument('--exp_name', type=str, default='0823-1')
    args = parser.parse_args()
    print(args)


    torch.manual_seed(1234)
    np.random.seed(1234)

    train(args)
