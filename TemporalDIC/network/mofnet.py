import torch.nn.functional as F
import numpy as np
import torch
from .twins import Twins
from torch import nn, einsum
from einops import rearrange

from utils.some_func import bilinear_sampler, coords_grid
autocast = torch.cuda.amp.autocast

def load_pretrain(model):
    ckpt = torch.load(r"./weight/twins.pth")
    #
    for key, value in ckpt.items():
        if key == "patch_embeds.0.proj.weight":
            value = value[:, 0, :, :]
            value = torch.unsqueeze(value, dim=1)
            ckpt[key] = value

    model.load_state_dict(ckpt)
    print('Finished load pretrain weight!')
    return model


class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True, del_layers=True):
        super().__init__()
        self.svt = Twins(img_size=256, patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32],
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1])
        self.svt = load_pretrain(self.svt)

        if del_layers:
            del self.svt.head
            del self.svt.patch_embeds[2]
            del self.svt.patch_embeds[2]
            del self.svt.blocks[2]
            del self.svt.blocks[2]
            del self.svt.pos_block[2]
            del self.svt.pos_block[2]

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == 0:
                x_16 = x.clone()
            if i == layer - 1:
                break

        return x

    def extract_ml_features(self, x, data=None, layer=2):
        res = []
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            x, size = embed(x)
            if i == layer - 1:
                x1 = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer - 1:
                break

        return x1, x

    def compute_params(self):
        num = 0

        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num += np.prod(param.size())
            for param in blocks.parameters():
                num += np.prod(param.size())
            for param in pos_blk.parameters():
                num += np.prod(param.size())
            for param in drop.parameters():
                num += np.prod(param.size())
            if i == 1:
                break
        return num


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class velocity_update_block(nn.Module):
    def __init__(self, C_in=43 + 128 + 43, C_out=43, C_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(C_in, C_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hidden, C_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hidden, C_out, 3, padding=1),
        )

    def forward(self, x):
        return self.mlp(x)


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius * 2 + 1) ** 2 * args.cost_heads_num * args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 128, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, 48, 1, 1))

        self.convf1_ = nn.Conv2d(4, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64 + 192 + 48 * 3, 128 - 4 + 48, k_conv=args.k_conv)

        self.velocity_update_block = velocity_update_block()
        self.conv_motion = nn.Conv2d(512, 48, kernel_size=3, padding=1)
        self.conv_motion1 = nn.Conv2d(768, 48, kernel_size=3, padding=1)

    def sample_flo_feat(self, flow, feat):

        sampled_feat = bilinear_sampler(feat.float(), flow.permute(0, 2, 3, 1))
        return sampled_feat

    def forward(self, motion_hidden_state, forward_flow, backward_flow, coords0, forward_corr, backward_corr, bs):

        BN, _, H, W = forward_flow.shape
        N = BN // bs

        # motion_hidden_state = None
        if motion_hidden_state is None:
            # print("initialized as None")
            motion_hidden_state = self.init_hidden_state.repeat(bs, N, 1, H, W)
        else:
            # print("later iterations")
            if motion_hidden_state.shape[1] == 512:
                motion_hidden_state = self.conv_motion(motion_hidden_state)
            elif motion_hidden_state.shape[1] == 768:
                motion_hidden_state = self.conv_motion1(motion_hidden_state)
            motion_hidden_state = motion_hidden_state.reshape(bs, N, -1, H, W)

        forward_loc = forward_flow + coords0
        backward_loc = backward_flow + coords0

        forward_motion_hidden_state = torch.cat(
            [motion_hidden_state[:, 1:, ...], torch.zeros(bs, 1, 48, H, W).to(motion_hidden_state.device)],
            dim=1).reshape(BN, -1, H, W)
        forward_motion_hidden_state = self.sample_flo_feat(forward_loc, forward_motion_hidden_state)
        backward_motion_hidden_state = torch.cat(
            [torch.zeros(bs, 1, 48, H, W).to(motion_hidden_state.device), motion_hidden_state[:, :N - 1, ...]],
            dim=1).reshape(BN, -1, H, W)
        backward_motion_hidden_state = self.sample_flo_feat(backward_loc, backward_motion_hidden_state)

        forward_cor = self.convc1(forward_corr)
        backward_cor = self.convc1(backward_corr)
        cor = F.gelu(torch.cat([forward_cor, backward_cor], dim=1))
        cor = self.convc2(cor)

        flow = torch.cat([forward_flow, backward_flow], dim=1)
        flo = self.convf1_(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo, forward_motion_hidden_state, backward_motion_hidden_state,
                             motion_hidden_state.reshape(BN, -1, H, W)], dim=1)
        out = self.conv(cor_flo)

        out, motion_hidden_state = torch.split(out, [124, 48], dim=1)

        return torch.cat([out, flow], dim=1), motion_hidden_state


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=384):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SKUpdateBlock6_Deep_nopoolres_AllDecoder2(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args

        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]

        hidden_dim_ratio = 256 // args.feat_dim

        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        self.gru = PCBlock4_Deep_nopool_res(128 + hidden_dim + hidden_dim + 128, 128 // hidden_dim_ratio,
                                            k_conv=args.PCUpdater_conv)
        # self.gru = ConvGRU()
        self.flow_head = PCBlock4_Deep_nopool_res(128 // hidden_dim_ratio, 4, k_conv=args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128 // hidden_dim_ratio, 256 // hidden_dim_ratio, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // hidden_dim_ratio, args.down_ratio ** 2 * 9 * 2, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, motion_hidden_state, inp, forward_corr, backward_corr, forward_flow, backward_flow, coords0,
                attention, bs):
        motion_features, motion_hidden_state = self.encoder(motion_hidden_state, forward_flow, backward_flow, coords0,
                                                            forward_corr, backward_corr, bs=bs)

        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))
        # net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 100.0 * self.mask(net)
        return net, motion_hidden_state, mask, delta_flow


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(
            self,
            *,
            args,
            dim,
            max_pos_size=100,
            heads=4,
            dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        return attn


class Aggregate(nn.Module):
    def __init__(
            self,
            args,
            dim,
            heads=4,
            dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class MOFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = hdim = self.cfg.feat_dim // 2
        self.context_dim = cdim = self.cfg.feat_dim // 2
        self.conv_cnet = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convgruf = ConvGRU(256, 256)
        self.convgrub = ConvGRU(256, 256)
        self.conls = nn.Conv2d(256, 512, 1)
        self.conv_backcorr = nn.Conv2d(324, 512, 1)
        # self.up_backcorr =
        cfg.corr_radius = 4

        # feature network, context network, and update block
        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
        hidden_dim_ratio = 256 // cfg.feat_dim

        if self.cfg.Tfusion == 'stack':
            print("[Using stack.]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg,
                                                                          hidden_dim=128 // hidden_dim_ratio)
        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        gma_down_ratio = 256 // cfg.feat_dim

        self.att = Attention(args=self.cfg, dim=128 // hidden_dim_ratio, heads=1, max_pos_size=160,
                             dim_head=128 // hidden_dim_ratio)

        if self.cfg.context_3D:
            print("[Using 3D Conv on context feature.]")
            self.context_3D = nn.Sequential(
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
            )

    def initialize_flow(self, img, bs, down_ratio):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(bs, H // down_ratio, W // down_ratio).to(img.device)
        coords1 = coords_grid(bs, H // down_ratio, W // down_ratio).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def upsample_flow_4x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)

    def upsample_flow_2x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 2 * H, 2 * W)

    def forward(self, images, data={}, flow_init=None):

        down_ratio = self.cfg.down_ratio

        B, N, _, H, W = images.shape

        images = 2 * (images / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.cfg.mixed_precision):
            # fmaps = self.fnet(images.reshape(B * N, 3, H, W)).reshape(B, N, -1, H // down_ratio, W // down_ratio)
            fmaps = self.fnet(images.reshape(B * N, 1, H, W)).reshape(B, N, -1, H // down_ratio, W // down_ratio)
        fmaps = fmaps.float()

        if self.cfg.corr_fn == "default":
            corr_fn = CorrBlock
        elif self.cfg.corr_fn == "efficient":
            corr_fn = AlternateCorrBlock

        new_input = torch.cat(
            (fmaps[:, 0, ...].unsqueeze(1), fmaps[:, 0, ...].unsqueeze(1), fmaps[:, 0, ...].unsqueeze(1)), dim=1)

        forward_corr_fn = corr_fn(new_input.reshape(B * (N - 2), -1, H // down_ratio, W // down_ratio),
                                  fmaps[:, 2:N, ...].reshape(B * (N - 2), -1, H // down_ratio, W // down_ratio),
                                  num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        backward_corr_fn = corr_fn(new_input.reshape(B * (N - 2), -1, H // down_ratio, W // down_ratio),
                                   fmaps[:, 1:N - 1, ...].reshape(B * (N - 2), -1, H // down_ratio, W // down_ratio),
                                   num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)



        with autocast(enabled=self.cfg.mixed_precision):

            cnet = self.cnet(images[:, 1:N - 1, ...].reshape(B * (N - 2), 1, H, W))
            cnet0 = self.cnet(images[:, 0:1, ...].expand(-1, 3, -1, -1, -1).reshape(B * (N - 2), 1, H, W))
            cnet1 = self.cnet(images[:, 1:N - 1, ...].reshape(B * (N - 2), 1, H, W))
            # 改动1
            cnet2 = self.cnet(images[:, 2:N, ...].reshape(B * (N - 2), 1, H, W))
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)

        forward_coords1, forward_coords0 = self.initialize_flow(images[:, 0, ...], bs=B * (N - 2),
                                                                down_ratio=down_ratio)
        backward_coords1, backward_coords0 = self.initialize_flow(images[:, 0, ...], bs=B * (N - 2),
                                                                  down_ratio=down_ratio)

        flow_predictions = []

        motion_hidden_state = None



        for itr in range(self.cfg.decoder_depth):

            forward_coords1 = forward_coords1.detach()
            backward_coords1 = backward_coords1.detach()

            forward_corr = forward_corr_fn(forward_coords1)
            backward_corr = backward_corr_fn(backward_coords1)
            motion_hidden_state = self.conv_backcorr(backward_corr)

            forward_flow = forward_coords1 - forward_coords0
            backward_flow = backward_coords1 - backward_coords0

            with autocast(enabled=self.cfg.mixed_precision):
                net, motion_hidden_state, up_mask, delta_flow = self.update_block(net, motion_hidden_state, inp,
                                                                                  forward_corr, backward_corr,
                                                                                  forward_flow, backward_flow,
                                                                                  forward_coords0, attention, bs=B)

            forward_up_mask, backward_up_mask = torch.split(up_mask, [down_ratio ** 2 * 9, down_ratio ** 2 * 9], dim=1)

            forward_coords1 = forward_coords1 + delta_flow[:, 0:2, ...]
            backward_coords1 = backward_coords1 + delta_flow[:, 2:4, ...]

            # upsample predictions
            if down_ratio == 4:
                forward_flow_up = self.upsample_flow_4x(forward_coords1 - forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow_4x(backward_coords1 - backward_coords0, backward_up_mask)
            elif down_ratio == 2:
                forward_flow_up = self.upsample_flow_2x(forward_coords1 - forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow_2x(backward_coords1 - backward_coords0, backward_up_mask)
            elif down_ratio == 8:
                forward_flow_up = self.upsample_flow(forward_coords1 - forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow(backward_coords1 - backward_coords0, backward_up_mask)

            flow_predictions.append(forward_flow_up.reshape(B, N - 2, 2, H, W))
            return flow_predictions