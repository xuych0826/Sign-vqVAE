import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.vqvae import VQVAE_251
from option_vq import get_args_parser


feature_data = np.random.rand(64, 64, 1024)
feature_data = torch.tensor(feature_data).cuda()

##### ---- Exp dirs ---- #####
args = get_args_parser()
torch.manual_seed(args.seed)
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

writer = SummaryWriter("logs")

net = VQVAE_251(args,
                args.nb_code,
                args.code_dim,
                args.output_emb_width,
                args.down_t,
                args.stride_t,
                args.width,
                args.depth,
                args.dilation_growth_rate,
                args.vq_act,
                args.vq_norm)
net.train()
net.cuda()

optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
Loss = torch.nn.SmoothL1Loss()

for i in range(1000):
    pred_motion, loss_commit, perplexity = net(feature_data)
    loss_motion = Loss(pred_motion, feature_data)
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_motion
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()






