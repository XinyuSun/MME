import math
import copy
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange, repeat
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from m3video.feat import VideoHOGFeat
import numpy as np


def shape_2_pos(traj_shape, traj_len=8):
    b,c,t,h,w = traj_shape.shape
    init_pos = np.stack(np.meshgrid(np.arange(w*8),np.arange(h*8)), axis=0)
    init_pos = init_pos[:,4::8,4::8] # c h w
    traj_pos = torch.zeros_like(traj_shape)
    for i in range(t//traj_len):
        traj_pos[:,:,0+i*traj_len,...] = torch.from_numpy(init_pos)
        for j in range(traj_len-1):
            t = j + i*traj_len
            traj_pos[:,:,t+1,...] = traj_shape[:,:,t,...] + traj_pos[:,:,t,...]
            # traj_pos[t+1,...] = traj_pos[t,...]
            traj_pos[:,0,t+1,...] = torch.clamp(traj_pos[:,0,t+1,...],0,w*8-1)
            traj_pos[:,1,t+1,...] = torch.clamp(traj_pos[:,1,t+1,...],0,h*8-1)
        
    return traj_pos.long()


def get_hog_meshgrid(shape):
    return torch.meshgrid([torch.arange(s) for s in shape])


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, target_feature='idt', feat_args=None, 
                    shuffle_frame=False, adaptive_mask=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction='none')
    loss_func_traj = nn.MSELoss(reduction='none')

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if target_feature in ['idt', 'hybrid']:
            videos, trajs, bool_masked_pos = batch
            trajs = trajs.to(device, non_blocking=True)
        else:
            videos, bool_masked_pos = batch
        
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
        if shuffle_frame:
            rand_inds = np.arange(0, 16)
            np.random.shuffle(rand_inds)
            videos = videos[:,:,rand_inds,:,:]

        with torch.no_grad():
            # calculate the predict label
            # mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            # std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            # unnorm_videos = videos * std + mean  # in [0, 1]
            
            if target_feature in ['hog', 'idt']:
                patch_size_3d = copy.deepcopy(feat_args['patch_size'])
                if feat_args['traj_varient'] == 'VolumeHOG':
                    patch_size_3d[0] = 1
                hog_extractor = VideoHOGFeat(
                    nbins=feat_args['nbins'], 
                    cell_size=feat_args['cell_size'], 
                    block_size=feat_args['block_size'], 
                    patch_size=patch_size_3d, 
                    norm=feat_args['norm_type'], 
                    channel=feat_args['channel'], 
                    device='cuda'
                )
                # video_hog = hog_extractor(unnorm_videos)
                video_hog = hog_extractor(videos)
                
                if feat_args['traj_varient'] == 'VolumeHOG':
                    hog_label = rearrange(video_hog, 'b c (t t2) (h h2) (w w2) -> b (t h w) (c t2 h2 w2)', h2=2, w2=2, t2=feat_args['traj_len'])
                    videos = videos[:,:,::feat_args['traj_len']//2,...]
                else:
                    hog_label = rearrange(video_hog, 'b c t (h h2) (w w2) -> b (t h w) (c h2 w2)', h2=2, w2=2)
                
                B, _, C = hog_label.shape
                hog_label = hog_label[bool_masked_pos].reshape(B, -1, C)

            if target_feature in ['idt']:
                traj_dim = feat_args['traj_len'] * 2 * 4
                traj_shape = trajs[:,:,:traj_dim,...] #TODO: adjustable length
                traj_shape = traj_shape / 20
                traj_type = trajs[:,0,traj_dim:,...].long()
                
                if feat_args['traj_varient'] == 'VolumeHOG':
                    traj_pos = shape_2_pos(traj_shape, feat_args['traj_len'])
                    mx = traj_pos[:,0,...] // 8
                    my = traj_pos[:,1,...] // 8
                    b,c,t,h,w = video_hog.shape
                    mb, mt, _, _ = get_hog_meshgrid((b,t,h,w))
                    hog_label = video_hog[mb, :, mt, my, mx]
                    hog_label = rearrange(hog_label, 'b (t t2) (h h2) (w w2) c -> b (t h w) (c t2 h2 w2)', t2=feat_args['traj_len'], h2=2, w2=2)
                    
                    B, _, C = hog_label.shape
                    hog_label = hog_label[bool_masked_pos].reshape(B, -1, C)
                
                if feat_args['traj_norm'] == 'PatchStd':
                    traj_shape = rearrange(traj_shape, 'b c (t t2) (h h2) (w w2) -> b (t h w) (c t2) (h2 w2)', t2=feat_args['traj_len'], h2=2, w2=2)
                    traj_shape = (traj_shape - traj_shape.mean(dim=-1, keepdim=True)
                                  ) / (traj_shape.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    traj_shape = rearrange(traj_shape, 'b n c1 c2 -> b n (c1 c2)')
                else:
                    traj_shape = rearrange(traj_shape, 'b c (t t2) (h h2) (w w2) -> b (t h w) (c t2 h2 w2)', t2=feat_args['traj_len'], h2=2, w2=2)
                
                
                traj_type = rearrange(traj_type, 'b t (h h2) (w w2) -> (h2 w2) b (t h w)', h2=2, w2=2).sum(dim=0)
                traj_type = (traj_type >= 1).long()
                
                B, _, C = traj_shape.shape

                traj_label = traj_shape[bool_masked_pos].reshape(B, -1, C)
                traj_mask = traj_type[bool_masked_pos].reshape(B, -1)
                if feat_args['loss_mask'] == False:
                    traj_mask = torch.ones_like(traj_mask)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)

            if target_feature == 'idt':
                loss_1 = loss_func_traj(input=outputs[1], target=traj_label).sum(-1) * traj_mask * 32 * feat_args['traj_loss_scale'] / traj_label.shape[-1]
                loss_2 = loss_func(input=outputs[0], target=hog_label)
                
                if feat_args['traj_varient'] == 'VolumeHOG':
                    loss_2 = rearrange(loss_2, 'b n (c t2 h2 w2) -> b n (c h2 w2) t2', t2=feat_args['traj_len'], h2=2, w2=2)
                    loss_2 *= torch.from_numpy(feat_args['alpha']).float().to(loss_2.device)
                    loss_2 = loss_2.mean(-1)
                else:
                    loss_2 = loss_2.sum(-1)
                
                loss_1 = loss_1.sum() / bool_masked_pos.sum()
                loss_2 = loss_2.sum() / bool_masked_pos.sum()
                
                loss = (loss_1 + loss_2) / len(outputs)
            elif target_feature == 'hog':
                loss_2 = loss_func(input=outputs, target=hog_label).sum() / bool_masked_pos.sum()
                loss_1 = torch.Tensor([0])
                
                loss = loss_2
            else:
                raise NotImplementedError(
                    "only support idt as target feature"
                )
                
        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_1=loss_1)
        metric_logger.update(loss_2=loss_2)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_1=loss_1, head="loss")
            log_writer.update(loss_2=loss_2, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
