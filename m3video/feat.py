import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch import Tensor
import math
import copy
import decord
import av
from skimage import draw
from skimage.feature import hog as ski_hog
from einops import rearrange, reduce, repeat
from pathlib import Path
import compress_pickle

decord.bridge.set_bridge('torch')

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='replicate')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class FeatGenerator:
    def __init__(self, feat_type, feat_args, device):
        self.feat_type = feat_type
        self.device = device
        self.patch_size = feat_args['patch_size']
        self.shuffle = feat_args['shuffle']
        self.video_hog = VideoHOGFeat(
            nbins=feat_args['nbins'],
            cell_size=feat_args['cell_size'],
            block_size=feat_args['block_size'],
            patch_size=feat_args['patch_size'],
            norm=feat_args['norm_type'],
            channel=feat_args['channel'],
            device=self.device
        )
        if feat_type in ['bt', 'idt']:
            self.traj_len = feat_args['traj_len']
            self.traj_max_len = feat_args['traj_max_len']
            self.traj_num = feat_args['traj_num']
            self.num_visible = feat_args['num_visible']
            self.clip_len = feat_args['cover_frames']
            self.shape_dim = feat_args['shape_dim']
            self.threshold = feat_args['threshold']
            self.traj_varient = feat_args['traj_varient']
            self.longer_traj = feat_args['longer_traj']
            self.traj_norm = feat_args['traj_norm']
            self.unnorm_video = feat_args['unnorm_video']
            self.flow_tracker = SparseTracker(
                traj_len=self.traj_len,
                clip_len=self.clip_len,
                device=self.device,
            )
            
    def _expand_feat(self, feat, len):
        # feat: b c t h w
        b,c,t,h,w = feat.shape
        inds = []
        for i in range(t):
            if i < t: # TODO: currently support len = 1
                inds = inds + [i, i+1]
                
        inds[-1] = t - 1
        
        feat = rearrange(feat[:,:,inds,...], 'b c (t1 t2) h w -> b (c t2) t1 h w', t2=len)
        return feat

    def _unnorm_video(self, videos):
        mean = torch.as_tensor((0.485, 0.456, 0.406)).to(videos.device)[None, :, None, None, None]
        std = torch.as_tensor((0.229, 0.224, 0.225)).to(videos.device)[None, :, None, None, None]
        unnorm_videos = videos * std + mean  # in [0, 1]
        return unnorm_videos

    def __call__(self, batch_clip:Tensor, batch_mv:Tensor=None, margin:int=None):
        if self.feat_type == 'hog':
            if self.unnorm_video:
                hog_feat = self.video_hog(self._unnorm_video(batch_clip))
            else:
                hog_feat = self.video_hog(batch_clip)
                
            if self.patch_size[1] > 8:
                scale = self.patch_size[1] // 8
                hog_feat = rearrange(hog_feat, 'b c t (h1 h2) (w1 w2) -> b (c h2 w2) t h1 w1', h2=scale, w2=scale)

            if self.shuffle:
                tt = self.patch_size[0]
                temporal_inds = np.arange(batch_clip.shape[2])
                np.random.shuffle(temporal_inds)
                batch_clip = batch_clip[:,:,temporal_inds,...]
                    
                hog_feat = hog_feat[:,:,temporal_inds[0::2]//tt,...]

            return (batch_clip, [[hog_feat], None])
        
        elif self.feat_type == 'bt':
            traj_feat, traj_pos, traj_type = self.flow_tracker(batch_mv, dense_flow=False)
            
            hog_feat = self.video_hog(batch_clip) #HOG
            # traj_feat = (traj_feat + 1) / 2 # norm from [-1, 1] to [0, 1]
            # traj_type[traj_feat.float().norm(dim=4).sum(dim=1) < self.threshold] = 0
            hog_feat /= 20
            target_feat = [hog_feat, traj_feat]
            
            return (batch_clip, [target_feat, traj_type])
        
        elif self.feat_type == 'idt':
            hog_feat = self.video_hog(batch_clip) #HOG
            total_traj_len = self.traj_len * self.traj_num
            traj_shape = batch_mv[:,:,:total_traj_len,...]
            traj_type = batch_mv[:,0,total_traj_len:,...].long()
            mask_gap = self.clip_len // self.num_visible
            
            if self.traj_varient == 'VolumeHOG':
                traj_pos = self.flow_tracker.shape_2_pos(traj_shape, self.traj_max_len)
                hog_feat = self.flow_tracker.fetch_hog(hog_feat, traj_pos)
                # b t h w c, caution!
                hog_feat = rearrange(hog_feat, 'b (t1 t2) h w c -> b (c t2) t1 h w', t1=self.traj_num)
                     
                rgb_inds = torch.linspace(0, self.clip_len-mask_gap, self.num_visible, dtype=torch.long)
                batch_clip = batch_clip[:,:,rgb_inds].contiguous()
            elif self.traj_varient == 'AccuHOG':
                traj_pos = self.flow_tracker.shape_2_pos(traj_shape, self.traj_max_len)
                fetch_inds = torch.linspace(0, self.clip_len-mask_gap, self.num_visible, dtype=torch.long)
                fetch_inds = torch.cat((fetch_inds, fetch_inds + self.traj_len - 1))
                fetch_inds,_ = torch.sort(fetch_inds)
                hog_feat = self.flow_tracker.fetch_hog(hog_feat, traj_pos[:,:,fetch_inds,...])
                hog_feat = rearrange(hog_feat, 'b (t1 t2) h w c -> b (c t2) t1 h w', t1=self.traj_num)
                
                rgb_inds = torch.linspace(0, 2*self.num_visible-1, self.num_visible, dtype=torch.long)
                batch_clip = batch_clip[:,:,rgb_inds].contiguous()
            elif self.traj_varient == 'AccuiDT':
                traj_pos = self.flow_tracker.shape_2_pos(traj_shape, self.traj_max_len)
                traj_pos = rearrange(traj_pos, 'b c (t1 t2) h w -> b c t1 t2 h w', t1=self.traj_num)
                traj_shape = traj_pos[:,:,:,-1,...] - traj_pos[:,:,:,0,...]
                traj_shape = traj_shape.float()
            
            traj_shape = rearrange(traj_shape, 'b c (t1 t2) (h1 h2) (w1 w2) -> c t2 (h2 w2) t1 h1 w1 b', t1=self.traj_num, h2=2, w2=2)
            if self.traj_norm == 'mean':
                traj_shape /= (torch.sum(torch.norm(traj_shape, dim=0), dim=0).mean(dim=0) + 1e-5)
            elif self.traj_norm == 'max':
                traj_shape /= (torch.sum(torch.norm(traj_shape, dim=0), dim=0).max(dim=0)[0] + 1e-5)
            elif self.traj_norm == 'sep':
                traj_shape /= (torch.sum(torch.norm(traj_shape, dim=0), dim=0) + 1e-5)
            else:
                traj_shape /= 20
            traj_shape = rearrange(traj_shape, 'c t2 n t1 h1 w1 b -> b (c t2 n) t1 h1 w1')
            
            hog_feat = rearrange(hog_feat, 'b c t (h1 h2) (w1 w2) -> b (c h2 w2) t h1 w1', h2=2, w2=2)
            
            traj_type = rearrange(traj_type, 'b t (h1 h2) (w1 w2) -> (h2 w2) b t h1 w1', h2=2, w2=2).sum(dim=0)
            traj_type = (traj_type == 4).long()
            
            # use tanh and nolonger need rearange
            target_feat = [hog_feat, traj_shape]
                
            if self.longer_traj > 1:
                for i in range(len(target_feat)):
                    target_feat[i] = self._expand_feat(target_feat[i], self.longer_traj)

            if self.shuffle:
                tt = self.patch_size[0]
                temporal_inds = np.arange(batch_clip.shape[2])
                np.random.shuffle(temporal_inds)
                batch_clip = batch_clip[:,:,temporal_inds,...]
                
                for i in range(len(target_feat)):
                    target_feat[i] = target_feat[i][:,:,temporal_inds[0::2]//tt,...]
                traj_type = traj_type[:,temporal_inds[0::2]//tt,...]
                
            return (batch_clip, [target_feat, traj_type])
        
        elif self.feat_type == 'mbh':
            feat_list = list()
            feat_list.append(self.video_hog(batch_mv[:,0,...][:,None,...])) #MBHX
            feat_list.append(self.video_hog(batch_mv[:,1,...][:,None,...])) #MBHY
            feat_list.append(self.video_hog(batch_mv, is_gradient=True)) #HOF
            feat_list.append(self.video_hog(batch_clip)) #HOG
            mbh_feat = torch.cat(feat_list, dim=1)
            return (batch_clip, [[mbh_feat], None])

        else:
            raise TypeError(f"no such type of feature! {self.feat_type}")


class PredecodeFlowLoader:
    def __init__(self, video_path, flow_root, feat_args=None):
        self.video_path = video_path
        self.flow_root = Path(flow_root)
        self.flow_path = self.flow_root / Path('/'.join(Path(self.video_path).parts[-4:]))
        self.feat_args = feat_args
        # RGB container
        self.rgb_container = decord.VideoReader(
            str(self.video_path),
            width=-1,
            height=-1,
            num_threads=1,
        )
        try:
            # Flow container
            self.flow_x_container = decord.VideoReader(
                str(self.flow_path / 'x.mp4'),
                width=-1,
                height=-1,
                num_threads=1,
            )
        except:
            self.flow_x_container = None
        try:
            self.flow_y_container = decord.VideoReader(
                str(self.flow_path / 'y.mp4'),
                width=-1,
                height=-1,
                num_threads=1,
            )
        except:
            self.flow_y_container = None

    def get_batch(self, inds, load_rgb=True):
        if load_rgb:
            clip_len = self.feat_args['cover_frames']
            num_visible = self.feat_args['num_visible']
            
            if len(inds) > num_visible:
                mask_gap = clip_len // num_visible
                rgb_inds = torch.linspace(0, clip_len-mask_gap, num_visible, dtype=torch.long)
                rgb_inds = inds[rgb_inds]
            else:
                rgb_inds = inds
            
            rgb_batch = self.rgb_container.get_batch(rgb_inds)
            t,h,w,c = rgb_batch.shape
            self.rgb_len = t
        else:
            rgb_batch = None
            clip_len,h,w = 0,0,0
            
        if self.feat_args is not None and self.feat_args['traj_varient'].startswith('Volume'):
            traj_len = self.feat_args['traj_len']
            inds = inds[0::2]
            inds = np.concatenate([np.arange(i,i+traj_len,step=1) for i in inds])
            inds = np.clip(inds, 0, self.__len__() - 2)
            self.flow_len = len(inds)

        try:
            # flow_x_batch = self.flow_x_container.get_batch(inds)[...,0] * 40.0 / 255.0 - 20
            # flow_y_batch = self.flow_y_container.get_batch(inds)[...,0] * 40.0 / 255.0 - 20
            flow_x_batch = self.flow_x_container.get_batch(inds)[...,0]
            flow_y_batch = self.flow_y_container.get_batch(inds)[...,0]
            flow_z_batch = torch.zeros_like(flow_y_batch)
            flow_batch = torch.stack((flow_x_batch, flow_y_batch, flow_z_batch), 3)
        except:
            print(f"load flow file {self.flow_path} failed")
            flow_batch = torch.zeros((clip_len,h,w,3))
        
        return rgb_batch, flow_batch, 1

    def __len__(self):
        return len(self.rgb_container)


class PredecodeTrajLoader:
    def __init__(self, video_path, traj_root, feat_args=None) :
        self.video_path = video_path
        self.traj_root = Path(traj_root)
        self.feat_args = feat_args
        self.container = decord.VideoReader(
            str(self.video_path),
            width=-1,
            height=-1,
            num_threads=1,
        )
    
    def get_batch(self, inds):        
        cover_frame = self.feat_args['cover_frames']
        num_visible = self.feat_args['num_visible']
        traj_len = self.feat_args['traj_len']
        traj_num = self.feat_args['traj_num']
        traj_max_len = self.feat_args['traj_max_len']
        traj_varient = self.feat_args['traj_varient']
        dense_traj = self.feat_args['dense_traj']
        tube_length = self.feat_args['patch_size'][0]
        
        assert traj_len * traj_num <= cover_frame
        assert traj_len <= traj_max_len
        
        traj_path = self.traj_root / Path('/'.join(Path(self.video_path).parts[1:]) + f'_{traj_max_len}.gz')
        
        if dense_traj:
            rgb_inds = inds
            traj_inds = inds[0::tube_length]
            
            if traj_varient == 'VolumeHOG':
                rgb_inds = rgb_inds[0::tube_length]
                rgb_inds = np.concatenate([np.arange(i,i+traj_len,step=1) for i in rgb_inds])
                rgb_inds = np.clip(rgb_inds, 0, self.__len__() - 1)
            
        start_frame = inds[0]
        
        rgb_batch = self.container.get_batch(rgb_inds)
        t,h,w,c = rgb_batch.shape
        self.rgb_len = t

        try:
            traj_dict = compress_pickle.load(traj_path, 'gzip')
            
            if dense_traj:
                assert traj_len <= traj_max_len
                max_avaliable = traj_dict['traj_feat'].shape[1]
                traj_inds = torch.clamp(torch.from_numpy(traj_inds), min=0, max=max_avaliable-1)
                traj_feat = traj_dict['traj_feat'][:,traj_inds,...] # c t h w
                traj_feat = rearrange(traj_feat, '(c n) t h w -> t n h w c', n=traj_max_len)
                traj_feat = traj_feat[:,:traj_len,...]
                traj_feat = rearrange(traj_feat, 't n h w c -> (t n) h w c')
                # traj_type = rearrange(traj_dict['traj_type'], 'n1 (n2 h) w -> (n1 n2) h w', n2=max_avaliable)[traj_inds,...] # t h w
                traj_type = traj_dict['traj_type'][traj_inds, ...]
            
            # concate together to perform augmentation
            traj_type_fake = np.stack((traj_type, traj_type), axis=3)
            traj_feat_n_type = np.concatenate((traj_feat, traj_type_fake), axis=0)
                
            # traj_pos = rearrange(traj_pos, 't h w c -> c t h w')
            # traj_target = np.concatenate((traj_pos, traj_type), axis=0) # t*2 + t//8, h, w
            # traj_target = rearrange(traj_target, 'c t h w -> t h w c')
        except:
            print(f"load traj file {traj_path} failed")
            traj_feat_n_type = np.zeros((cover_frame+cover_frame//traj_len, h//8, w//8, 2))
        
        traj_batch = torch.from_numpy(traj_feat_n_type)
        return rgb_batch, traj_batch, 8

    def __len__(self):
        return len(self.container)
    
    
class SparseTracker(object):
    def __init__(self, clip_len, traj_len, device):
        self.clip_len = clip_len
        self.traj_len = traj_len
        self.device = device
        self.filter = MedianPool2d(3, 1, 1)
        
    def get_init_pos(self, bs, height, width, device):
        return [i.to(device) for i in 
                torch.meshgrid(torch.arange(bs), torch.arange(height), torch.arange(width))]

    def get_hog_pos(self, bs, len, height, width, device):
        return [i.to(device) for i in 
                torch.meshgrid(torch.arange(bs), torch.arange(len), torch.arange(height), torch.arange(width))]

    def shape_2_pos(self, traj_shape, traj_max_len=8):
        b,c,t,h,w = traj_shape.shape
        init_pos = np.stack(np.meshgrid(np.arange(w*8),np.arange(h*8)), axis=0)
        init_pos = init_pos[:,4::8,4::8] # c h w
        traj_pos = torch.zeros_like(traj_shape)
        for i in range(t//traj_max_len):
            traj_pos[:,:,0+i*traj_max_len,...] = torch.from_numpy(init_pos)
            for j in range(traj_max_len-1):
                t = j + i*traj_max_len
                traj_pos[:,:,t+1,...] = traj_shape[:,:,t,...] + traj_pos[:,:,t,...]
                # traj_pos[t+1,...] = traj_pos[t,...]
                traj_pos[:,0,t+1,...] = torch.clamp(traj_pos[:,0,t+1,...],0,w*8-1)
                traj_pos[:,1,t+1,...] = torch.clamp(traj_pos[:,1,t+1,...],0,h*8-1)
            
        return traj_pos.long()
    
    def fetch_hog(self, hogs, traj_pos):
        b,c,t,h,w = traj_pos.shape
        traj_pos_align = traj_pos // 8
        _x,_y = traj_pos_align[:,0,...],traj_pos_align[:,1,...]
        _b,_t,_,_ = self.get_hog_pos(b, t, h, w, traj_pos.device)
        return hogs[_b,:,_t,_x,_y] # b t h w c

    def __call__(self, flow, feats:list=None, dense_flow=True):
        # flow: [b,c,n,h,w]
        b,c,n,h,w = flow.shape
        flow = flow.int()
        # flow = rearrange(flow, 'b c n h w -> b (c n) h w')
        # flow = self.filter(flow).int()
        # flow = rearrange(flow, 'b (c n) h w -> b c n h w', n=n)
        
        num_traj = self.clip_len // self.traj_len
        
        traj_pos_list = list()
        traj_shape_list = list()
        traj_type_list = list()
        margin = 1 if dense_flow else 8
        h,w = h*margin, w*margin
        (_, y_last_dense, x_last_dense) = self.get_init_pos(b, h, w, flow.device)
        (b_last,_,_) = self.get_init_pos(b, h//8, w//8, flow.device)
            
        for i in range(num_traj):
            x_last_sparse = x_last_dense[:,4:h//8*8:8,4:w//8*8:8]
            y_last_sparse = y_last_dense[:,4:h//8*8:8,4:w//8*8:8]
            
            shape_list = list() # shape of trajectory, which encodes local motion patterns (in $3.2.1)
            pos_list = list() # position of each point in each trajectory
            traj_type = torch.ones(x_last_sparse.shape, dtype=torch.long).to(x_last_sparse.device)
            
            # initial position in first frame
            # pos_list.append(torch.stack((x_last, y_last), dim=3))
            for j in range(self.traj_len):
                x_out = (x_last_sparse >= w) | (x_last_sparse < 0)
                y_out = (y_last_sparse >= h) | (y_last_sparse < 0)
                traj_type[x_out | y_out] = 0
                x_last_sparse = torch.clamp(x_last_sparse, min=0, max=w-1)
                y_last_sparse = torch.clamp(y_last_sparse, min=0, max=h-1)
                
                # $P_{t+1}$
                pos_list.append(torch.stack((x_last_sparse, y_last_sparse), dim=3))

                t = j + i * self.traj_len

                x_last_sparse_align = x_last_sparse // margin
                y_last_sparse_align = y_last_sparse // margin

                x_shape = flow[b_last, 0, t, y_last_sparse_align, x_last_sparse_align]
                y_shape = flow[b_last, 1, t, y_last_sparse_align, x_last_sparse_align]

                x = x_last_sparse + x_shape
                y = y_last_sparse + y_shape
                
                x_last_sparse = x
                y_last_sparse = y
                
                # $\Delta P_t = (P_{t+1} - P_t)$
                shape_list.append(torch.stack((x_shape, y_shape), dim=3))

            shape_matrix = torch.stack(shape_list, dim=1)
            pos_matrix = torch.stack(pos_list, dim=1)
            traj_shape_list.append(shape_matrix)
            traj_pos_list.append(pos_matrix)
            traj_type_list.append(traj_type)
            
        traj_shape = torch.stack(traj_shape_list, dim=1) # [b t n h w c]
        traj_pos = torch.stack(traj_pos_list, dim=1)
        traj_type = torch.stack(traj_type_list, dim=1)
        
        traj_pos = rearrange(traj_pos, 'b n1 n2 h w c -> b (n1 n2) h w c')
        
        # traj_pos_align = traj_pos // 8
        # _x,_y = traj_pos_align[...,0],traj_pos_align[...,1]
        # if dense_flow:
        #     _b,_t,_,_ = self.get_hog_pos(b, n, h//8, w//8, flow.device)
        # else:
        #     _b,_t,_,_ = self.get_hog_pos(b, n, h, w, flow.device)
        
        # # HOG, HOF, MBHX, MBHY
        # traj_feat = list()
        # for feat in feats:
        #     feat = feat[_b,:,_t,_x,_y] # wrong here! x is w, y is h
        #     feat = rearrange(feat, 'b (n1 n2) h w c -> b (c n2) n1 h w', n2=self.traj_len)
        #     traj_feat.append(feat)
        
        # traj_shape = rearrange(traj_shape, 'b n1 n2 h w c -> (c n2) b n1 h w', n2=self.traj_len).float()
        # traj_shape /= (traj_shape.norm(dim=0) + 1e-5)
        # traj_shape = rearrange(traj_shape, 'c b n h w -> b c n h w')
        
        traj_shape = rearrange(traj_shape, 'b n1 n2 h w c -> b (c n2) n1 h w')
        traj_shape = torch.clamp(traj_shape, min=-20, max=20)
        
        # traj_feat = torch.cat((traj_shape, *traj_feat), dim=1)
        traj_feat = traj_shape
        
        return traj_feat, traj_pos, traj_type
    
    
class HOGDescriptorCPU(object):
    def __init__(self, img_size=(224,224), block_size=16, block_stride=16, cell_size=8, nbins=9):
        assert not img_size[0] % block_size \
            and not img_size[1] % block_size \
            and not block_size % cell_size \
            and not block_stride % cell_size

        self.img_size = img_size
        self.block_size = (block_size, block_size)
        self.block_stride = (block_stride, block_stride)
        self.cell_size = (cell_size,cell_size)
        self.nbins = nbins
        # self.hog = cv2.HOGDescriptor(self.img_size, self.block_size, self.block_stride,
        #                              self.cell_size, self.nbins)
        self.hog = lambda image : ski_hog(
            image, 
            orientations=self.nbins, 
            pixels_per_cell=self.cell_size, 
            cells_per_block=(block_size//cell_size, block_size//cell_size),
            visualize=False,
            channel_axis=-1
        )

    def __call__(self, clip):
        t,h,w,c = clip.shape
        hog_clip = list()
        for i in range(t):
            # hog_feat = self.hog.compute(clip[i,...])
            hog_feat = self.hog(image=clip[i,...])
            # _h = self.img_size[0]//self.block_stride[0]
            _h = self.img_size[0]//self.cell_size[0] - 1
            hog_feat = rearrange(hog_feat, '(h w c) -> h w c', h=_h, w=_h)
            inds = np.linspace(0, _h-1, (_h+1)//2, dtype=np.int16)
            _grid_x, _grid_y = np.meshgrid(inds,inds)
            hog_feat = hog_feat[_grid_x,_grid_y,:]
            _h2 = self.block_stride[0]//self.cell_size[0]
            hog_feat = rearrange(hog_feat, 'h1 w1 (h2 w2 c) -> (h1 h2) (w1 w2) c', h2=_h2, w2=_h2)
            hog_clip.append(hog_feat)
        return np.stack(hog_clip)


# https://gist.github.com/etienne87/b79c6b4aa0ceb2cff554c32a7079fa5a
# https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py
class HOGDescriptorByConv(nn.Module):
    def __init__(self, nbins=10, cell_size=8, block_size=16, max_angle=math.pi, stride=1, padding=1, dilation=1, eps=1e-5, norm='L2'):
        super(HOGDescriptorByConv, self).__init__()
        self.nbins = nbins
        self.cell_size = cell_size
        self.block_size = block_size
        self.max_angle = max_angle
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.eps = eps
        self.norm = norm
        
        sobel_operator = torch.FloatTensor([[1, 0, -1],
                                            [2, 0, -2],
                                            [1, 0, -1]])
        mat = torch.cat((sobel_operator[None], sobel_operator.rot90(k=-1)[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(cell_size, stride=cell_size, padding=0, ceil_mode=False, count_include_pad=True)
        # use replica padding instead of zero padding
        self.replica_pad = nn.ReplicationPad2d(padding=tuple([self.padding] * 4))

        self.conv2d = nn.Conv2d(1,2, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv2d.weight = torch.nn.Parameter(torch.stack((sobel_operator[None], sobel_operator.rot90(k=-1)[None])))

    def forward(self, x, is_gradient=False):
        # 0. Convert RGB image into gray scale
        if not is_gradient and x.size(1) > 1:
            x = x.mean(dim=1)[:,None,:,:]

        # n_block = list(map(lambda x: x//self.block_size, x.shape[-2:]))

        # 1. Calculate gradient with Sobel operator
        if not is_gradient:
            x = self.replica_pad(x)
            # x = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, 1)
            with torch.no_grad():
                x = self.conv2d(x) # nn.conv2d is faster then F.conv2d

        # 2. Mag/ Phase
        mag = x.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:, None, :, :]
        phase_int[phase_int<0] += 9
        phase_ceil = phase_int.ceil().long()
        phase_floor = phase_int.floor().long()

        n, c, h, w = x.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=x.device)
        out.scatter_(1, phase_ceil % self.nbins, norm * (phase_int - phase_floor))
        out.scatter_(1, phase_floor % self.nbins, norm * (phase_ceil - phase_int))

        out = self.pooler(out)

        # 4. Normalization in each block
        ncell_per_block = self.block_size // self.cell_size
        out = rearrange(out, 'b c (h h1) (w w1) -> (c h1 w1) (b h w)', h1=ncell_per_block, w1=ncell_per_block)
        if self.norm == 'L1':
            out /= torch.sqrt(torch.sum(out, dim=0) + self.eps)
        if self.norm == 'L2':
            out /= torch.sqrt(torch.sum(out ** 2, dim=0) + self.eps ** 2)
        elif self.norm == 'L2-Hys':
            out /= torch.sqrt(torch.sum(out ** 2, dim=0) + self.eps ** 2)
            # out = torch.minimum(out, torch.FloatTensor(0.2).to(out.device))
            out = torch.clamp(out, 0, 0.2)
            out /= torch.sqrt(torch.sum(out ** 2, dim=0) + self.eps ** 2)
        elif self.norm == 'std':
            mean = out.mean(dim=0, keepdim=True)
            var = out.var(dim=0, keepdim=True)
            out = (out - mean) / (var + 1.e-6)**.5
        else:
            pass
        out = rearrange(out, '(c h1 w1) (b h w) -> b c (h h1) (w w1)', h1=ncell_per_block, w1=ncell_per_block, h=h//self.block_size, w=w//self.block_size)

        return out


class VideoHOGFeat(object):
    def __init__(self, nbins=9, cell_size=8, block_size=16, patch_size=None, norm='L2', channel='rgb', device='cuda') -> None:
        super().__init__()
        self.hog = HOGDescriptorByConv(nbins=nbins, cell_size=cell_size, block_size=block_size, norm=norm).to(device)
        self.patch_size = patch_size
        self.channel = channel

    def __call__(self, clip, is_gradient=False):
        b,c,t,h,w = clip.shape
        if self.channel == "gray":
            clip = rearrange(clip, 'b c t h w -> (b t) c h w')
            cr = 1
        if self.channel == "rgb":
            clip = repeat(clip, 'b c t h w -> (b t c) c0 h w', c0=1)
            cr = 3
        hog_feat = self.hog(clip, is_gradient=is_gradient)
        hog_feat = rearrange(hog_feat, '(b t cr) ch h w -> b (cr ch) t h w', b=b, t=t,cr=cr)
        # pick up hog of middle frame as target feature
        # refered in paper $4.3
        if self.patch_size[0] > 1:
            d = self.patch_size[0]
            d1 = t // d
            hog_idx = np.arange(0,d1,step=1)*d+d//2
            hog_feat = hog_feat[:,:,hog_idx,:,:]

        return hog_feat


def HogVisual(hog, img_size=(224,224), cell_size=8, nbins=9, topk=9):
    # input parameters:
    #   - hog: np.ndarray([nbins, ncell_x, ncell_y]) -> hog descriptor
    #   - img_size: (height, width) -> size of the visualization image
    #   - cell_size: int -> pixels of the hog cell
    #   - nbins: int -> number of the gradient orientation bins
    #   - topk: int -> visualized top k orientation based on the magnitude
    # output:
    #   - visual_img: visualized hog descriptor
    # author:
    #   - XinyuSun -> xinyusun.cs@icloud.com
    assert topk <= hog.shape[2]

    # hog /= hog.max()

    height, width = list(map(lambda x: x*cell_size, hog.shape[:2]))
    visual_img = np.zeros(img_size, dtype=np.float64)
    angles = np.linspace(0, np.pi - np.pi // nbins, nbins)
    for x in range(width // cell_size):
        for y in range(height // cell_size):
            for i in np.argsort(hog[x,y,:])[-topk:]:
                a = angles[i]
                xcent = x * cell_size + cell_size // 2
                ycent = y * cell_size + cell_size // 2
                x1 = int(xcent + (cell_size // 2 - 1) * math.cos(-a))
                y1 = int(ycent + (cell_size // 2 - 1) * math.sin(-a))
                x2 = int(xcent - (cell_size // 2 - 1) * math.cos(-a))
                y2 = int(ycent - (cell_size // 2 - 1) * math.sin(-a))
                xx, yy = draw.line(x1, y1, x2, y2)
                visual_img[xx, yy] += float(hog[x,y,i])
                # cv2.line(visual_img, (y1, x1), (y2, x2), int(255 * (hog[x,y,i])))

    return visual_img


def FlowVisual(flow, show_style=2):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#将直角坐标系光流场转成极坐标系

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    # hsv转bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr