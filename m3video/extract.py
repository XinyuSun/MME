import torch
import numpy as np
import cProfile
import decord
import compress_pickle
from pathlib import Path
import argparse
from threading import Thread
import torch.multiprocessing as mp
from queue import Queue
import glob
import os
import argparse
from tqdm import tqdm
from einops import rearrange
import warnings
import os
import pandas as pd

from m3video.feat import PyAVDecodeMotionVector, VideoHOGFeat, PredecodeFlowLoader, SparseTracker

decord.bridge.set_bridge('torch')


def extract_mv(t_id:int, queue:Queue, dump_base:Path, args):
    if t_id == 0:
        job_num = queue.qsize()
        bar = tqdm(total=job_num)
        bar_pos = 0
    while not queue.empty():
        file_path = queue.get()
        file_with_class = Path('/'.join(file_path.parts[-3:]) + '.gz')
        dump_path = dump_base / file_with_class

        if not dump_path.exists():
            mvdecode = PyAVDecodeMotionVector(
                multi_thread=True, 
                mode='efficient', 
                path=str(file_path)
            )
            num_frames = len(mvdecode)
            inds = np.linspace(0, num_frames-1, num_frames)
            mv_output = mvdecode.get_batch(inds)
            
            if not dump_path.parent.exists():
                os.makedirs(dump_path.parent)
            compress_pickle.dump(mv_output, dump_path, 'gzip')

        if t_id == 0:
            exist_job = queue.qsize()
            bar_len = job_num - exist_job
            cur_update = bar_len - bar_pos
            bar_pos += cur_update
            bar.update(cur_update)


def extract_traj_dense(t_id:int, queue:Queue, dump_base:Path, args):
    t_device = f'cuda:{t_id%args.gpu_num}'
    if t_id == 0:
        job_num = queue.qsize()
        bar = tqdm(total=job_num)
        bar_pos = 0
    while not queue.empty():
        file_path = queue.get()
        file_with_class = Path('/'.join(file_path.parts[-3:]) + f'_{args.traj_len}.gz')
        dump_path = dump_base / file_with_class

        if not dump_path.exists():
            try:
                vr = PredecodeFlowLoader(file_path, 'data/flows')
                clip_len = len(vr) - 1
                inds = np.arange(clip_len)
                _, flow, _ = vr.get_batch(inds, load_rgb=False)
                flow = rearrange(flow, 'n h w c -> c n h w').to(t_device)
                tracker = SparseTracker(
                    clip_len=args.traj_len, # track trajectory for each frame
                    traj_len=args.traj_len,
                    device=t_device
                )
                
                max_traj_num = clip_len - args.traj_len
                traj_feat_list = []
                traj_type_list = []
                for i in range(max_traj_num):
                    traj_feat, traj_pos, traj_type = tracker(flow[None,:,i:i+args.traj_len,...])
                    traj_feat_list.append(traj_feat)
                    traj_type_list.append(traj_type)
                    
                traj_feat = torch.cat(traj_feat_list, dim=2)
                # traj_type = torch.concat(traj_type_list, dim=2)
                # TODO: be wong here:
                # please update the dataloader if fix and re-generate it
                traj_type = torch.cat(traj_type_list, dim=1)
                
                if not dump_path.parent.exists():
                    os.makedirs(dump_path.parent)
                compress_pickle.dump({
                    'traj_feat':traj_feat[0].cpu().numpy().astype(np.int8),
                    # 'traj_pos': traj_pos[0].numpy().astype(np.int16),
                    'traj_type':traj_type[0].cpu().numpy().astype(np.uint8)
                    }, dump_path, 'gzip')
            except:
                  print(file_path)
        
        if t_id == 0:
            exist_job = queue.qsize()
            bar_len = job_num - exist_job
            cur_update = bar_len - bar_pos
            bar_pos += cur_update
            bar.update(cur_update) 


def extract_traj(t_id:int, queue:Queue, dump_base:Path, args):
    t_device = f'cuda:{t_id%args.gpu_num}'
    if t_id == 0:
        job_num = queue.qsize()
        bar = tqdm(total=job_num)
        bar_pos = 0
    while not queue.empty():
        file_path = queue.get()
        file_with_class = Path('/'.join(file_path.parts[-3:]) + f'_{args.traj_len}.gz')
        dump_path = dump_base / file_with_class

        if not dump_path.exists():
            try:
                vr = PredecodeFlowLoader(file_path, 'data/flows')
                clip_len = ((len(vr) - 1) // args.traj_len) * args.traj_len # align to traj_len
                inds = np.arange(clip_len)
                _, flow, _ = vr.get_batch(inds, load_rgb=False)
                flow = rearrange(flow, 'n h w c -> c n h w').to(t_device)
                
                tracker = SparseTracker(
                    clip_len=clip_len, 
                    traj_len=args.traj_len,
                    device=t_device
                )
                traj_feat, traj_pos, traj_type = tracker(flow[None])
                if not dump_path.parent.exists():
                    os.makedirs(dump_path.parent)
                compress_pickle.dump({
                    'traj_feat':traj_feat[0].cpu().numpy().astype(np.int8),
                    # 'traj_pos': traj_pos[0].numpy().astype(np.int16),
                    'traj_type':traj_type[0].cpu().numpy().astype(np.uint8)
                    }, dump_path, 'gzip')
            except:
                  print(file_path)  
        if t_id == 0:
            exist_job = queue.qsize()
            bar_len = job_num - exist_job
            cur_update = bar_len - bar_pos
            bar_pos += cur_update
            bar.update(cur_update)
            

class MultiThreadExtractor:
    def __init__(self, ds_base:Path, dump_base:Path, num_process=16) -> None:
        self.num_process = num_process
        self.queue = Queue()
        self.files = glob.glob(str(ds_base / Path('train_video/*/*.mp4')))
        self.dump_base = dump_base
        print(f'total files parsed: {len(self.files)}')

        for file in self.files:
            self.queue.put(Path(file))

    def __call__(self, target, args):
        self.threads = list()
        for i in range(self.num_process):
            t = Thread(target=target, args=(i, self.queue, self.dump_base, args))
            t.run()
            self.threads.append(t)

        for t in self.threads:
            t.join()
            

class MultiProcessExtractor:
    def __init__(self, ds_base:Path, dump_base:Path, num_process=16):
        self.num_process = num_process
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        if os.path.isfile(str(ds_base)):
            self.files = pd.read_csv(ds_base, sep=' ', header=None, usecols=[0])
            self.files = [f[0] for f in self.files.values.tolist()]
        else:   
            if ds_base.stem == 'smth-smth-v2':
                folder = Path('*/*.mp4')
            else:
                folder = Path('train_video/*/*.mp4')
            self.files = glob.glob(str(ds_base / folder))
        
        self.dump_base = dump_base
        print(f'total files parsed: {len(self.files)}')

        for file in self.files:
            self.queue.put(Path(file))

    def __call__(self, target, args):
        mp.spawn(target, args=(self.queue, self.dump_base, args), nprocs=self.num_process)
            

def main(args):
    mte = MultiProcessExtractor(
       ds_base=args.ds_base,
       dump_base=args.dump_base,
       num_process=args.num_process
    )
    if args.job_type == 'mvs':
        target = extract_mv
    elif args.job_type == 'trajs':
        target = extract_traj
    elif args.job_type == 'trajs_dense':
        target = extract_traj_dense
    mte(target=target, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_base', type=Path, default='data/kinetics400')
    parser.add_argument('--dump_base', type=Path, default='data/trajs')
    parser.add_argument('--num_process', '-n', type=int, default=8)
    parser.add_argument('--job_type', type=str, default='trajs')
    parser.add_argument('--traj_len', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, default=8)
    args = parser.parse_args()
    main(args)