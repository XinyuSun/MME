# Extract Motion Trajectories From Scratch

We also provide scripts to extract motion trajectories. The extraction process can be summerize as:
<ul>
    <li>A. Calculate optical flow.</li>
    <li>B. Extract motion trajectories using the optical flow.</li>
</ul>



❕ As this process require complex multi-nodes computation, we strongly recommend to download and [use the pre-extracted motion trajectories](#A-using-pre-extracted).


## A. Calculate Optical Flow
We compile a flow extractor with multi-node and multi-gpu support based on the `dense_flow` [repository](https://github.com/XinyuSun/dense_flow). The binary file can be download according to your machine architecture and GPU type.
| System |  Arch  |    GPU    | Download Link |
|:------:|:------:|:---------:|:-------------:|
| Linux  | x86_64 | A100/A800 | extractor.zip |
| Linux  | x86_64 |    3090   | [extractor.zip](https://scut.cowtransfer.com/s/2de5a04347e040) |
<!-- | Linux  | x86_64 |  TiTan XP | extractor.zip | -->

After you download the extractor file, unzip it into current dir.
```bash
unzip -d ./flow extractor.zip
```
Input the data csv file in `data/csv` to determine videos to calculate optical flow. For a device with 8 x 3090 GPUs, we set `NUM_PROCESS` = 8 or 16 and `NUM_GPU` = 8. You can split the data csv file into several chunks according to the number of your machines to enable multi-node extraction.
> GPU memory size is not a crucial factor in the extraction process. The extraction speed is mainly limited by the number of CPU cores, memory bandwidth, and storage speed.
```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./flows/lib ./flows/extract_warp_gpu_parallel -f=data/csv/ucf101/train.csv -d=data/flows -n=NUM_PROCESS -g=NUM_GPU
```

The extracted flow has a same organization with dataset:
```bash
data/flow
└── UCF-101
    └── ApplyEyeMakeup
        └── v_ApplyEyeMakeup_g24_c05.avi
            ├── x.mp4
            └── x.mp4
```

## B. Extract motion trajectories using the optical flow

Now you can extract motion trajectory using the calculated optical flow:
```bash
python m3video/extract.py --ds_base=data/csv/ucf101/train.csv --dump_base data/trajs --num_process=NUM_PROCESS --job_type=trajs_dense --traj_len=6 --gpu_num=NUM_GPU
```
This process will dump the motion trajectories into compressed pickle files under the organization as:
```bash
data/trajs
└── UCF-101
    └── ApplyEyeMakeup
        └── v_ApplyEyeMakeup_g24_c05.avi_6.gz
```
