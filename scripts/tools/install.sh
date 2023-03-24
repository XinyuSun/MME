# conda create environment
conda create -n mme python=3.8
source activate mme

# install pytorch 1.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# install other pacakges
pip install triton==1.0.0 timm==0.4.12 TensorboardX decord einops opencv-python av scikit-image compress-pickle pandas

# build and install deepspeed
DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed