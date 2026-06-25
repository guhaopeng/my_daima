#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n physiopt python=3.10
conda activate physiopt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers psutil
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast

mkdir -p /tmp/extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install /tmp/extensions/diffoctreerast --no-build-isolation

mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

mkdir -p /tmp/extensions
cp -r deps/trellis-physiopt/extensions/vox2seq /tmp/extensions/vox2seq
pip install /tmp/extensions/vox2seq --no-build-isolation

pip install spconv-cu118
