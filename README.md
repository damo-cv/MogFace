
# Installation

```
git clone git@github.com:veesion-io/MogFace.git
cd MogFace
nvidia-docker run --gpus all --name blurring --security-opt seccomp=unconfined \
  --net=host --ipc=host -v /dev/shm:/dev/shm --ulimit memlock=-1 \
  -v /path/to/MogFace:/workspace/ -v /path/to/your/videos:/workspace/videos/ \
  --ulimit stack=67108864 -it nvcr.io/nvidia/pytorch:23.06-py3 /bin/bash

cd /workspace/
pip install gdown
gdown https://drive.google.com/uc?id=1s8LFXQ5-zsSRJKVHLFqmhow8cBn4JDCC
mkdir -p snapshots/MogFace && mv model_140000.pth snapshots/MogFace/
mkdir annotations

cd utils/nms && python setup.py build_ext --inplace && cd ../..
cd utils/bbox && python setup.py build_ext --inplace && cd ../..

apt update && DEBIAN_FRONTEND=noninteractive apt install ffmpeg -y
pip install ffprobe3
```

# Detect and blur faces : 

Detect faces with a deep learning model and save .txt annotations files.
```
CUDA_VISIBLE_DEVICES=0 python test_multi.py -c configs/mogface/MogFace.yml -n 140
```

Use the created files to create videos with faces blurred : 
```
python3 blur_detected_faces.py
```
