
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

# Docker

## annotation api

Run annotation API:

```bash
 docker run -it --rm -p 8000:8000 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name mogface -d 097234065497.dkr.ecr.eu-west-1.amazonaws.com/mogface uvicorn --host 0.0.0.0 annotation_api:app
```

## Send request to API

```bash
curl -l -X POST "localhost:8000/cloud" -H "Content-Type: application/json" -d '{"bucket_name": "veesion-test-blurring", "video_key": "a2pas-alma/2020-11-03_16h58m38s_to_2020-11-03_16h58m55s_camera_9_ip_192.168.1.108_port_37777.mp4"}'
```
