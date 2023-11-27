FROM nvcr.io/nvidia/pytorch:22.10-py3

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa         \
    && apt-get update                                \
    && apt-get install -y                            \
        libcap-dev                                   \
        curl                                         \
        ffmpeg                                       \
        python3.8-dev                                \
        libsdl2-dev                                  \
    && rm -rf /var/lib/apt/lists/*                   \
    && pip3 install --upgrade pip


RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1s8LFXQ5-zsSRJKVHLFqmhow8cBn4JDCC

RUN mkdir -p snapshots/MogFace
RUN mv model_140000.pth snapshots/MogFace
RUN mkdir annotations

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install ffmpeg -y
RUN pip install ffprobe3

COPY utils ./utils

RUN cd /workspace/utils/nms && python setup.py build_ext --inplace
RUN cd /workspace/utils/bbox && python setup.py build_ext --inplace

COPY requirements_annotation_api.txt .

RUN pip install -r requirements_annotation_api.txt

RUN pip3 uninstall -y opencv-contrib-python \
    && rm -rf /opt/conda/lib/python3.8/site-packages/cv2 \
    && pip install opencv-contrib-python

COPY . .

CMD ["/bin/bash"]
