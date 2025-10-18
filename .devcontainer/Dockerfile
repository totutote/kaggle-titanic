FROM gcr.io/kaggle-gpu-images/python:latest
LABEL purpose="kaggle container"

# RTX 5070 Ti (CUDA sm_120) サポートのため、最新のPyTorchをインストール
RUN pip install --no-cache-dir --upgrade \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130

# 追加で必要なパッケージのみインストール
#RUN pip install --no-cache-dir \
#  grad-cam plotly

#USER kaggle
WORKDIR /kaggle/working