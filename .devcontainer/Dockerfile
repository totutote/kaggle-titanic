FROM gcr.io/kaggle-gpu-images/python:latest
LABEL purpose="kaggle container"

# 追加で必要なパッケージのみインストール
#RUN pip install --no-cache-dir \
#  grad-cam plotly

# Kaggle環境に合わせたユーザー設定（必要に応じて）
# 既存のkaggleユーザーがある場合はそれを使用、なければ作成
RUN if ! id kaggle 2>/dev/null; then \
  useradd -m -s /bin/bash kaggle; \
fi

USER kaggle
WORKDIR /kaggle/working