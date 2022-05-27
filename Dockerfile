FROM bitnami/pytorch:latest

LABEL maintainer="thyme"
LABEL description="pytorch/bert"

# 更新pip版本
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
# 安装pytorch
RUN pip install torch torchvision torchaudio  -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装transformers
RUN pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装pandas xlrd
RUN pip install pandas xlrd -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV LC_ALL zh_CN.UTF-8
ENV TZ Asia/Shanghai
ENV PATH /usr/local/python3/bin/:$PATH

