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
RUN pip install pandas xlrd openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p /tmp/bert-base-chinese
RUN mkdir -p /tmp/models
COPY bert-base-chinese/* /tmp/bert-base-chinese/
COPY models/* /tmp/models/

ENV LC_ALL zh_CN.UTF-8
ENV TZ Asia/Shanghai
ENV PATH /usr/local/python3/bin/:$PATH