## sentiment:berts镜像使用

```
$ docker run -it --rm --privileged=true --name sentiment1 -v /data/app:/app -u 1000:1000 sentiment:berts python app.py --input test.xlsx --step 128 --output res_sentiment

[docker]
--rm 执行完删除容器
--privileged=true
-v 创建数据卷
-u 用户/用户组
[python]
--input alls.xlsx 输入文件路径 支持xlsx，csv，必须参数
--field 文本字段名称，默认phrase
--step 128 多少缓存一次，默认128000
--output 输入文件名称 例如 res_sentiment、res.csv
```
## alg/pytorch:sentiment_bert 镜像使用

```
$ docker run -it --rm --privileged=true --name sentiment_bert -v /opt/DOCKER:/app -u 1000:1000 alg/pytorch:sentiment_bert python app.py --input test.xlsx --step 128 --output res_sentiment

[docker]
--rm 执行完删除容器
--privileged=true
-v 创建数据卷
-u 用户/用户组
[python]
--model 可选参数 ['car','jd','jd_long']，默认car
--input alls.xlsx 输入文件路径 支持xlsx，csv，必须参数
--field 文本字段名称，默认phrase
--step 128 多少缓存一次，默认128000
--output 输入文件名称 例如 res_sentiment、res.csv
```

## alg/pytorch:sentiment_bert_gpu 镜像使用

```
$ docker run -it --gpus all --rm --privileged=true --name sentiment_bert -v /opt/DOCKER:/workspace -u 1000:1000 alg/pytorch:sentiment_bert_gpu python app.py --input test.xlsx --step 128 --output res_sentiment

[docker]
--rm 执行完删除容器
--gpus all 加载GPU
--privileged=true
-v 创建数据卷
-u 用户/用户组
[python]
--model 可选参数 ['car','jd','jd_long']，默认car
--input alls.xlsx 输入文件路径 支持xlsx，csv，必须参数
--field 文本字段名称，默认phrase
--step 128 多少缓存一次，默认128000
--output 输入文件名称 例如 res_sentiment、res.csv
```


## test

- 测试环境alg001
```
docker run -it --rm --privileged=true --name sentiment_v1 -v /opt/DOCKER:/app -u 1000:1000 sentiment:bert python app.py --input test.csv --step 128 --output res_sentiment
```

docker run -it --rm --privileged=true --name sentiment -v /opt/DOCKER:/app db27413ed081 /bin/bash

- 正式环境
```
- 指定自定义输入输出路径命令
$ docker run -it --rm --privileged=true --name sentiment1 -u 1000:1000 sentiment:berts -v /data/app:/app -v /data/alg/wangzixiang:/data python app.py --input /data/test.csv --step 128 --output res_sentiment

$ docker run -it --rm --privileged=true --name sentiment1 -u 1000:1000 sentiment:berts -v /data/app:/app -v /data/alg/wangzixiang:/data python app.py --input /data/test.csv --step 128 --output /data/res_sentiment.csv

--$ docker run -it --rm --privileged=true --name sentiment1 -u 1000:1000 sentiment:berts -v /data/app:/app -v $(pwd):/data python app.py --input /data/test.csv --step 128 --output /data/res_sentiment.csv--

```

