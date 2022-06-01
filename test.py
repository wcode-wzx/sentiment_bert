#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd


class BaseData():
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.fname = file_path.split('\\')[-1].split('/')[-1]
        self.ftype = self.fname.split('.')[-1]
        print(self.fname, self.ftype)
        # df = pd.read_csv(file_path)

    def read(self):
        if self.ftype == 'csv':
            df = pd.read_csv(self.file_path)
        if self.ftype == 'xlsx':
            df = pd.read_excel(self.file_path)
        return df



class BuildData():
    def __init__(self, file_path) -> None:
        file_path = file_path
        fname = file_path.split()
        df = pd.read_csv(file_path)

    


# df1 = BaseData("test.csv")

# print(df1.read())

from multiprocessing import Pool
import os
import time
import random
   
def worker(msg):
   t_start = time.time()
   print("%s开始执行,进程号为%d" % (msg, os.getpid()))
   # random.random()随机生成0~1之间的浮点数
   time.sleep(random.random()*2)
   t_stop = time.time()
   print(msg, "执行完毕，耗时%0.2f" % (t_stop-t_start))
   
if __name__ == "__main__":
    po = Pool(3) # 定义一个进程池，最大进程数3
    for i in range(0, 8):
    # Pool().apply_async(要调用的目标,(传递给目标的参数元祖,))
    # 每次循环将会用空闲出来的子进程去调用目标
        po.apply_async(worker, (i,))
    
    print("----start----")
    # 关闭进程池，关闭后po不再接收新的请求
    po.close()
    # 等待po中所有子进程执行完成，必须放在close语句之后
    po.join()
    print("-----end-----")