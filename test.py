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

    


df1 = BaseData("test.csv")

print(df1.read())