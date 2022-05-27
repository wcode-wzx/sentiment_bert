#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys, time, shutil
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Sentiment by bert')

parser.add_argument('--model', default='bert_car_0426.pkl', type=str, help='model path')
parser.add_argument('--input', default='all.xlsx',required=True, type=str, help='input path')
parser.add_argument('--field', default='phrase', type=str, help='Select a field to brush emotions')
parser.add_argument('--step', default='128000', type=int, help='chunk size')
parser.add_argument('--output', default='res_sentiment', type=str, help='output path')


args = parser.parse_args()

model_name = args.model  # bert

print("input:", args.input)
print("step:", args.step)

# 加载模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model.load_state_dict(torch.load("/content/drive/MyDrive/fast_nlp/save_model/bert_article_0406.pkl"))
model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

# 分词器，词典
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# 数据集读取
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    
    def __len__(self):
        return len(self.labels)

def get_dataloader(news_text, test_label):
    
    test_encoding = tokenizer(news_text, truncation=True, padding=True, max_length=128)
    test_dataset = NewsDataset(test_encoding, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_dataloader

def predict(test_dataloader):
    res_label = []
    res_probability = []
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask) 

            res_label.append([i.argmax().item() for i in outputs["logits"]])
            res_probability.append([max(nn.Softmax(dim=0)(i).tolist()) for i in outputs["logits"]])
    return res_label, res_probability


def load_excel(file_path, ind, step):
    dfs = pd.read_excel(file_path).loc[int(ind)*int(step):(int(ind)+1)*int(step)]
    return dfs

def load_csv(file_path, ind, step):
    dfs = pd.read_csv(file_path).loc[int(ind)*int(step):(int(ind)+1)*int(step)]
    return dfs


def workflow(file_path, ind):
    #字段
    field = args.field
    # 加载数据
    if file_path.split('.')[-1] == 'csv':
        chunk = load_csv(file_path, ind, step=args.step)
    if file_path.split('.')[-1] == 'xlsx':
        chunk = load_excel(file_path, ind, step=args.step)

    news_text = [str(i).replace(' ','').replace('\n','') for i in chunk[field]]
    test_label = ["0" for i in range(0,len(news_text))]
    # 加载tokenizer
    test_dataloader = get_dataloader(news_text,test_label)
    # 推理
    res_label, res_probability = predict(test_dataloader)

    chunk['res_label'] = [j for i in res_label for j in i]
    chunk['res_probability'] = [j for i in res_probability for j in i]

    return chunk
    
    
def get_files_name(path):
    '''Merge all dfs under path'''
    for root,dirs,files in os.walk(path):
        for file in files:
            yield os.path.join(root, file)
            
            
def get_concat_df(path):
    '''Merge all dfs under path'''
    temp = []
    for i in get_files_name(path):
        if i.split('.')[-1] == 'csv':
            temp.append(pd.read_csv(i))
        if i.split('.')[-1] == 'xlsx':
            temp.append(pd.read_excel(i))
    dfs = pd.concat(temp)
    return dfs


def main(file_path):

    if file_path.split('.')[-1] == 'csv':
        lens = len(pd.read_csv(file_path))
    elif file_path.split('.')[-1] == 'xlsx':
        lens = len(pd.read_excel(file_path))
    else:
        print("Only supports data in csv and xlsx formats")
        sys.exit()
    # 清除缓存
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    os.makedirs('temp', exist_ok=True)
    
    for ind in range(0,int(lens/args.step)+1):
        chunk = workflow(file_path, ind)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        chunk.to_csv("temp/"+str(ind)+"_"+now+".csv", index = False)
        
    # 合并保存        
    if args.output.split('.')[-1] == 'csv':
        get_concat_df("temp").to_csv(args.output, index = False)
        print("Finish and save the file to " + args.output)
    elif args.output.split('.')[-1] == 'xlsx':
        get_concat_df("temp").to_excel(args.output, index = False)
        print("Finish and save the file to " + args.output)
    else:    
        if file_path.split('.')[-1] == 'csv':            
            get_concat_df("temp").to_csv(args.output+".csv", index = False)
            print("Finish and save the file to " + args.output+".csv")
        if file_path.split('.')[-1] == 'xlsx':
            get_concat_df("temp").to_excel(args.output+".xlsx", index = False)
            print("Finish and save the file to " + args.output+".xlsx")
    
    
if __name__=="__main__":
    main(file_path=args.input)