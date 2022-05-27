import torch.nn as nn
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# 加载模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model.load_state_dict(torch.load("/content/drive/MyDrive/fast_nlp/save_model/bert_article_0406.pkl"))
model.load_state_dict(torch.load("1652954229.2630832_bert_car_0519.pkl", map_location=torch.device('cpu')))

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


def predict(res_label, res_probability,test_dataloader):
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask) 

            res_label.append([i.argmax().item() for i in outputs["logits"]])
            res_probability.append([max(nn.Softmax(dim=0)(i).tolist()) for i in outputs["logits"]])
    return res_label,res_probability



def pre():
    res_label = []
    res_probability = []
    import json
    text=request.get_data()
    set_config = json.loads(text)
    news_text = [str(i).replace(' ','').replace('\n','') for i in set_config]
    test_label = ["0" for i in range(0,len(news_text))]

    try:
        test_dataloader = get_dataloader(news_text,test_label)
        res_label,res_probability = predict(res_label, res_probability, test_dataloader)
        new_res_label = [j for i in res_label for j in i]
        new_probability = [j for i in res_probability for j in i]

        out =[[i, j] for i,j in zip(new_res_label, new_probability)]   
        return jsonify({"result":out})

    except Exception as e:

        print(e)

        return jsonify({"result":"Model Failed"})

if __name__=="__main__":
    app.run('0.0.0.0',port=5000,debug=True)