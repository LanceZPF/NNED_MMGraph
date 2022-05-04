import datetime
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
from sklearn import metrics
from model import *
from datasets import *
import os
import random



##########################
# 学习参数的设置
epoch = 80
learning_rate = 0.0000001
batchsize = 8
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
##########################


def seed_torch(seed=1999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    # 用于训练EL模型
    seed_torch(1999)
    train_loader = DataLoader(EL_datasets('Train'), batch_size = batchsize, shuffle = True)
    model = gcn_bert0( t=0.4, adj_file='data/god_adj.pkl').cuda()
    model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load("./ELmodel/GCN_Supervised.ckpt")) # 重新训练
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=0.05,
                         t_total=len(train_loader) * epoch)
    total_batch = 0 # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_f1 = 0
    f1 = 0
    flag = False  # 记录是否很久没有效果提升
    m = 0
    for i in range(epoch):
        print('Epoch [{}/{}]'.format(i + 1, epoch))
        for sample in tqdm(train_loader):
            outputs = model(sample).float()
            label = sample[5]
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                f1 = evaluate(model, 'Dev')
                print('f1:{}'.format(f1))
                if f1 > dev_best_f1:
                    dev_best_f1 = f1
                    torch.save(model.state_dict(), './ELmodel/GCN_Supervised.ckpt')
            total_batch += 1
            print('train_loss:{} current_f1:{} best_f1:{}'.format(loss.item(), f1, dev_best_f1))
        if flag:
            break
    print(dev_best_f1)



def evaluate(model, phase):
    # 测试模型在测试集或训练集上的效果
    train_loader = DataLoader(EL_datasets(phase), batch_size=batchsize, shuffle=True)
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for sample in tqdm(train_loader):
            outputs = model(sample).float()
            _, predic = torch.max(outputs, dim=1)
            predic = predic.cpu().numpy()
            true = sample[5].cpu().squeeze().numpy()
            labels_all = np.append(labels_all,true)
            predict_all = np.append(predict_all,predic)
    f1 = metrics.f1_score(labels_all, predict_all)
    return f1


if __name__ == '__main__':
    train()
