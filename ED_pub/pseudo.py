import datetime
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
from sklearn import metrics
from model import gcn_bert0
import os
import random
from datasets import *

T1 = 10
T2 = 20
af = 3

def seed_torch(seed=1999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def alpha_weight(epoch):
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
         return ((epoch-T1) / (T2-T1))*af

def semisup_train(model, train_loader, unlabeled_loader,):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=0.0000001,
                         warmup=0.05,
                         t_total=len(train_loader) * 100)
    EPOCHS = 10
    dev_best_f1 = 0
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 10
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, x_unlabeled in tqdm(enumerate(unlabeled_loader)): # 使用伪标签
            # Forward Pass to get the pseudo labels
            model.eval()
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train()

            unlabeled_loss = alpha_weight(step) * F.cross_entropy(output_unlabeled, pseudo_labeled)
            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()

            # For every 50 batches train one epoch on labeled data
            if batch_idx % 5000 == 0:  # 使用标注数据进行训练
                # Normal training procedure
                for sample in tqdm(train_loader):
                    y_batch = sample[6].long()
                    output = model(sample)
                    labeled_loss = F.cross_entropy(output, y_batch)
                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()
                # Now we increment step by 1
                f1 = evaluate(model, 'Dev') # 临时测试
                if f1 > dev_best_f1:  # 如果提升
                    dev_best_f1 = f1
                    torch.save(model.state_dict(), './ELmodel/GCN_pseudo.ckpt')  # 直接采用模型覆盖的方式
                print('current_f1 = {}'.format(f1))
                step += 1
            batch_idx += 1

            print('best_f1 = {}'.format(dev_best_f1))
        print('Epoch: {} : Alpha Weight : {:.5f} | Test f1 : {:.5f}  '.format(epoch, alpha_weight(step), f1))
        model.train()

def evaluate(model, phase):
    # 测试模型在测试集或训练集上的效果
    train_loader = DataLoader(EL_datasets(phase), batch_size=6, shuffle=True)
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for sample in tqdm(train_loader):
            outputs = model(sample).float()
            _, predic = torch.max(outputs, dim=1)
            predic = predic.cpu().numpy()
            true = sample[6].cpu().squeeze().numpy()
            labels_all = np.append(labels_all, true)
            predict_all = np.append(predict_all, predic)
    # acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all)
    return f1


if __name__ == '__main__':
    # seed_torch(1999)
    train_loader = DataLoader(EL_datasets('Train'), batch_size=6, shuffle=True)
    unlabeled_loader = DataLoader(EL_Raw_datasets(), batch_size=6, shuffle=True)
    model = gcn_bert0(t=0.4, adj_file='data/god_adj.pkl')
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./ELmodel/GCN_Supervised.ckpt'))  # 载入与训练好的模型
    model.cuda()
    model.train()
    semisup_train(model, train_loader, unlabeled_loader)