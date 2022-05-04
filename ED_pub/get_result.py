import torch.utils.data.dataloader
import torch
import pickle
import json
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from model import EL_model,  EL_model_version2, EL_model_version3
from datasets import *
from model import gcn_bert0
from utils import *
import copy
'''
用于最终提交结果的生成
直接拿到输入的文件名，即可进行输入配置
队伍：想吃海底捞
'''


def process(phase):
    # 处理预测出来的结果
    # 可以获得label以及对应的
    result = dict()
    result['team_name'] = '想吃海底捞'
    result['submit_result'] = []
    with open('./extract_data/text_id2text{}.json'.format(phase)) as f:
        text_id2text = json.load(f)
    with open('./result/index.json') as f:
        json_data = json.load(f)
    with open('./extract_data/Id2Name.json') as f:
        Id2Name = json.load(f)
    predict = np.load('result/predict.npy').tolist()
    confidence = np.load('result/confidence.npy').tolist()
    for i in range(len(predict)): # 对所有预测结果进行
        flag = False  # 没有找到这个text
        text_id = int(json_data['text_id'][i])
        text = text_id2text[str(json_data['text_id'][i])]
        for item in result['submit_result']:
            if item['text_id'] == text_id:
                D = dict()
                D['mention'] = Id2Name[json_data['entity_id'][i]]
                D['offset'] = json_data['offset'][i]
                if predict[i] == 1:
                    D['kb_id'] = int(json_data['entity_id'][i])
                else:
                    D['kb_id'] = -1
                D['confidence'] = confidence[i]
                item['mention_result'].append(D)
                flag = True
                break
        if flag != True:
            d = dict()
            d['text_id'] = text_id
            d['text'] = text
            d['mention_result'] = []
            D = dict()
            D['mention'] = Id2Name[json_data['entity_id'][i]]
            D['offset'] = json_data['offset'][i]
            if predict[i] == 1:
                D['kb_id'] = int(json_data['entity_id'][i])
            else:
                D['kb_id'] = -1
            D['confidence'] = confidence[i]
            d['mention_result'].append(D)
            result['submit_result'].append(d)
    with open('./result/result.json','w',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def sort():
    # 对结果进行排序， 就是将text_id按从低到高的顺序进行排列
    with open('./result/result.json') as f:
        result = json.load(f)
    submit_result = result['submit_result']
    sorted_result = []
    for i in range(len(submit_result)):
        for item in submit_result:
            if item['text_id'] == i:
                sorted_result.append(item)
                break
    result['submit_result'] = sorted_result
    with open('./result/result.json','w',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


# def f1():
#     # 用于计算f1值
#     # 首先计算查准率precision
#     # 然后计算查全率recall
#     with open('./result/result.json') as f:
#         result = json.load(f)
#     with open('./data/dev.json') as f:
#         dev_data = json.load(f)
#     # 计算查准率
#     zhenque = 0
#     yuce = 0
#     gt = 0
#     error_list = []
#     submit_reuslt = result['submit_result']
#     for i, item in enumerate(submit_reuslt):
#         true_result = dev_data[i]['lab_result']
#         gt += len(true_result)
#         predict_result = item['mention_result']
#         for mention in predict_result:
#             yuce += 1
#             mention_temp = copy.deepcopy(mention)
#             mention_temp.pop('confidence')
#             if mention_temp in true_result:
#                 zhenque+=1
#             else:
#                 error_list.append(item)
#
#     with open('abc.json','w') as f:
#         json.dump(error_list, f)
#     precision = zhenque / yuce
#     recall = zhenque / gt
#     f1 = 2 * precision * recall / (precision + recall)
#     return f1


# def predict():
#     # 直接预测即可
#     train_loader = DataLoader(Evaluate_datasets_version3(), batch_size=16, shuffle=True)
#     model = EL_model_version3()
#     model.load_state_dict(torch.load('./ELmodel/ELversion2 2020-02-28 18:37:55.ckpt'))
#     model.eval()
#     model.cuda()
#     predict_all = np.array([], dtype=int)
#     confidence_all = np.array([], dtype=float)
#     index = {'text_id':[], 'offset':[], 'entity_id':[]}
#     with torch.no_grad():
#         for sample in tqdm(train_loader):
#             index['text_id'] += sample[0].numpy().tolist()
#             index['offset'] += sample[5].numpy().tolist()
#             index['entity_id'] += list(sample[6])
#             outputs = model(sample).float()
#
#             confidence, predic = torch.max(outputs, dim=1)  # 增加了一个confidence
#             predic = predic.cpu().numpy()
#             confidence = confidence.cpu().numpy()
#             predict_all = np.append(predict_all, predic)
#             confidence_all = np.append(confidence_all, confidence)
#     np.save('result/predict.npy', predict_all,)
#     np.save('result/confidence.npy', confidence_all)
#     with open('./result/index.json', 'w') as f:
#         json.dump(index, f)


def result(phase):
    # 直接集成，生存所有结果。
    # 只需修改注释处的代码即可
    train_loader = DataLoader(EL_Result_datasets(phase), batch_size=8, shuffle=True)  # 改数据集类
    model = gcn_bert0(t=0.4, adj_file='data/god_adj.pkl')         # 改模型名称
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./ELmodel/GCN_pseudo.ckpt')) # 改模型地址

    model.eval().cuda()
    predict_all = np.array([], dtype=int)
    confidence_all = np.array([], dtype=float)
    index = {'text_id': [], 'offset': [], 'entity_id': []}
    with torch.no_grad():
        for i,sample in enumerate(tqdm(train_loader)):
            # if i==10:
            #         break
            index['text_id'] += sample[19].numpy().tolist()  # 句子位置
            index['offset'] += sample[20].numpy().tolist()  # 偏移量
            index['entity_id'] += list(sample[21])           # id
            outputs = model(sample).float()
            confidence, predic = torch.max(outputs, dim=1)  # 增加了一个confidence
            predic = predic.cpu().numpy()
            confidence = confidence.cpu().numpy()
            predict_all = np.append(predict_all, predic)
            confidence_all = np.append(confidence_all, confidence)
    np.save('result/predict.npy', predict_all) # 两者通过索引进行对应
    np.save('result/confidence.npy', confidence_all)
    with open('./result/index.json', 'w') as f:
        json.dump(index, f)
    process(phase)
    sort()


# def Evaluate(model):
#     # 这个用来测试结果
#     # 直接预测即可
#     train_loader = DataLoader(Evaluate_datasets('Dev'), batch_size=16, shuffle=True)
#     model.eval()
#     predict_all = np.array([], dtype=int)
#     index = {'text_id': [], 'offset': [], 'entity_id': []}
#     with torch.no_grad():
#         for sample in tqdm(train_loader):
#             index['text_id'] += sample[6].numpy().tolist()
#             index['offset'] += sample[4].numpy().tolist()
#             index['entity_id'] += list(sample[5])
#             outputs = model(sample).float()
#             _, predic = torch.max(outputs, dim=1)
#             predic = predic.cpu().numpy()
#             predict_all = np.append(predict_all, predic)
#     np.save('result/predict.npy', predict_all)
#     with open('./result/index.json', 'w') as f:
#         json.dump(index, f)
#     process()
#     sort()
#     return f1()


if __name__ == '__main__':
    print(result('test'))

    pass


