#######################
###QT中的一些汉函数功能
#########################
import json
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# from utils import *
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from model import *



class Trie:
    '''
    字典树
    有助于加快搜索速度
    '''

    def __init__(self):
        self.root = {}
        self.end = -1

    def insert(self, word):
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
        curNode[self.end] = True

    def search(self, word):
        curNode = self.root
        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]
        if not self.end in curNode:
            return False
        return True

    def startsWith(self, prefix):
        curNode = self.root
        for c in prefix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        return True

    def search_entity_reverse(self, text):
        '''
        反向实体搜索
        :param text: 一段文本
        :return: 文本中实体列表
        '''
        # text = text.lower()
        entitys = []
        i = len(text)
        while i > 0:
            e = i - 1
            t = 0
            flag = False
            while t <= 6:
                t += 1
                if self.search(text[e:i]):
                    entitys.append((text[e:i], e))
                    i = e
                    flag = True
                    break
                else:
                    e -= 1

            if flag == False:
                i -= 1
        return entitys

    def search_entity_forward(self, text):
        '''
        正向最大实体搜索
        :param text: 一段文本
        :return: 文本中实体列表
        '''
        # text = text.lower()
        entitys = []
        i = 0
        while i < len(text):
            e = i + 1
            param = self.startsWith(text[i:e])
            if param:
                en = text[i:e]
                while e <= len(text):
                    p = self.startsWith(text[i:e])
                    if p:
                        en = text[i:e]
                        e += 1
                    else:
                        break
                if self.search(en):
                    entitys.append((en, i))
                    i = e - 1
                    # i+=1
                else:
                    i += 1
            else:
                i += 1
        return entitys

    def search_entity2way(self, text):
        '''
        双向搜索
        :param text: 一段文本
        :return: 文本中实体列表
        '''
        # text = text.lower()
        match_list_forward = self.search_entity_forward(text)  # 获得正向匹配结果
        match_list_reverse = self.search_entity_reverse(text)

        return list(set(match_list_forward) | set(match_list_reverse))


def get_Trie(min_len=2):
    '''
    构建实体字典树，长度小于2的不插入
    :param min_len: 实体长度
    :return:
    '''
    trie_obj = Trie()
    with open('./data/company_2_code_sub.txt') as f:
        for index, en in enumerate(f.readlines()):
            if (index == 0):
                pass
            else:
                ee = en.strip().split('\t')[1]
                trie_obj.insert(ee)
    return trie_obj


def seq_padding(seq, max_len, value=0):
    x = [value] * max_len
    x[:len(seq)] = seq[:max_len]
    return x


def seq_padding_2(seq, max_len, value=0):
    if len(seq) < max_len:
        # 进行填充
        x = [value] * max_len
        x[:len(seq)] = seq[:max_len]
        attention_mask = [0] * max_len
        attention_mask[:len(seq)] = [1] * len(seq)
    else:
        # 裁减
        x = seq[:max_len]
        attention_mask = [1] * max_len
    return x, attention_mask




def get_text_id2text(phase):
    text_id2text = dict()
    with open('./data/{}.txt'.format(phase)) as f:
        for line in tqdm(f.readlines()):
            text_id, text = line.strip().split('\t')
            text_id = int(text_id)
            text_id2text[text_id] = text
    with open('./extract_data/text_id2text{}.json'.format(phase.capitalize()), 'w') as f:
        json.dump(text_id2text, f)


def get_id2name():
    with open('./data/database_company.json') as f:
        json_Data = json.load(f)
    id2name = dict()
    for item in json_Data:
        id2name[int(item['kb_id'])] = item['entity_name']
    with open('extract_data/Id2Name.json', 'w') as f:
        json.dump(id2name, f)


def get_len(text_lens, max_len=510, min_len=30):
    """
    戒断过长文本你的长度，小于30不在截断，大于30按比例截断
    :param text_lens: 列表形式 data 字段中每个 predicate+object 的长度
    :param max_len: 最长长度
    :param min_len: 最段长度
    :return: 列表形式 戒断后每个 predicate+object 保留的长度
            如 input：[638, 10, 46, 9, 16, 22, 10, 9, 63, 6, 9, 11, 34, 10, 8, 6, 6]
             output：[267, 10, 36, 9, 16, 22, 10, 9, 42, 6, 9, 11, 31, 10, 8, 6, 6]

    """
    new_len = [min_len]*len(text_lens)
    sum_len = sum(text_lens)
    del_len = sum_len - max_len
    del_index = []
    for i, l in enumerate(text_lens):
        if l > min_len:
            del_index.append(i)
        else:
            new_len[i]=l
    del_sum = sum([text_lens[i]-min_len for i in del_index])
    for i in del_index:
        new_len[i] = text_lens[i] - int(((text_lens[i]-min_len)/del_sum)*del_len) - 1
    return new_len

def name_text(max_len=510,min_len=30):
    with open("./data/database_company.json") as f:
        entity2text = {}
        json_data = json.load(f)
        for item in tqdm(json_data):
            name = item['entity_name']
            texts = []
            text = ''
            entity_id = item["entity_id"]
            entity_description = item["entity_description"]
            text = ""
            for data in entity_description:
                texts.append(data['predicate'] + ':' + data['object'] + ', ')
            text_lens = []
            for t in texts:
                text_lens.append(len(t))
            if sum(text_lens) < max_len:
                for t in texts:
                    text = text+t
            else:
                new_text_lens = get_len(text_lens,max_len=max_len, min_len=min_len)
                for t, l in zip(texts, new_text_lens):
                    text = text + t[:l]

            entity2text[name] = text
    with open("./extract_data/Name2Text.json", 'w') as f:
        json.dump(entity2text, f)


def process_single_text(text):
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')  # 获得分词器
    with open('./extract_data/Name2Text.json') as f:
        Name2Text = json.load(f)
    with open('./extract_data/Name2Id.json') as f:
        Name2Id = json.load(f)
    inputs = {'ids1': [], 'ids2': [], 'mask1': [], 'mask2': [], 'begin': [], 'end': [], 'label': [], 'entity': [],
              'mask3': [], 'real_t': [], 'real_m': [], 'left_word1': [], 'left_word2': [], 'left_word3': [],
              'left_word4': [], 'right_word1': [], 'right_word2': [], 'right_word3': [], 'right_word4': [],
              'text_id': [], 'offset': [], 'entity_id': []}
    trie_obj = get_Trie()
    textid2text = dict()
    match_en = trie_obj.search_entity2way(text)
    if len(match_en)==0:
        print('None')
        return False
    for men in match_en:
        mention = men[0]
        offset = men[1]
        entity_text = Name2Text[mention]
        mention = men[0]
        offset = men[1]
        entity_text = Name2Text[mention]
        ids_text1 = text
        ids_text2 = entity_text
        real_text = text + entity_text
        tokenized_text1 = tokenizer.tokenize(ids_text1)
        tokenized_text2 = tokenizer.tokenize(ids_text2)
        real_tt = tokenizer.tokenize(real_text)
        indices1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
        indices1 = [101] + indices1 + [102]
        indices1, mask1 = seq_padding_2(indices1, 400)
        indices1[-1] = 102
        inputs['ids1'].append(indices1)
        indices2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
        indices2 = [101] + indices2 + [102]
        indices2, mask2 = seq_padding_2(indices2, 400)
        indices2[-1] = 102
        inputs['ids2'].append(indices2)
        inputs['mask1'].append(mask1)
        inputs['mask2'].append(mask2)
        indices3 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(mention))
        indices3 = [101] + indices3 + [102]
        indices3, mask3 = seq_padding_2(indices3, 400)
        indices3[-1] = 102
        real_indices = tokenizer.convert_tokens_to_ids(real_tt)
        real_indices = [101] + real_indices + [102]
        real_indices, real_mask = seq_padding_2(real_indices, 400)
        real_indices[-1] = 102
        inputs['real_t'].append(real_indices)
        inputs['real_m'].append(real_mask)
        inputs['entity'].append(indices3)
        inputs['mask3'].append(mask3)
        begin = offset
        end = begin + len(mention) + 1
        Begin = np.zeros((400,))
        End = np.zeros((400,))
        Begin[begin] = 1
        End[end] = 1
        inputs['begin'].append(Begin)
        inputs['end'].append(End)
        inputs['label'].append(-1)  # 直接测试不在乎结果
        left_word1 = np.zeros((400,))
        left_word2 = np.zeros((400,))
        left_word3 = np.zeros((400,))
        left_word4 = np.zeros((400,))
        right_word1 = np.zeros((400,))
        right_word2 = np.zeros((400,))
        right_word3 = np.zeros((400,))
        right_word4 = np.zeros((400,))

        len_mention = end - begin

        left_word1[offset - 1] = 1
        left_word2[offset - 2] = 1
        left_word3[offset - 3] = 1
        left_word4[offset - 4] = 1
        right_word1[offset + len_mention + 1] = 1
        right_word2[offset + len_mention + 2] = 1
        right_word3[offset + len_mention + 3] = 1
        right_word4[offset + len_mention + 4] = 1

        inputs['left_word1'].append(left_word1)
        inputs['left_word2'].append(left_word2)
        inputs['left_word3'].append(left_word3)
        inputs['left_word4'].append(left_word4)
        inputs['right_word1'].append(right_word1)
        inputs['right_word2'].append(right_word2)
        inputs['right_word3'].append(right_word3)
        inputs['right_word4'].append(right_word4)
        # inputs['text_id'].append(int(text_id))  # 新增
        inputs['offset'].append(offset)  # 新增
        inputs['entity_id'].append(Name2Id[mention])  # 新增


    # print(inputs['ids'][0])
    with open('intermediate/ELSingleInput.pkl', 'wb') as f:
        pickle.dump(inputs, f)
    with open('intermediate/text_id2textSingle.json', 'w') as f:
        json.dump(textid2text, f)
    return True


class UI_Single_datasets(Dataset):
    def __init__(self):
        self.inputs = pd.read_pickle('intermediate/ELSingleInput.pkl')
    def __getitem__(self, index):
        # 首先得到ids mask begin end labels
        # 在将他们转化成tensor的形式
        ids1 = torch.tensor(self.inputs['ids1'][index]).cuda()
        ids2 = torch.tensor(self.inputs['ids2'][index]).cuda()
        mask1 = torch.tensor(self.inputs['mask1'][index]).cuda()
        mask2 = torch.tensor(self.inputs['mask2'][index]).cuda()
        mask3 = torch.tensor(self.inputs['mask3'][index]).cuda()
        begin = torch.tensor(self.inputs['begin'][index]).cuda()
        end = torch.tensor(self.inputs['end'][index]).cuda()
        label = torch.tensor(self.inputs['label'][index]).cuda()
        entity = torch.tensor(self.inputs['entity'][index]).cuda()
        realt = torch.tensor(self.inputs['real_t'][index]).cuda()
        realm = torch.tensor(self.inputs['real_m'][index]).cuda()
        left_word1 = torch.tensor(self.inputs['left_word1'][index]).cuda()
        left_word2 = torch.tensor(self.inputs['left_word2'][index]).cuda()
        left_word3 = torch.tensor(self.inputs['left_word3'][index]).cuda()
        left_word4 = torch.tensor(self.inputs['left_word4'][index]).cuda()
        right_word1 = torch.tensor(self.inputs['right_word1'][index]).cuda()
        right_word2 = torch.tensor(self.inputs['right_word2'][index]).cuda()
        right_word3 = torch.tensor(self.inputs['right_word3'][index]).cuda()
        right_word4 = torch.tensor(self.inputs['right_word4'][index]).cuda()
        offset = self.inputs['offset'][index]
        entity_id = self.inputs['entity_id'][index]

        sample = [ids1, ids2, mask1, mask2, begin, end, label, entity, mask3, realt, realm, left_word1, left_word2,  left_word3, left_word4, right_word1, right_word2, right_word3, right_word4, offset,entity_id]
        return sample

    def __len__(self):
        return len(self.inputs['real_t'])


def UI_Single_Process(text):
    with open('./extract_data/Id2Name.json') as f:
        Id2Name = json.load(f)
    index = {'entity_name': [], 'offset': [], 'entity_id': []}
    if process_single_text(text):
        process_dataloader = DataLoader(UI_Single_datasets(),batch_size=1, shuffle=True)
        model = gcn_bert0(t=0.4, adj_file='./data/god_adj.pkl').cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('./ELmodel/GCN_pseudo.ckpt')) # 改模型地址
        model.eval().cuda()
        predict_all = np.array([], dtype=int)
        confidence_all = np.array([], dtype=float)

        with torch.no_grad():
            for i, sample in enumerate(process_dataloader):
                index['offset'] += sample[19].numpy().tolist()  # 偏移量
                # index['entity_id'] += list(sample[20])  # id
                outputs = model(sample).float()
                confidence, predic = torch.max(outputs, dim=1)  # 增加了一个confidence
                predic = predic.cpu().numpy()
                confidence = confidence.cpu().numpy()
                predict_all = np.append(predict_all, predic)
                confidence_all = np.append(confidence_all, confidence)
                index['entity_name'].append(Id2Name[list(sample[20])[0]])
                if predic[0]==1:
                    index['entity_id'] += list(sample[20])
                else:
                    index['entity_id'].append(-1)

        return index
    else:
        return index  # 返回没有正常的处理
    # np.save('intermediate/predictSingle.npy', predict_all)  # 两者通过索引进行对应
    # np.save('intermediate/confidenceSingle.npy', confidence_all)
    # with open('intermediate/index.json', 'w') as f:
    #     json.dump(index, f)
    # process(phase)
    # sort()
    # return f1()


def UI_get_Result_input(filename):
    ######################################
    ###############################
    ########################
    # 用来生成结果的，主要在输入上增加了text_id  offset entity_id
    ##################
    ###############
    ##########
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')  # 获得分词器
    with open('./extract_data/Name2Text.json') as f:
        Name2Text = json.load(f)
    with open('./extract_data/Name2Id.json') as f:
        Name2Id = json.load(f)
    inputs = {'ids1': [], 'ids2': [], 'mask1': [], 'mask2': [], 'begin': [], 'end': [], 'label': [], 'entity': [],
              'mask3': [], 'real_t': [], 'real_m': [], 'left_word1': [], 'left_word2': [], 'left_word3': [],
              'left_word4': [], 'right_word1': [], 'right_word2': [], 'right_word3': [], 'right_word4': [],'text_id':[],'offset':[],'entity_id':[]}
    trie_obj = get_Trie()
    textid2text=dict()
    with open(filename) as f:
        for line in tqdm(f.readlines()):
            text_id, text = line.strip().split('\t')
            textid2text[text_id] = text
            match_en = trie_obj.search_entity2way(text)
            for men in match_en:
                mention = men[0]
                offset = men[1]
                entity_text = Name2Text[mention]
                ids_text1 = text
                ids_text2 = entity_text
                real_text = text + entity_text
                tokenized_text1 = tokenizer.tokenize(ids_text1)
                tokenized_text2 = tokenizer.tokenize(ids_text2)
                real_tt = tokenizer.tokenize(real_text)
                indices1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
                indices1 = [101] + indices1 + [102]
                indices1, mask1 = seq_padding_2(indices1, 400)
                indices1[-1] = 102
                inputs['ids1'].append(indices1)
                indices2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
                indices2 = [101] + indices2 + [102]
                indices2, mask2 = seq_padding_2(indices2, 400)
                indices2[-1] = 102
                inputs['ids2'].append(indices2)
                inputs['mask1'].append(mask1)
                inputs['mask2'].append(mask2)
                indices3 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(mention))
                indices3 = [101] + indices3 + [102]
                indices3, mask3 = seq_padding_2(indices3, 400)
                indices3[-1] = 102
                real_indices = tokenizer.convert_tokens_to_ids(real_tt)
                real_indices = [101] + real_indices + [102]
                real_indices, real_mask = seq_padding_2(real_indices, 400)
                real_indices[-1] = 102
                inputs['real_t'].append(real_indices)
                inputs['real_m'].append(real_mask)
                inputs['entity'].append(indices3)
                inputs['mask3'].append(mask3)
                begin = offset
                end = begin + len(mention) + 1
                Begin = np.zeros((400,))
                End = np.zeros((400,))
                Begin[begin] = 1
                End[end] = 1
                inputs['begin'].append(Begin)
                inputs['end'].append(End)
                inputs['label'].append(-1) # 直接测试不在乎结果
                left_word1 = np.zeros((400,))
                left_word2 = np.zeros((400,))
                left_word3 = np.zeros((400,))
                left_word4 = np.zeros((400,))
                right_word1 = np.zeros((400,))
                right_word2 = np.zeros((400,))
                right_word3 = np.zeros((400,))
                right_word4 = np.zeros((400,))

                len_mention = end - begin

                left_word1[offset - 1] = 1
                left_word2[offset - 2] = 1
                left_word3[offset - 3] = 1
                left_word4[offset - 4] = 1
                right_word1[offset + len_mention + 1] = 1
                right_word2[offset + len_mention + 2] = 1
                right_word3[offset + len_mention + 3] = 1
                right_word4[offset + len_mention + 4] = 1

                inputs['left_word1'].append(left_word1)
                inputs['left_word2'].append(left_word2)
                inputs['left_word3'].append(left_word3)
                inputs['left_word4'].append(left_word4)
                inputs['right_word1'].append(right_word1)
                inputs['right_word2'].append(right_word2)
                inputs['right_word3'].append(right_word3)
                inputs['right_word4'].append(right_word4)
                inputs['text_id'].append(int(text_id)) # 新增
                inputs['offset'].append(offset)     #新增
                inputs['entity_id'].append(Name2Id[mention]) # 新增


    # print(inputs['ids'][0])
    open_file = open('extract_data/ELGroupInput.pkl', 'wb')
    pickle.dump(inputs, open_file)
    with open('extract_data/text_id2textGroup.json','w') as f:
        json.dump(textid2text,f)
    open_file.close()


class UI_EL_Result_datasets(Dataset):
    def __init__(self):
        self.inputs = pd.read_pickle('extract_data/ELGroupInput.pkl')
        pass
    def __getitem__(self, index):
        # 首先得到ids mask begin end labels
        # 在将他们转化成tensor的形式
        ids1 = torch.tensor(self.inputs['ids1'][index]).cuda()
        ids2 = torch.tensor(self.inputs['ids2'][index]).cuda()
        mask1 = torch.tensor(self.inputs['mask1'][index]).cuda()
        mask2 = torch.tensor(self.inputs['mask2'][index]).cuda()
        mask3 = torch.tensor(self.inputs['mask3'][index]).cuda()
        begin = torch.tensor(self.inputs['begin'][index]).cuda()
        end = torch.tensor(self.inputs['end'][index]).cuda()
        label = torch.tensor(self.inputs['label'][index]).cuda()
        entity = torch.tensor(self.inputs['entity'][index]).cuda()
        realt = torch.tensor(self.inputs['real_t'][index]).cuda()
        realm = torch.tensor(self.inputs['real_m'][index]).cuda()
        left_word1 = torch.tensor(self.inputs['left_word1'][index]).cuda()
        left_word2 = torch.tensor(self.inputs['left_word2'][index]).cuda()
        left_word3 = torch.tensor(self.inputs['left_word3'][index]).cuda()
        left_word4 = torch.tensor(self.inputs['left_word4'][index]).cuda()
        right_word1 = torch.tensor(self.inputs['right_word1'][index]).cuda()
        right_word2 = torch.tensor(self.inputs['right_word2'][index]).cuda()
        right_word3 = torch.tensor(self.inputs['right_word3'][index]).cuda()
        right_word4 = torch.tensor(self.inputs['right_word4'][index]).cuda()
        text_id = self.inputs['text_id'][index]
        offset = self.inputs['offset'][index]
        entity_id = self.inputs['entity_id'][index]

        sample = [ids1, ids2, mask1, mask2, begin, end, label, entity, mask3, realt, realm, left_word1, left_word2,  left_word3, left_word4, right_word1, right_word2, right_word3, right_word4, text_id,offset,entity_id]
        return sample

    def __len__(self):
        return len(self.inputs['real_t'])


def process():
    # 处理预测出来的结果
    # 可以获得label以及对应的
    result = dict()
    result['team_name'] = '想吃海底捞'
    result['submit_result'] = []
    with open('./extract_data/text_id2textGroup.json') as f:
        text_id2text = json.load(f)
    with open('./intermediate/index.json') as f:
        json_data = json.load(f)
    with open('./extract_data/Id2Name.json') as f:
        Id2Name = json.load(f)
    predict = np.load('intermediate/predict.npy').tolist()
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
    with open('./result/UI_result.json','w',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def sort():
    # 对结果进行排序， 就是将text_id按从低到高的顺序进行排列
    with open('./result/UI_result.json') as f:
        result = json.load(f)
    submit_result = result['submit_result']
    sorted_result = []
    for i in range(len(submit_result)):
        for item in submit_result:
            if item['text_id'] == i:
                sorted_result.append(item)
                break
    result['submit_result'] = sorted_result
    with open('./result/UI_result.json','w',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def result(filename):
    # 直接集成，生存所有结果。
    # 只需修改注释处的代码即可
    UI_get_Result_input(filename)
    train_loader = DataLoader(UI_EL_Result_datasets(), batch_size=16, shuffle=True)  # 改数据集类
    model = gcn_bert0(t=0.4, adj_file='data/god_adj.pkl')         # 改模型名称
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./ELmodel/GCN_Supervised.ckpt')) # 改模型地址

    model.eval().cuda()
    predict_all = np.array([], dtype=int)
    confidence_all = np.array([], dtype=float)
    index = {'text_id': [], 'offset': [], 'entity_id': []}
    with torch.no_grad():
        for i,sample in enumerate(tqdm(train_loader)):
            if i==1:
                    break
            index['text_id'] += sample[19].numpy().tolist()  # 句子位置
            index['offset'] += sample[20].numpy().tolist()  # 偏移量
            index['entity_id'] += list(sample[21])           # id
            outputs = model(sample).float()
            confidence, predic = torch.max(outputs, dim=1)  # 增加了一个confidence
            predic = predic.cpu().numpy()
            confidence = confidence.cpu().numpy()
            predict_all = np.append(predict_all, predic)
            confidence_all = np.append(confidence_all, confidence)
    np.save('intermediate/predict.npy', predict_all) # 两者通过索引进行对应
    np.save('intermediate/confidence.npy', confidence_all)
    with open('intermediate/index.json', 'w') as f:
        json.dump(index, f)
    process()
    sort()

if __name__ == '__main__':
    result('./data/test.txt')