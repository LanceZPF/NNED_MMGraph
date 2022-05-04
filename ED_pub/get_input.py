from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle
import json
import numpy as np
import os
from tqdm import tqdm
import pickle
from datasets import *
from utils import *


def get_Id_Text():
    # 用于获取id对应的text
    # 用于获取name对应的id
    id_text = dict()
    with open('./extract_data/Name2Text.json') as f:
        name_text = json.load(f)
    with open('./data/database_company.json') as f:
        json_data = json.load(f)
    for item in tqdm(json_data):
        name = item['entity_name']
        id = item['kb_id']
        id_text[id] = name_text[name]
    with open('./extract_data/Id2Text.json', 'w') as f:
        json.dump(id_text, f)


def get_Id_Name():
    with open('./data/database_company.json') as f:
        json_Data = json.load(f)
    id2name = dict()
    for item in json_Data:
        id2name[int(item['kb_id'])] = item['entity_name']
    with open('extract_data/Id2Name.json', 'w') as f:
        json.dump(id2name, f)

#name2id
def get_Name_Id():
    # 用于获取name对应的id
    name_id = dict()
    with open('./data/database_company.json') as f:
        json_data = json.load(f)
    for item in tqdm(json_data):
        name = item['entity_name']
        id = item['kb_id']
        name_id[name] = id
    with open('./extract_data/Name2Id.json', 'w') as f:
        json.dump(name_id, f)

#name2text
def get_Name_text(max_len=510,min_len=30):
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

# 获取实体知识库嵌入
def get_Entity_Embedding():
    # 将实体描述的嵌入向量先提取出来，然后进行保存，方便后来使用
    # 字典键为id号
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')  # 获得分词器
    model = BertModel.from_pretrained('./bert_pretrain')  # 获得Bert模型
    model.eval()
    model.to('cuda')
    vector = np.zeros((327, 768))
    names = []
    count = -1
    with open("./extract_data/Id2Text.json", 'r') as f:
        id_text = json.load(f)

    for i, id in enumerate(id_text):
        sum = torch.zeros(1, 768).cuda()
        num = 0
        text = id_text[id]
        tokenized_text = tokenizer.tokenize(text)  # 对文本进行分词
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
        with torch.no_grad():
            _, pooled = model(tokens_tensor)
        vector[int(id)] = pooled.cpu().numpy()
    np.save("./extract_data/Id2Vector.npy", vector)

#训练数据集预处理
def get_EL_Train_input():
    '''
    用于获取EL的输入
    inputs：{ids:[], begin:[], end:[], label:[]}
    恍然大悟：每一个mention就是一个sample，但是还有一点就是一个mention是一个名词，可能对应多个实体，那么不同的实体就对应不同的label，正确实体打上1，不正确的实体打上0
    最后一个问题：如果可以在实体库中链接成功，这种情况就打1
    知识库不需要扩充
    :return:inputs：ids1, ids2, mask1, mask2, begin, end, label, entity, mask3
    '''
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')  # 获得分词器
    with open('./extract_data/Name2Text.json') as f:
        Name2Text = json.load(f)
    inputs = {'ids1': [], 'ids2': [], 'mask1': [], 'mask2': [], 'begin': [], 'end': [], 'label': [], 'entity': [],
              'mask3': [], 'real_t': [], 'real_m': [], 'left_word1': [], 'left_word2': [], 'left_word3': [],
              'left_word4': [], 'right_word1': [], 'right_word2': [], 'right_word3': [], 'right_word4': []}
    with open('./data/train.json') as f:
        train_json = json.load(f)
    for item in tqdm(train_json):
        text = item['text']
        mention_data = item['lab_result']
        for men in mention_data:
            mention = men['mention']
            offset = men['offset']
            kb_id = men['kb_id']
            entity_text = Name2Text[mention]  # 这里要获取entity的文本描述
            # 先不管了，直接将两句话相加
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
            if kb_id == -1:
                inputs['label'].append(0)
            else:
                inputs['label'].append(1)

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

    # print(inputs['ids'][0])
    open_file = open('extract_data/ELTrainInputGCN.pkl', 'wb')
    pickle.dump(inputs, open_file)
    open_file.close()

#测试数据集预处理
def get_EL_Dev_input():
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')  # 获得分词器
    with open('./extract_data/Name2Text.json') as f:
        Name2Text = json.load(f)
    inputs = {'ids1': [], 'ids2': [], 'mask1': [], 'mask2': [], 'begin': [], 'end': [], 'label': [], 'entity': [],
              'mask3': [], 'real_t': [], 'real_m': [], 'left_word1': [], 'left_word2': [], 'left_word3': [],
              'left_word4': [], 'right_word1': [], 'right_word2': [], 'right_word3': [], 'right_word4': []}
    with open('./data/dev.json') as f:
        train_json = json.load(f)
    for item in tqdm(train_json):
        text = item['text']
        mention_data = item['lab_result']
        for men in mention_data:
            mention = men['mention']
            offset = men['offset']
            kb_id = men['kb_id']
            entity_text = Name2Text[mention]  # 这里要获取entity的文本描述
            # 先不管了，直接将两句话相加
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
            if kb_id == -1:
                inputs['label'].append(0)
            else:
                inputs['label'].append(1)

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
    # print(inputs['ids'][0])
    open_file = open('extract_data/ELDevInputGCN.pkl', 'wb')
    pickle.dump(inputs, open_file)
    open_file.close()
    # pd.to_pickle(inputs, 'extract_data/ELDevInput.pkl')


#为标注数据集预处理
def get_EL_Raw_input():
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')  # 获得分词器
    with open('./extract_data/Name2Text.json') as f:
        Name2Text = json.load(f)
    inputs = {'ids1': [], 'ids2': [], 'mask1': [], 'mask2': [], 'begin': [], 'end': [], 'label': [], 'entity': [],
              'mask3': [], 'real_t': [], 'real_m': [], 'left_word1': [], 'left_word2': [], 'left_word3': [],
              'left_word4': [], 'right_word1': [], 'right_word2': [], 'right_word3': [], 'right_word4': []}
    trie_obj = get_Trie()
    with open('./data/raw_texts.txt') as f:
        for i, line in tqdm(enumerate(f.readlines())):
            if i==50000:
                break
            entity, text = line.strip().split('\t')# 获取位置
            match_en = trie_obj.search_entity2way(text) # 获取实体
            for men in match_en:
                if men[0] == entity:
                    mention = men[0]
                    offset = men[1]
                    entity_text = Name2Text[mention]  # 这里要获取entity的文本描述
                    # 先不管了，直接将两句话相加
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
                    inputs['label'].append(-1)
                    # if kb_id == -1:
                    #     inputs['label'].append(0)
                    # else:
                    #     inputs['label'].append(1)

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
                        # print(inputs['ids'][0])
    pd.to_pickle(inputs, 'extract_data/ELRawInputGCN.pkl')

#为最终训练集进行预处理
def get_Result_input(phase):
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
    with open('./data/{}.txt'.format(phase)) as f:
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
    open_file = open('extract_data/EL{}InputFinally.pkl'.format(phase), 'wb')
    pickle.dump(inputs, open_file)
    with open('extract_data/text_id2text{}.json'.format(phase),'w') as f:
        json.dump(textid2text,f)
    open_file.close()



#为最终训练集进行预处理
def get_Result_input(phase):
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
    with open('./data/{}.txt'.format(phase)) as f:
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


                exit()

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
    open_file = open('extract_data/EL{}InputFinally.pkl'.format(phase), 'wb')
    pickle.dump(inputs, open_file)
    with open('extract_data/text_id2text{}.json'.format(phase),'w') as f:
        json.dump(textid2text,f)
    open_file.close()


if __name__ == '__main__':
   # get_Name_text()
   # get_Id_Name()
   # get_Id_Text()
   # get_Name_Id()
   #
   # get_Entity_Embedding()
   # # get_EL_Train_input()
   # # get_EL_Dev_input()
   # # get_EL_Raw_input()
   # print('训练数据生成成功')
    get_Result_input('test')