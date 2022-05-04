# 一些杂七杂八的函数
import json
from tqdm import tqdm

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




def softmax_binary(input):
    # 二分类的softmax
    first = input
