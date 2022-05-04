from torch.utils.data import Dataset
import torch
import pickle
import pandas as pd

# 训练用datasets
class EL_datasets(Dataset):
    def __init__(self, phase):
        open_file = open('./extract_data/EL{}InputGCN.pkl'.format(phase),'rb')
        self.inputs = pickle.load(open_file)
        open_file.close()
        #self.inputs = pd.read_pickle('./extract_data/EL{}Input.pkl'.format(phase))

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
        sample = [ids1, ids2, mask1, mask2, begin, end, label, entity, mask3, realt, realm, left_word1, left_word2,  left_word3, left_word4, right_word1, right_word2, right_word3, right_word4]
        return sample

    def __len__(self):
        return len(self.inputs['real_t'])

# 伪标签datasets
class EL_Raw_datasets(Dataset):
    def __init__(self):
        self.inputs = pd.read_pickle('./extract_data/ELRawInputGCN.pkl')

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
        sample = [ids1, ids2, mask1, mask2, begin, end, label, entity, mask3, realt, realm, left_word1, left_word2,  left_word3, left_word4, right_word1, right_word2, right_word3, right_word4]
        return sample

    def __len__(self):
        return len(self.inputs['real_t'])


# 测试用datasets
class EL_Result_datasets(Dataset):
    def __init__(self, phase):
        self.inputs = pd.read_pickle('extract_data/EL{}InputFinally.pkl'.format(phase))
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


