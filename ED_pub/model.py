import torch
import numpy as np
from torch.utils.data import dataloader
from torch.utils.data import dataset
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
import json
from util import *


class NER_model(nn.Module):
    def __init__(self):
        super(NER_model, self).__init__()
        self.entity_embedding = torch.from_numpy(np.load('./extract_data/Id2Vector.npy')).cuda()
        self.bert = BertModel.from_pretrained('./bert_pretrain')  # bert预训练
        self.conv = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.linear = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, sample):
        ids = sample[0]
        mask = sample[1]
        begin = sample[2]
        end = sample[3]
        entity = sample[5]
        with torch.no_grad():
            encoded, clsout = self.bert(ids, attention_mask=mask)
        bert_out = encoded[11]  # torch.Size([8, 400, 768])
        begin_vec = bert_out * begin.unsqueeze(-1)
        begin_vec = torch.sum(begin_vec, dim=1)  # [8, 768]   begin [8,400]
        end_vec = bert_out * end.unsqueeze(-1)
        end_vec = torch.sum(end_vec, dim=1)
        entity_vec = self.entity_embedding * entity.unsqueeze(-1)
        entity_vec = torch.sum(entity_vec, dim=1)
        output = torch.cat((clsout.unsqueeze(1).float(), begin_vec.unsqueeze(1).float(), end_vec.unsqueeze(1).float(),
                            entity_vec.unsqueeze(1).float()),
                           dim=1)
        output = output.unsqueeze(-1)
        output = self.conv(output).squeeze(-1)
        output = self.linear(output)
        output = torch.sigmoid(output)
        output = F.softmax(output, dim=-1).squeeze(1)
        return output


class ner_model(nn.Module):
    # ner模型
    # 输入是一句话，输出就是一个label，就是对候选实体进行打label
    def __init__(self):
        super(ner_model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')
        self.bert = BertModel.from_pretrained('./bert_pretrain')
        self.bert.cuda()
        self.trie_obj = get_Trie()
        with open('./extract_data/entity_vector_name.json') as f:
            self.entity_vector_name = json.load(f)
        self.entity_vector = np.load('./extract_data/ER_entity_embedding.npy')  # 存有实体名称的词嵌入
        self.entity_vector = torch.from_numpy(self.entity_vector).to('cuda')
        self.conv = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, ).cuda()
        self.linear = torch.nn.Linear(in_features=768, out_features=2).cuda()

    def forward(self, sample):
        # step1 得到候选实体
        # 转入到gpu其实在外面就应该设计好了
        ids = sample[0].to('cuda')
        begin = sample[1].to('cuda')
        end = sample[2].to('cuda')
        labels = sample[3].to('cuda')
        entity_id = sample[4].to('cuda')
        match_en = sample[5]
        # 先获得测试文本的bert输出pooled
        with torch.no_grad():
            encoded, pooled = self.bert(ids)
        pass
        bert_input = encoded[11]
        # begin_numpy = begin.numpy()
        # begin = begin.unsqueeze(axis=-1)
        # begin_onehot = np.eye(52)[begin_numpy]
        # begin_onehot = np.logical_and(begin_onehot,begin_numpy)

        # bert_begin为开始位置的向量
        # bert_end位置为结束位置的向量
        bert_begin = bert_input.unsqueeze(axis=1).to('cuda')
        begin = begin.unsqueeze(axis=-1)
        bert_begin = begin * bert_begin
        bert_begin = bert_begin.sum(axis=-2)

        bert_end = bert_input.unsqueeze(axis=1).to('cuda')
        end = end.unsqueeze(axis=-1)
        bert_end = end * bert_end
        bert_end = bert_end.sum(axis=-2)
        # 实体的向量
        # 关键在于获取相应的实体词嵌入
        entity_id = entity_id.unsqueeze(-1)  # [8*13*328*1]
        entity_embedding = entity_id * self.entity_vector
        entity_embedding = entity_embedding.sum(axis=-2)

        # 最后要获得text文本的实体，其实就是前面的pooled

        # bert_begin [8*13*768]
        # bert_end [8*13*768]
        # entity_embedding [8*13*768]
        # pooled [8*768]

        pooled = pooled.unsqueeze(-2)
        f = lambda x: x.unsqueeze(-2)
        pooled = f(pooled)
        # pooled = f(pooled)
        bert_begin = f(bert_begin)
        bert_end = f(bert_end)
        entity_embedding = f(entity_embedding)
        pooled = pooled.repeat(1, 12, 1, 1)
        # a = torch.cat((pooled.float(),bert_begin.float()), axis = -2)
        a = torch.cat((pooled.float(), bert_begin.float(), bert_end.float(), entity_embedding.float()), axis=-2)

        a = a.view((-1, 4, 768, 1)).float()
        a = self.conv(a)
        a = a.view((-1, 12, 768))  # 第二个1修改过
        out = self.linear(a)
        out = torch.sigmoid(out)
        out = F.softmax(out, dim=-1)
        # out = out.view(-1,13,1)
        # out = out.unsqueeze(axis = -1)
        out = out.view(-1, 2)
        pass
        # 获取开始与结束位置的bert
        # begin_vec = encoded[11][begin]
        return out


class EL_model(nn.Module):
    # EL模型
    # 最基础的模型
    def __init__(self):
        super(EL_model, self).__init__()  # 这句话很重要
        self.bert = BertModel.from_pretrained('./bert_pretrain')  # bert预训练
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.linear = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, sample):
        # 获得text
        ids = sample[0]
        mask = sample[1]
        begin = sample[2]
        end = sample[3]
        with torch.no_grad():
            encoded, clsout = self.bert(ids, attention_mask=mask)
        bert_out = encoded[11]  # torch.Size([8, 400, 768])
        begin_vec = bert_out * begin.unsqueeze(-1)
        begin_vec = torch.sum(begin_vec, dim=1)  # [8, 768]   begin [8,400]
        end_vec = bert_out * end.unsqueeze(-1)
        end_vec = torch.sum(end_vec, dim=1)
        output = torch.cat((clsout.unsqueeze(1).float(), begin_vec.unsqueeze(1).float(), end_vec.unsqueeze(1).float()),
                           dim=1)
        output = output.unsqueeze(-1)
        output = self.conv(output).squeeze(-1)
        output = F.relu(output)  # 新加了一个relu
        output = self.linear(output)
        output = torch.sigmoid(output)
        output = F.softmax(output, dim=-1).squeeze(1)
        return output


class EL_model_version15(nn.Module):
    # EL模型
    def __init__(self):
        super(EL_model_version15, self).__init__()  # 这句话很重要
        self.bert = BertModel.from_pretrained('./bert_pretrain')  # bert预训练
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.linear = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, sample):
        # 获得text
        ids = sample[0]
        mask = sample[1]
        begin = sample[2]
        end = sample[3]

        encoded, clsout = self.bert(ids, attention_mask=mask)
        bert_out = encoded[11]  # torch.Size([8, 400, 768])
        begin_vec = bert_out * begin.unsqueeze(-1)
        begin_vec = torch.sum(begin_vec, dim=1)  # [8, 768]   begin [8,400]
        end_vec = bert_out * end.unsqueeze(-1)
        end_vec = torch.sum(end_vec, dim=1)
        output = torch.cat((clsout.unsqueeze(1).float(), begin_vec.unsqueeze(1).float(), end_vec.unsqueeze(1).float()),
                           dim=1)
        output = output.unsqueeze(-1)
        output = self.conv(output).squeeze(-1)
        output = F.relu(output)  # 新加了一个relu
        output = self.linear(output)
        output = torch.sigmoid(output)
        output = F.softmax(output, dim=-1).squeeze(1)
        return output


class EL_model_version2(nn.Module):
    # EL模型
    # 改动如下
    # 1 将拼接从通道变为，长度拼接，也就是将矩阵变为向量
    # 1 将卷积变为了全连接
    # 1 然后通过
    # 2 bert采用最后倒数第二层
    def __init__(self):
        super(EL_model_version2, self).__init__()  # 这句话很重要
        self.bert = BertModel.from_pretrained('./bert_pretrain')  # bert预训练
        self.linear1 = torch.nn.Linear(in_features=768 * 3, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=2)
        self.drop = nn.Dropout(0.15)

    def forward(self, sample):
        # 获得text
        ids = sample[0]
        mask = sample[1]
        begin = sample[2]
        end = sample[3]
        with torch.no_grad():
            encoded, clsout = self.bert(ids, attention_mask=mask)
        bert_out = encoded[10]  # torch.Size([8, 400, 768])
        begin_vec = bert_out * begin.unsqueeze(-1)
        begin_vec = torch.sum(begin_vec, dim=1)  # [8, 768]   begin [8,400]
        end_vec = bert_out * end.unsqueeze(-1)
        end_vec = torch.sum(end_vec, dim=1)
        output = torch.cat((clsout.float(), begin_vec.float(), end_vec.float()), dim=-1)
        output = self.linear1(output)
        output = self.drop(output)
        output = F.relu(output)  # 新加了一个relu
        output = self.linear2(output)
        output = torch.sigmoid(output)
        output = F.softmax(output, dim=-1).squeeze(1)
        return output


class EL_model_version3(nn.Module):
    # 比较大的版本更新，采用了文本相似度的方式来进行判度。
    def __init__(self):
        super(EL_model_version3, self).__init__()  # 这句话很重要
        self.bert = BertModel.from_pretrained('./bert_pretrain')  # bert预训练
        self.classifier = torch.nn.Linear(in_features=768, out_features=2)


    def forward(self, sample):
        ids = sample[0]
        mask = sample[1]
        seg = sample[2]
        _, clsout = self.bert(input_ids=ids, token_type_ids=seg, attention_mask=mask)
        outputs = self.classifier(clsout)
        return outputs


class EL_model_version4(nn.Module):
    def __init__(self):
        super(EL_model_version4, self).__init__()
        self.bert = BertModel.from_pretrained('./bert_pretrain')  # bert预训练
        self.linear1 = torch.nn.Linear(in_features=768 * 3, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=2)
        self.drop = nn.Dropout(0.15)

    def forward(self, sample):
        # 获得text
        ids = sample[0]
        mask = sample[1]
        seg = sample[2]
        begin = sample[3]
        end = sample[4]
        with torch.no_grad():
            encoded, clsout = self.bert(ids, attention_mask=mask, token_type_ids = seg)
        bert_out = encoded[10]  # torch.Size([8, 400, 768])
        begin_vec = bert_out * begin.unsqueeze(-1)
        begin_vec = torch.sum(begin_vec, dim=1)  # [8, 768]   begin [8,400]
        end_vec = bert_out * end.unsqueeze(-1)
        end_vec = torch.sum(end_vec, dim=1)
        output = torch.cat((clsout.float(), begin_vec.float(), end_vec.float()), dim=-1)
        output = self.linear1(output)
        output = self.drop(output)
        output = F.relu(output)  # 新加了一个relu
        output = self.linear2(output)
        output = torch.sigmoid(output)
        output = F.softmax(output, dim=-1).squeeze(1)
        return output


# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNBert(nn.Module):
    def __init__(self, model, in_channel=768, t=0, adj_file=None):
        super(GCNBert, self).__init__()
        self.entity_embedding = torch.from_numpy(np.load('./extract_data/Id2Vector.npy')).cuda()
        self.bert = model

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 768)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(13, t, adj_file)
        self.A = Parameter(_adj.float())

        self.linear = nn.Sequential(
            torch.nn.Linear(in_features=1536, out_features=768),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=768, out_features=384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=384, out_features=2))

    def forward(self, feature):
        '''
        for n in range(num):
            for m in range(num):
                feature[n][m] = self.bert(feature[n])
                feature[n][m] = self.pooling(feature[n])
                feature[n][m] = feature.view(feature[n].size(0), -1)
        '''
        tex = feature[9]
        mas = feature[10]

        enco, clso = self.bert(tex, attention_mask=mas)

        # inherited from ying
        ids1 = feature[0]
        ids2 = feature[1]
        mask1 = feature[2]
        mask2 = feature[3]
        begin = feature[4]
        end = feature[5]
        entity = feature[7]
        mask3 = feature[8]

        with torch.no_grad():
            encoded, clsout = self.bert(ids1, attention_mask=mask1)
            encoded2, e_out = self.bert(entity, attention_mask=mask3)  # representation of entity
            encoded1, et_out = self.bert(ids2, attention_mask=mask2)  # representation of entity describtion text

        bert_out = enco[11]  # representation of train text
        # print(begin.unsqueeze(-1).shape)
        begin_vec = bert_out.float() * begin.unsqueeze(-1).float()
        begin_vec = torch.sum(begin_vec, dim=1)  # representation of begin word
        end_vec = bert_out.float() * end.unsqueeze(-1).float()
        end_vec = torch.sum(end_vec, dim=1)  # representation of end word

        left_vec1 = feature[11]
        left_vec2 = feature[12]
        left_vec3 = feature[13]
        left_vec4 = feature[14]
        right_vec1 = feature[15]
        right_vec2 = feature[16]
        right_vec3 = feature[17]
        right_vec4 = feature[18]

        left_vec1 = bert_out.float() * left_vec1.unsqueeze(-1).float()
        left_vec1 = torch.sum(left_vec1, dim=1)  # representation of begin word
        left_vec2 = bert_out.float() * left_vec2.unsqueeze(-1).float()
        left_vec2 = torch.sum(left_vec2, dim=1)  # representation of begin word
        left_vec3 = bert_out.float() * left_vec3.unsqueeze(-1).float()
        left_vec3 = torch.sum(left_vec3, dim=1)  # representation of begin word
        left_vec4 = bert_out.float() * left_vec4.unsqueeze(-1).float()
        left_vec4 = torch.sum(left_vec4, dim=1)  # representation of begin word

        right_vec1 = bert_out.float() * right_vec1.unsqueeze(-1).float()
        right_vec1 = torch.sum(right_vec1, dim=1)  # representation of begin word
        right_vec2 = bert_out.float() * right_vec2.unsqueeze(-1).float()
        right_vec2 = torch.sum(right_vec2, dim=1)  # representation of begin word
        right_vec3 = bert_out.float() * right_vec3.unsqueeze(-1).float()
        right_vec3 = torch.sum(right_vec3, dim=1)  # representation of begin word
        right_vec4 = bert_out.float() * right_vec4.unsqueeze(-1).float()
        right_vec4 = torch.sum(right_vec4, dim=1)  # representation of begin word

        '''
        inp.append(bert_out)
        inp.append(e_out)
        inp.append(et_out)
        inp.append(begin)
        inp.append(end)

        print(clsout.shape)
        print(e_out.shape)
        print(et_out.shape)
        print(begin.shape)
        print(end.shape)
        exit()
        '''

        nip = torch.cat((clsout.unsqueeze(1).float(), e_out.unsqueeze(1).float(), et_out.unsqueeze(1).float(),
                         begin_vec.unsqueeze(1).float(), end_vec.unsqueeze(1).float(), left_vec1.unsqueeze(1).float(),
                         left_vec2.unsqueeze(1).float(), left_vec3.unsqueeze(1).float(), left_vec4.unsqueeze(1).float(),
                         right_vec1.unsqueeze(1).float(), right_vec2.unsqueeze(1).float(),
                         right_vec3.unsqueeze(1).float(), right_vec4.unsqueeze(1).float()), 1)

        adj = gen_adj(self.A).detach()
        # print(np.shape)
        # print(adj.shape)
        # exit()
        x = self.gc1(nip, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        # print(x.shape)
        x = torch.sum(x, dim=1)
        x = torch.cat((clso, x), dim=1)
        # print(x.shape)
        # exit()
        # x = x.transpose(0, 1)
        # x = torch.matmul(feature, x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        # x = F.softmax(x, dim=-1).squeeze(1)

        return x

    def get_config_optim(self, lr, lrp):
        return [
            # {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.bert.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.linear.parameters(), 'lr': lr},
        ]


def gcn_bert0(t, pretrained=True, adj_file=None, in_channel=768):
    model = BertModel.from_pretrained('./bert_pretrain')
    return GCNBert(model, t=t, adj_file=adj_file, in_channel=in_channel)