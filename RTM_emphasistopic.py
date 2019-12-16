import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class RTM(torch.nn.Module):
    def __init__(self, user_size, item_size, topic_size, embed_size, attention_size, dropout, user_review_len, item_review_len):
        super(RTM, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.topic_size = topic_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.user_review_len = user_review_len
        self.item_review_len = item_review_len
        self.attention_size = attention_size

        def init_weights(m):
            # if isinstance(m, nn.Conv2d):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

        self.user_embed = torch.nn.Embedding(self.user_size, self.embed_size)
        self.item_embed = torch.nn.Embedding(self.item_size, self.embed_size)
        self.user_embed.weight.data.normal_(0, 0.05)
        self.item_embed.weight.data.normal_(0, 0.05)

        self.user_attention = torch.nn.Embedding(self.user_size, self.embed_size)
        self.item_attention = torch.nn.Embedding(self.item_size, self.embed_size)
        self.user_attention.weight.data.normal_(0, 0.05)
        self.item_attention.weight.data.normal_(0, 0.05)

        self.topic_embed = torch.nn.Embedding(self.topic_size, self.embed_size)
        self.topic_embed.weight.data.normal_(0, 0.05)

        self.FC_u = nn.Linear(self.embed_size, self.attention_size)
        self.FC_i = nn.Linear(self.embed_size, self.attention_size)
        self.FC_review_u = nn.Linear(self.embed_size, self.attention_size)
        self.FC_review_i = nn.Linear(self.embed_size, self.attention_size)
        init_weights(self.FC_u)
        init_weights(self.FC_i)
        init_weights(self.FC_review_u)
        init_weights(self.FC_review_i)

        #self.h_u = torch.nn.Parameter(torch.randn(self.attention_size), requires_grad=True)
        #torch.nn.init.constant(self.h_u, 0.1)
        #self.h_i = torch.nn.Parameter(torch.randn(self.attention_size), requires_grad=True)
        #torch.nn.init.constant(self.h_i, 0.1)
        self.h_u = torch.nn.Linear(self.attention_size, 1)
        self.h_i = torch.nn.Linear(self.attention_size, 1)
        init_weights(self.h_u)
        init_weights(self.h_i)

        #A3ncf融合
        self.fusion_u = torch.nn.Linear(self.embed_size, self.embed_size)
        self.fusion_i = torch.nn.Linear(self.embed_size, self.embed_size)
        self.fusion_u.apply(init_weights)
        self.fusion_i.apply(init_weights)
        self.MLP_layers = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.predicts = torch.nn.Linear(self.embed_size, 1)

        #concat, mlp融合
        self.FC_pre = nn.Linear(2 * embed_size, 1)
        init_weights(self.FC_pre)

        # dot
        self.user_bias = nn.Embedding(self.user_size, 1)
        self.item_bias = nn.Embedding(self.item_size, 1)
        self.user_bias.weight.data.normal_(0, 0.01)
        self.item_bias.weight.data.normal_(0, 0.01)
        self.bias = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.bias.data.uniform_(0, 0.1)

        self.relu = nn.ReLU()

    def forward(self, user, item, user_r_topic, item_r_topic):
        uids_list = user.cuda()
        user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
        user_attention = self.user_attention(Variable(uids_list))
        sids_list = item.cuda()
        item_embedding = self.item_embed(torch.autograd.Variable(sids_list))
        item_attention = self.item_attention(Variable(sids_list))

        user_reviewTopic = Variable(torch.FloatTensor(user_r_topic).cuda())
        item_reviewTopic = Variable(torch.FloatTensor(item_r_topic).cuda())


        ##############################################topic-->review#########################################
        all_topic = Variable(torch.LongTensor(np.array(range(self.topic_size))).cuda())  # [topic_num]
        one = Variable(torch.ones(user_r_topic.shape).long().cuda())  # user ---> [batch, review_num, topic_num]
        all_topic_u = one * all_topic  # [batch, review_num, topic_num]
        all_topic_u = self.topic_embed(all_topic_u.view(-1, all_topic_u.shape[2]))
        all_topic_u = all_topic_u.view(user_r_topic.shape[0], user_r_topic.shape[1], user_r_topic.shape[2],
                                       -1)  # [batch, review_num, topic_num, embedding]
        # user_reviewTopic = user_reviewTopic.unsqueeze(3)
        user_review = all_topic_u * user_reviewTopic.unsqueeze(3)
        user_review = user_review.sum(2)  # batch, review_num, embedding]

        one = Variable(torch.ones(item_r_topic.shape).long().cuda())  # item ---> [batch, review_num, topic_num]
        all_topic_i = one * all_topic
        all_topic_i = self.topic_embed(all_topic_i.view(-1, all_topic_i.shape[2]))
        all_topic_i = all_topic_i.view(item_r_topic.shape[0], item_r_topic.shape[1], item_r_topic.shape[2], -1)
        item_review = all_topic_i * item_reviewTopic.unsqueeze(3)
        item_review = item_review.sum(2)

        #####################################review attention##############################################
        topic_emphasize = all_topic * Variable(torch.ones(user_r_topic.shape[0], self.topic_size).long().cuda())    #[batch, topic_size]
        topic_emphasize = self.topic_embed(topic_emphasize) #[batch, topoic_size ,embed]

        weight_user = self.h_u(self.relu(self.FC_u(user_attention).unsqueeze(1) + self.FC_review_u(user_review))).squeeze() #[batch,review_num]
        weight_user = nn.Softmax(dim=1)(weight_user)
        user_feature = (weight_user.unsqueeze(2) * user_review).sum(1)
        user_feature = (user_feature.unsqueeze(1) * topic_emphasize).sum(dim=1)

        weight_item = self.h_i( self.relu(self.FC_i(item_attention).unsqueeze(1) + self.FC_review_i(item_review))).squeeze()
        weight_item = nn.Softmax(dim=1)(weight_item)
        item_feature = (weight_item.unsqueeze(2) * item_review).sum(1)
        item_feature = (item_feature.unsqueeze(1) * topic_emphasize).sum(dim=1)

        ####################################prediction#####################################################
        '''#a3ncf的融合 relu(FC(embed, topic)), FC(pu*qi)
        pu = self.relu(self.fusion_u(user_embedding + user_feature))
        qi = self.relu(self.fusion_i(item_embedding + item_feature))
        f = pu * qi
        zl = self.MLP_layers(f)
        pred = self.predicts(f)'''

        #concat -> mlp
        tmp = torch.cat([user_feature + user_embedding, item_feature + item_embedding], dim=1)
        pred = self.relu(self.FC_pre(tmp))

        '''# dot
        tmp_u = user_embedding + user_feature
        tmp_i = item_embedding + item_feature
        bias = self.user_bias(Variable(uids_list)) + self.item_bias(Variable(sids_list))
        pred = ((tmp_u*tmp_i).sum(1)).squeeze() + bias.squeeze() + self.bias'''


        return pred.squeeze()

class RTMDataset(Dataset):
    def __init__(self, file):
        """
        After load_data processing, read train or test data. Num_neg is different for
        train and test. User_neg is the items that users have no explicit interaction.
        """
        f = open(file, 'r')
        u,i,l = [],[],[]
        for eachline in f:
            eachline = eachline.strip().split('\t')
            u.append(int(eachline[0]))
            i.append(int(eachline[1]))
            l.append(int(float(eachline[2])))
        feature = {}
        feature['user'] = u
        feature['item'] = i
        self.features = feature
        self.labels = l

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user = self.features['user'][idx]
        item = self.features['item'][idx]
        label = self.labels[idx]
        sample = {'user': user, 'item': item, 'label': label}

        return sample