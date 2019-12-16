import os, time, argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict


from RTM_emphasistopic import RTM, RTMDataset

from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=0.001, type=float,
					help="learning rate.")
parser.add_argument("--dropout", default=0.5, type=float,
					help="dropout rate.")
parser.add_argument("--batch_size", default=128, type=int,
					help="batch size when training.")
parser.add_argument("--gpu", default="0", type=str,
					help="gpu card ID.")
parser.add_argument("--epochs", default=20, type=str,
					help="training epoches.")
parser.add_argument("--top_k", default=10, type=int,
					help="compute metrics@top_k.")
parser.add_argument("--clip_norm", default=5.0, type=float,
					help="clip norm for preventing gradient exploding.")
parser.add_argument("--embed_size", default=25, type=int, help="embedding size for users and items.")
parser.add_argument("--attention_size", default=50, type=int, help="embedding size for users and items.")
#parser.add_argument("--review_vec_size", default=50, type=int, help="embedding size for users and items.")


#################################evaluate############################################
def metrics(model, test_dataloader, user_vec, item_vec):
    #rmse, mse, mae = [],[],[]
    rmse, mse, mae = 0,0,0
    count = 0
    for batch_data in test_dataloader:
        user = batch_data['user'].long().cuda()
        item = batch_data['item'].long().cuda()
        label = batch_data['label'].float().cuda()
        user_review_topic = user_vec[user]
        item_review_topic = item_vec[item]

        prediction = (model(user, item, user_review_topic, item_review_topic)).cpu().data.numpy()
        prediction = prediction.reshape(prediction.shape[0])
        label = label.cpu().numpy()
        my_rmse = np.sum((prediction - label) ** 2)
        my_mse = np.sum((prediction - label) ** 2)
        my_mae = np.sum(np.abs(prediction - label))
        # my_rmse = torch.sqrt(torch.sum((prediction - label) ** 2) / FLAGS.batch_size)
        rmse+=my_rmse
        mse+=my_mse
        mae+=my_mae
        count += len(user)

    my_mse = mse/count
    my_rmse = np.sqrt(rmse/count)
    my_mae = mae/count
    return my_rmse,my_mse,my_mae
###########################################################################


def trans_vector(user_reviewVec, topic_num=50):
    user_reviewVec = OrderedDict(sorted(user_reviewVec.items(), key=lambda t: t[0]))
    user_values = list(user_reviewVec.values())
    u_len_list = [len(i) for i in user_values]
    u_len_list2 = sorted(u_len_list)
    #u_len = 317
    u_len = u_len_list2[int(0.90*len(u_len_list2))]    #根据0.9*len(u)， 把每个用户的评论条数固定成统一长度
    u_vector = []
    for i in user_values:
        vec = []
        for j in i:
            vec.append(list(j.values())[0])
        if len(vec) > u_len:
            vec = vec[:u_len]
        elif len(vec) < u_len:
            count = u_len - len(vec)
            for j in range(count):
                vec.append( np.array(topic_num*[1/topic_num], dtype=np.float32))
        vec = np.array(vec)
        u_vector.append(vec)
    u_vector = np.array(u_vector)
    #print(u_len_list, len(u_len_list))
    return u_vector, u_len, u_len_list



if __name__ == '__main__':

    f_reviewDict = '../garden_dataset/review_topic_25.pkl'
    f_train = '../garden_dataset/garden_clean_train.dat'
    f_test = '../garden_dataset/garden_clean_valid.dat'

    FLAGS = parser.parse_args()
    print("\nParameters:")
    print(FLAGS.__dict__)

    with open(f_reviewDict, 'rb') as f:
        review_dict = pickle.load(f)
    user_reviewTopic = review_dict['user_reviewTopic']
    item_reviewTopic = review_dict['item_reviewTopic']
    user_num = review_dict['user_num']
    item_num = review_dict['item_num']
    review_num = review_dict['review_num']
    topic_num = review_dict['topic_num']
    print(user_num, item_num, review_num, topic_num)

    user_vec, user_len, u_len_list = trans_vector(user_reviewTopic, topic_num=topic_num)  # review_vec([1686,14,59]), 固定评论数，每个用户评论数
    item_vec, item_len, i_len_list = trans_vector(item_reviewTopic, topic_num=topic_num)
    print(user_vec.shape, item_vec.shape)
    del user_reviewTopic, item_reviewTopic

    train_dataset = RTMDataset(f_train)
    test_dataset = RTMDataset(f_test)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset,  batch_size=FLAGS.batch_size, shuffle=False, num_workers=0)

    model = RTM(user_num, item_num, topic_num, FLAGS.embed_size, FLAGS.attention_size, FLAGS.dropout, user_len, item_len)
    # model = A3NCF_Update(user_size, item_size, FLAGS.embed_size, FLAGS.dropout, u_feat_arr, i_feat_arr)
    model.cuda()

    loss_function = torch.nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.001)

    writer = SummaryWriter()  # For visualization
    best_rmse = 5

    count = 0
    for epoch in range(FLAGS.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()

        for idx, batch_data in enumerate(train_dataloader):
            # Assign the user and item on GPU later.
            user = batch_data['user'].long().cuda()
            item = batch_data['item'].long().cuda()
            user_review_topic = user_vec[user]
            item_review_topic = item_vec[item]
            label = batch_data['label'].float().cuda()
            model.zero_grad()
            prediction = model(user, item, user_review_topic, item_review_topic)

            label = Variable(label)
            loss = loss_function(prediction, label)

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), FLAGS.clip_norm)
            optimizer.step()
            # if (count % 200 == 0):
            #    print('epoch: ', epoch, 'loss: ', loss)
            writer.add_scalar('data/loss', loss.data, count)
            count += 1

        tmploss = torch.sqrt(loss / FLAGS.batch_size)
        print(50 * '#')
        print('epoch: ', epoch, '     ', tmploss)
        # print(model.tran_v.data)

        model.eval()
        rmse,mse,mae = metrics(model, test_dataloader, user_vec, item_vec)
        print('test rmse,mse,mae: ', rmse,mse,mae)
        """if (rmse < best_rmse):
            best_rmse = rmse
            f_name = f_model + str(best_rmse)[:7] + '.dat'
            torch.save(model, f_name)
            print('save ok')"""