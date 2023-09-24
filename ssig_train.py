import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm.auto import tqdm
import time
import json
import numpy as np
import os
from common.preprocessing import *
from common.functions import subsampling_prob, subsampling, unigramsampler, make_contexts
from ssig import Model_ssig

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='ssig_3')
parser.add_argument('--least_freq', type=int, default=5)
parser.add_argument('--sub_t', type=float, default=1e-04)
parser.add_argument('--negative_num', type=float, default=5)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--max_epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.05)  
parser.add_argument('--embedding_size', type=int, default=100)
parser.add_argument('--eval_interval', type=int, default=5000000)

args = parser.parse_args()
log_dir = 'train_log/{}_{}_{}epoch/'.format(args.model, args.embedding_size, args.max_epoch)
hash_size = 2218316

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(log_dir + 'argparse.json', 'w') as f:
    json.dump(args.__dict__, f)

tb_dir = os.path.join(log_dir, 'tb')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)

tb_writer = SummaryWriter(tb_dir)

weight_dir = os.path.join(log_dir, 'weight')
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

data_path = 'data/fil9'

# load dictionary
dictionary_pkl = 'data/vocab.pkl'
with open(dictionary_pkl, 'rb') as fr:
    vocab, word_to_id, id_to_word = pickle.load(fr)
    fr.close()
print('Length of vocab:', len(vocab))
vocab_size = len(vocab)

with open('data/texts.pkl', 'rb') as fr:
    lines, tot_word_num = pickle.load(fr)

# tot_word_num = 0
# for word in vocab:
#     tot_word_num += vocab[word]
print('The number of total words: ', tot_word_num)



# probability of subsampling
id_to_sub_p = subsampling_prob(vocab, word_to_id, tot_word_num, args.sub_t)

sample_table = []
sample_table = unigramsampler(vocab, word_to_id, power=1/2)
sample_table = np.array(sample_table)

model = Model_ssig(hash_size, vocab_size, args.embedding_size, sample_table, args.negative_num)
epoch_present = 0
# epoch_present = 5

# training start
current = 0
alpha = 1
lr = args.lr * alpha

total_score = 0
total_loss = 0
loss_count = 0
avg_loss = 0
avg_score = 0

time_list = []
start_t = time.time()
print('-' * 30 + 'Start training')
for epoch in range(epoch_present, args.max_epoch):
    print("epoch: %d/%d" % (epoch + 1, args.max_epoch))
    sampled_line = subsampling(lines, id_to_sub_p)

    # per center word
    for center_idx, center in enumerate(tqdm(sampled_line)):

        alpha = 1 - current / (tot_word_num * args.max_epoch)   
        if alpha <= 0.00001:
            alpha = 0.00001

        lr = args.lr * alpha
        current += 1

        # center에 맞게 window_size만큼 앞뒤로 contexts 만듦
        contexts = make_contexts(args.window_size, sampled_line, center_idx)
        if len(contexts)==0:
            continue
        # training
        center_hash = get_subword_ids(id_to_word[center])
        loss, score = model.forward(center_hash, contexts)
        model.backward(lr)

        total_loss += loss
        total_score += score
        loss_count += 1

        # 학습한 중심 단어의 갯수가 eval_interval 배수가 될 때마다 loss와 score평균, lr 저장
        if (args.eval_interval is not None) and (current % args.eval_interval == 0):
            elapsed_t = time.time() - start_t
            avg_loss = total_loss / loss_count
            avg_score = total_score / loss_count
            print('avg_loss: ', avg_loss, 'avg_score: ', avg_score)
            total_loss = 0
            total_score = 0
            loss_count = 0

            tb_writer.add_scalar('score/real_train_word(*{})'.format(args.eval_interval), avg_score,
                                    current)
            tb_writer.add_scalar('loss/real_train_word(*{})'.format(args.eval_interval), avg_loss,
                                    current)
            tb_writer.add_scalar('lr/real_train_word(*{})'.format(args.eval_interval), lr, current)
            tb_writer.flush()

        # save temp weight per dataset segment
        
    et = (time.time() - start_t) / 3600
    time_list.append(et)
    print("epoch: {}/{}, elapsed_time: {}[h]".format(epoch + 1, args.max_epoch, et))

    print("The number of trained words per epoch: ", current)

    # save the weights

        
    with open(os.path.join(weight_dir, '{}_{}epoch.pkl'.format(args.model, epoch+1)), 'wb') as fw:
        pickle.dump((model.W_hash_in, model.W_out), fw)  

    if os.path.exists(weight_dir+'{}_{}epoch.pkl'.format(args.model, epoch)):
        os.remove(weight_dir+'{}_{}epoch.pkl'.format(args.model, epoch))

