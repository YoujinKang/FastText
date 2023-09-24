import numpy as np
from tqdm.auto import tqdm
import collections


def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out

def subsampling_prob(vocabulary, word_to_id, total_word, sub_t):
    sub_p = []
    idx = []
    for (word, f) in vocabulary.items():
        idx.append(word_to_id[word])
        p = f/total_word
        sub_p.append((1+np.sqrt(p/sub_t)) * sub_t/p)
    id_to_sub_p = dict(np.stack((idx, sub_p), axis=1))
    return id_to_sub_p


def subsampling(sentence_, id_to_sub_p):  # sentence: list, id_to_sub_p : list
    corpus = []
    random_num = np.random.random()
    for id in sentence_:
        if id_to_sub_p[id] > random_num:
            corpus.append(id)
    return corpus


def unigramsampler(vocab, word_to_id, power = 3/4):
    sample_table = []
    current = 0
    pos_idx = []
    for word in tqdm(vocab.keys(), desc='Unigram Sampling'):
        freq = int(pow(vocab[word], 3/4))
        pos_idx.append((current, current+freq))
        current += freq
        temp = [int(word_to_id[word])] * freq
        sample_table.extend(temp)
    print('Unigram Table length: ', len(sample_table))
    return sample_table


# def make_contexts(window_size, corpus, target_idx):
#     contexts = []
#     while True:
#         window = int(np.random.randint(1, window_size+1))
#         for j in range(-window, window+1):  # window 가 2면 j= -2, -1, 0, 1, 2
#             position = target_idx+j
#             # target = 0 이면 contexts 는 0 다음부터만 생각해야됨, target은 contexts에 포함시키지 않음.
#             if position >= 0 and position != target_idx:
#                 # else:
#                 #     contexts.append(corpus[position])         # complete contexts
#                 if position >= len(corpus):  # corpus 의 마지막을 넘어가면 안되고 for문도 더 돌 필요가 없음.
#                     break
#                 else:
#                     contexts.append(corpus[position])
#         if len(contexts) > 0:
#             break
#     return contexts

def make_contexts(window_size, corpus, target_idx):
    contexts = []
    window = int(np.random.randint(1, window_size+1))
    position_start = max(0, target_idx - window)
    position_end = min(target_idx + window + 1, len(corpus))
    contexts.extend(corpus[position_start:target_idx])
    contexts.extend(corpus[target_idx + 1:position_end])
    return contexts


def count_total_word(sentence_list):
    total_word = 0
    for sentence in sentence_list:
        for _ in sentence:
            total_word += 1
    print("total training word: ", total_word)
    return total_word


class UnigramSampler:
    def __init__(self, sample_size, vocab, id_to_word):
        self.sample_size = sample_size
        self.vocab_size = len(vocab)
        
        self.word_p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = vocab[id_to_word[i]]
        self.word_p = np.power(self.word_p, 3/4)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        
        return negative_sample

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L-1):
            for j in range(i+1, L):
                # 가중치를 공유하는 경우
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def update(params, grads, lr):
    for i in range(len(params)):
        params[i] -= lr * grads[i]

def create_contexts_target(corpus, window_size):
    target = corpus
    window = int(np.random.randint(1, window_size+1))
    contexts = []
        
    print("Making contexts --- ")
    for idx in tqdm(range(len(corpus))):
        cs = []
        for t in range(-window, window+1):
            if t == 0:
                continue
            if idx + t >= len(corpus):
                break
            elif idx + t >= 0 and idx + t < len(corpus):
                cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)