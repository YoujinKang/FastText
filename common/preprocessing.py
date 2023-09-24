import re
import fasttext
from collections import Counter

model_sg = fasttext.load_model("save_model/baseline/model_ssig.bin")
# 3 <= n-gram <= 6

# hash_size = 2218316
# vocab_size = 218316

def get_subword_ids(word):
    return model_sg.get_subwords(word)[1]

def list_to_subword_ids(list):
    subword_ids = []
    for word in list:
        ids = get_subword_ids(word)
        subword_ids.append(ids)
    return subword_ids

def list_id_to_subword_ids(list_ids, id_to_word):
    words = []
    for id in list_ids:
        words.append(id_to_word[id])
    # wordsg 형태로 단어를 모음 
    subword_ids = list_to_subword_ids(words)
    return subword_ids

def get_word_id(word):
    return model_sg.get_word_id(word)

def list_to_word_ids(lines):
    word_ids = []
    vocab_num = 0
    for word in lines:
        ids = get_word_id(word)
        if ids == -1:
            continue
        word_ids.append(ids)
        vocab_num += 1
    return word_ids, vocab_num

def make_corpus(file):
    with open(file, 'r') as fr:
        data = fr.read()
    lines = model_sg.get_line(data)
    return lines[0]

def make_vocab_map(words):
    num = 0
    word_to_id = {}
    for word in words:
        word_to_id[word] = num
        num += 1
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return word_to_id, id_to_word

def make_vocab(corpus, min_count):
    counter = Counter(corpus)
    temp = Counter()
    for word in counter.keys():
        if counter[word] < min_count:
            temp[word] += 100  # 무조건 없어지게 min_count보다 훨씬 큰 값을 할당
    counter -= temp
    vocabulary = dict(counter.most_common())
    return vocabulary