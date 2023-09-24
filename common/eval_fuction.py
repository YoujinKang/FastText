import numpy as np
import pickle
from common.preprocessing import get_subword_ids

def get_word_vec(word, W_hash_in, W_in, word_to_id):
    word = word.lower()
    if word in word_to_id.keys():
        vec = W_in[word_to_id[word]]
    else:
        print('Not in the vocabulary: ', word)
        subword_ids = get_subword_ids(word)  # common.preprocessing
        vec = np.sum(W_hash_in[subword_ids], axis=0)
        vec /= len(subword_ids)
    
    norm = np.linalg.norm(vec)
    vec /= (norm + 1e-08)
    return np.array(vec)

def convert_to_vec(questions, W_hash_in, W_in, word_to_id):
    vec0 = []
    vec1 = []
    vec2 = []
    vec3 = []
    for sentence in questions:
        vec0.append(get_word_vec(sentence[0], W_hash_in, W_in, word_to_id))
        vec1.append(get_word_vec(sentence[1], W_hash_in, W_in, word_to_id))
        vec2.append(get_word_vec(sentence[2], W_hash_in, W_in, word_to_id))
        vec3.append(get_word_vec(sentence[3], W_hash_in, W_in, word_to_id))
    return np.array(vec0), np.array(vec1), np.array(vec2), np.array(vec3)
            

def cos_similarity(predict, W_in):
    similarity = np.dot(predict, W_in.T)  # 1*V 짜리 
    similarity /= np.add(np.linalg.norm(W_in, axis=1), 1e-08)
    
    return similarity

def count_in_top4(similarity, questions, id_to_word):  
    count = 0
    top4_idx = []
    top4_sim = []
    for i in range(len(similarity)):
        ques_set = set(questions[i])
        sort_idx = np.argsort(similarity[i])
        max_arg = sort_idx[::-1]  # 내림차순 정리
        temp = list(max_arg[:4])
        top4_idx.append(temp)
        top4_sim.append(list(similarity[i][temp]))

        for j in range(4):
            if temp[j] not in id_to_word.keys():
                continue
            else:
                pred = id_to_word[temp[j]]
                if pred in ques_set:
                    if pred == questions[i][3].lower():
                        count += 1
                else:
                    continue

    return top4_idx, top4_sim, count


def prepare_eval():
    with open('data/questions-words.txt', 'r') as fr:
        loaded = fr.readlines()
    count = 0
    semantic = []
    syntatic = []
    for line in loaded:
        if line[0] == ':':  # 새로운 유형의 question
            count += 1
            continue
        elif line == '\n':  # 다음 question 
            continue
        
        if count < 6:  # 총 16개의 유형 중 앞 5개는 semantic, 11개는 syntatic question
            semantic.append(line.lower().split())
        else:
            syntatic.append(line.lower().split())

    with open('data/question.pkl', 'wb') as fw:
        pickle.dump((semantic, syntatic), fw)

# prepare_eval()