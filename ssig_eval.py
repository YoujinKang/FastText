import pickle
import fasttext
from common.eval_fuction import *
from common.preprocessing import *

with open('data/vocab.pkl', 'rb') as fr:
    _, word_to_id, id_to_word = pickle.load(fr)

with open('data/question.pkl', 'rb') as fr:
    semantic, syntatic = pickle.load(fr)

weight_file = 'train_log/ssig_3_100_5epoch/weight/ssig_3_4epoch.pkl'
with open(weight_file, 'rb') as fr:
    W_hash_in, _ = pickle.load(fr)

# model = fasttext.load_model("save_model/baseline/model_100_ssig.bin")
# W_hash_in = model.get_input_matrix()

W_in = []
for id in range(len(id_to_word)):
    subword_ids = get_subword_ids(id_to_word[id])
    vec = np.sum(W_hash_in[subword_ids], axis=0)
    vec /= len(subword_ids)
    W_in.append(vec.tolist())
W_in = np.array(W_in)


batch1 = len(semantic)//10 + 1
batch2 = len(syntatic)//10 + 1

sem_counts = 0
syn_counts = 0

for i in range(10):
    print("-"*10 +f'Chekcing question {i+1}/10' + '-'*10)
    sem = semantic[i * batch1 : (i + 1) * batch1]
    syn = syntatic[i * batch2 : (i + 1) * batch2]

    sem_v1, sem_v2, sem_v3, sem_v4 = convert_to_vec(sem, W_hash_in, W_in, word_to_id)
    pred_sem = sem_v2 - sem_v1 + sem_v3
    similarity_sem = cos_similarity(pred_sem, W_in)
    sem_max_top4, sem_sim_top4, sem_count = count_in_top4(similarity_sem, sem, id_to_word)
    sem_counts += sem_count

    sem_top4 = []
    for i in range(4):
        sem_top4.append(id_to_word[sem_max_top4[1][i]])
    print(sem[1][3], " : ", sem_top4)
    print("Sementic counts and length: {} / {}".format(sem_count , len(sem)))

    syn_v1, syn_v2, syn_v3, syn_v4 = convert_to_vec(syn, W_hash_in, W_in, word_to_id)
    pred_syn = syn_v2 - syn_v1 + syn_v3
    similarity_syn = cos_similarity(pred_syn, W_in)
    syn_max_top4, syn_sim_top4, syn_count = count_in_top4(similarity_syn, syn, id_to_word)
    syn_counts += syn_count

    syn_top4 = []
    for i in range(4):
        syn_top4.append(id_to_word[syn_max_top4[0][i]])
    print(syn[0][3], " : ", syn_top4)
    print("Syntatic counts and length: {} / {}".format(syn_count , len(syn)))


sem_acc = sem_counts / len(semantic) * 100
syn_acc = syn_counts / len(syntatic) * 100
tot_acc = (sem_counts + syn_counts) / (len(semantic) + len(syntatic)) * 100
print("-"*40)
print('Semantic accuracy: ', sem_acc)
print('Syntatic accuracy: ', syn_acc)
print('Total accuracy: ', tot_acc)
