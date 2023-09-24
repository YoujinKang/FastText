import numpy as np
import pickle


def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out

class Model_ssig:
    def __init__(self, hash_size, vocab_size, embedding_size, sample_table, negative_num, re_train=False):
        self.embedding_size = embedding_size
        self.sample_table = sample_table
        self.negative_num = negative_num
        
        self.W_hash_in = 0.1 * np.random.randn(hash_size, embedding_size).astype('f')
        self.W_out = np.zeros((vocab_size, embedding_size)).astype('f')   

                 
    # @jit(nopython=True)
    def forward(self, center_hash, contexts):
        # center = (center_word_id 하나짜리랑, center_hash_ids 리스트) 로 이루어짐
        # contexts = [contexts_id] 로만 되어있음 

        h = np.sum(self.W_hash_in[center_hash], axis=0)
        h /= len(center_hash)
        
        tot_label = []
        neg_samples = []
        
        while True:
            b = np.random.randint(low=0, high=len(self.sample_table), size=(self.negative_num * len(contexts)))
            if len(np.intersect1d(self.sample_table[b], np.array(contexts)).tolist()) == 0:  # 교집합이 있는지 확인
                neg_samples = self.sample_table[b]
                break
        for i, context in enumerate(contexts):
            label = np.append(context, neg_samples[i*self.negative_num : (i+1)*self.negative_num])
            tot_label.append(label)

        w_out = self.W_out[np.array(tot_label)].transpose(0, 2, 1)

        out = sigmoid(np.dot(h, w_out)) # 각 결과들의 레이블마다 시그모이드 적용
        # print('out= ', out)  # out: (len(contexts), negative_num + 1))
        p_loss = -np.sum(np.log(out[:, :1] + 1e-07))  # out = (1, label) -> 0번째 레이블 = 정답 -> 1이 아니면 모두 로스
        n_loss = -np.sum(np.log(1 - out[:, 1:] + 1e-07))  # 0번째 이후는 모두 오답 -> 0이 아니면 모두 로스
        loss = p_loss + n_loss

        self.cache = (h, center_hash, tot_label, out, len(contexts), w_out)

        return loss/len(contexts), np.sum(out)/len(out)

    # @jit(nopython=True)
    def backward(self, lr):
        h, center_hash, tot_label, out, len_contexts, w_out = self.cache
        w_out = w_out.transpose(0, 2, 1)

        h = h.reshape(self.embedding_size, 1)
        # dh = np.zeros((len_contexts, self.embedding_size))

        dout = out.copy()
        dout[:,:1] -= 1  # dout[i][:1] -= 1  # 정답에 대한 out값  -> y-t = p_out-1
        dout = dout.reshape(len_contexts, 1, self.negative_num+1)

        dW_out = np.dot(h, dout.T).T  # dW_out : (len(contexts), negative_num+1 embedding_size)
        dh = np.matmul(dout, w_out).squeeze(1)  # (len(contexts), embedding_size)

        self.W_hash_in[np.array(center_hash)] -= np.sum(dh, axis=0) * lr
        self.W_out[np.array(tot_label)] -= dW_out * lr
        return None