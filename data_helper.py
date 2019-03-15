"""
准备训练数据
"""
from collections import Counter
import pickle
import random
import gensim
import numpy as np


class DataSet(object):
    def __init__(self, filename, embedding_size, ratio=0.05, is_concat=True):
        self.filename = filename
        self.ratio = ratio
        self.embedding_size = embedding_size
        self.is_concat = is_concat
        self.label_to_idx = {"0": 0, "1": 1}

    def _read_data(self):
        with open(self.filename, "r", encoding="utf8") as fr:
            data = [line.strip().split("<SEP>") for line in fr.readlines()]

        return data

    def _get_vocab(self, data):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        all_words = [word for row in data for word in row[0].strip().split(" ")] + \
                    [word for row in data for word in row[1].strip().split(" ")]

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        print("total word number: {}".format(len(word_count)))

        # 去除低频词
        words = [item[0] for item in sort_word_count if item[1] >= 1]
        print("sub word number: {}".format(len(words)))

        vocab, word_embedding = self._get_word_embedding(words)

        self.word_to_idx = dict(zip(vocab, list(range(len(vocab)))))
        self.idx_to_word = dict(zip(list(range(len(vocab))), vocab))

        with open("word2vec/vocab.txt", "a", encoding="utf8") as fw:
            for word in vocab:
                fw.write(word)

        np.save("word2vec/word_embedding.npy", word_embedding)

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("word2vec/word2idx.pkl", "wb") as f:
            pickle.dump(self.word_to_idx, f)

        with open("word2vec/idx2word.pkl", "wb") as f:
            pickle.dump(self.idx_to_word, f)

    def _get_word_embedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        word_vec = gensim.models.KeyedVectors.load_word2vec_format("word2vec/word2vec.bin", binary=True)
        vocab = []
        word_embedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("<PAD>")
        vocab.append("<UNK>")
        vocab.append("<SEP>")
        word_embedding.append(np.zeros(self.embedding_size))
        word_embedding.append(np.random.normal(scale=0.1, size=self.embedding_size))
        word_embedding.append(np.random.normal(scale=0.1, size=self.embedding_size))

        count = 0
        for word in words:
            try:
                vector = word_vec.wv[word]
                vocab.append(word)
                word_embedding.append(vector)
            except:
                count += 1

        print("有 {} 不存在于word2vec中".format(count))

        return vocab, np.array(word_embedding)

    def _process_data(self, batch: list) -> dict:
        """
        对每个batch进行按最大长度补全处理
        :param batch:
        :return:
        """
        if self.is_concat:
            batch_length = [len(row[0]) for row in batch]
            max_length = max(batch_length)
            new_data = [row[0] + [self.word_to_idx["<PAD>"]] * (max_length - len(row[0]))
                        for row in batch]
            new_label = [row[1] for row in batch]

            return dict(x=new_data, x_length=batch_length, label=new_label)

        else:
            first_sent_length = [len(row[0]) for row in batch]
            first_max_length = max(first_sent_length)
            first_sent = [row[0] + [self.word_to_idx["<PAD>"]] * (first_max_length - len(row[0]))
                          for row in batch]

            second_sent_length = [len(row[1]) for row in batch]
            second_max_length = max(second_sent_length)
            second_sent = [row[1] + [self.word_to_idx["<PAD>"]] * (second_max_length - len(row[1]))
                           for row in batch]

            new_label = [row[2] for row in batch]

            return dict(first_x=first_sent, first_len=first_sent_length, second_x=second_sent,
                        second_len=second_sent_length, label=new_label)

    def _transition_idx(self, x):
        return [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in x]

    def gen_train_eval_data(self):
        """
        返回的数据结构：[[[concat_sen], [label]], [[], []]] 或者 [[[fir_sen], [sec_sen], [label]], [[], [], []]]
        :param data:
        :return:
        """
        data = self._read_data()
        self._get_vocab(data)

        sub_data = []
        for item in data:
            if item[2] == "1":
                sub_data.append(item)
            else:
                sample = random.choice([0, 1, 2, 3])
                if sample == 0:
                    sub_data.append(item)

        print("sample data numble: {}".format(len(sub_data)))
        random.shuffle(sub_data)
        if self.is_concat:
            x = [[self._transition_idx(
                row[0].strip().split(" ") + ["<SEP>"] + row[1].strip().split(" "))]
                for row in sub_data]
        else:
            x = [[self._transition_idx(row[0].strip().split(" ")),
                  self._transition_idx(row[1].strip().split(" "))]
                 for row in sub_data]

        label = [[self.label_to_idx[row[2]]] for row in sub_data]

        new_data = []
        for i in range(len(x)):
            sentence = x[i]
            sentence.append(label[i])
            new_data.append(sentence)

        train_index = int(len(label) * self.ratio)

        train_data = new_data[train_index:]

        eval_data = new_data[:train_index]

        return train_data, eval_data

    def next_batch(self, data, batch_size) -> dict:
        """
        用生成器的形式返回batch数据
        :param data:
        :param batch_size:
        :return:
        """
        random.shuffle(data)
        batch_num = len(data) // batch_size

        for i in range(batch_num):
            batch_data = data[batch_size * i: batch_size * (i + 1)]
            new_batch = self._process_data(batch_data)
            yield new_batch


# if __name__ == "__main__":
#     dataset = DataSet("data/tokens.txt", 200)
#
#     train_data, eval_data = dataset.gen_train_eval_data()
#     print("train_data sample\n")
#     print(train_data[0])
#     print("train data number: {}".format(len(train_data)))
#     print("eval data number: {}".format(len(eval_data)))
#     print("batch sample\n")
#     print(next(dataset.next_batch(train_data, batch_size=2)))