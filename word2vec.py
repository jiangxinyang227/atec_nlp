import logging
import os
import gensim
from gensim.models import word2vec


if not os.path.exists("data/word2vec.txt"):
    with open("data/tokens.txt", "r", encoding="utf8") as fr:
        data = [line.strip().split("<SEP>") for line in fr.readlines()]

    with open("data/word2vec.txt", "a", encoding="utf8") as fw:
        for row in data:
            fw.write(row[0] + "\n")
            fw.write(row[1] + "\n")


# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("data/word2vec.txt")

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=200, window=1, min_count=3, sg=1, iter=10)
model.wv.save_word2vec_format("word2vec/word2vec" + ".bin", binary=True)

