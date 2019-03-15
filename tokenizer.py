import jieba


FREQ_WORD = ["借呗", "花呗", "蚂蚁花呗", "蚂蚁借呗"]

for word in FREQ_WORD:
    jieba.suggest_freq(word, tune=True)


with open("data/atec_nlp_sim_train.csv", "r", encoding="utf8") as fr1:
    data1 = [line.strip().split("\t") for line in fr1.readlines()]


with open("data/atec_nlp_sim_train1.csv", "r", encoding="utf8") as fr2:
    data2 = [line.strip().split("\t") for line in fr2.readlines()]


with open("data/atec_nlp_sim_train2.csv", "r", encoding="utf8") as fr3:
    data3 = [line.strip().split("\t") for line in fr3.readlines()]


data = data1 + data2 + data3


def tokenizer():
    new_data = []
    for item in data:
        source = " ".join(jieba.cut(item[1]))
        target = " ".join(jieba.cut(item[2]))

        new_data.append("<SEP>".join([source, target, item[3]]))

    print("total data number: {}".format(len(new_data)))

    with open("data/tokens.txt", "a", encoding="utf8") as fw:
        for line in new_data:
            fw.write(line + "\n")


if __name__ == "__main__":
    tokenizer()