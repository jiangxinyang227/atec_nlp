### ATEC蚂蚁开发者大赛

#### 依赖
    python == 3.6
    tensorflow-gpu == 1.10.0
    
#### 处理原始数据
    执行tokenizer.py
    
#### 生成词向量
    执行word2vec.py

#### 训练blstm attention模型
    作为基准模型，将句子拼接输入，中间用<SEP>分隔符分隔
    执行train_blstm_atten.py

