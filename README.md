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

#### 训练mut attention cosine模型
    将两个句子分别输入到两个bilstm模型中，再对两个句子的输入互相用attention去得到单个向量输出，最后对两个句子的输出做element-wise的乘积
    执行train_mut_atten_cosine.py