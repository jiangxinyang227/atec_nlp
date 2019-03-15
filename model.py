import tensorflow as tf


# 构建模型
class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, embedding_size, hidden_sizes, word_embedding, learning_rate,
                 max_gradient_norm=3, l2_reg_lambda=0.0):
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.l2_reg_lambda = l2_reg_lambda

        # 定义模型的输入
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.sequence_len = tf.placeholder(tf.int32, [None], name="sequence_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.max_len = tf.reduce_max(self.sequence_len, name="max_len")
        self.mask = tf.sequence_mask(self.sequence_len, self.max_len, dtype=tf.float32, name='mask')

        # 定义l2损失
        l2_loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.embedding_w = tf.Variable(tf.cast(word_embedding, dtype=tf.float32, name="word_embedding"),
                                           name="embedding_w")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded_words = tf.nn.embedding_lookup(self.embedding_w, self.input_x)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hidden_size in enumerate(self.hidden_sizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，
                    # 其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                                   self.embedded_words,
                                                                                   dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embedded_words = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embedded_words, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H)
            output_size = self.hidden_sizes[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[1]), name="output_b")
            l2_loss += tf.nn.l2_loss(output_w)
            l2_loss += tf.nn.l2_loss(output_b)
            self.predictions = tf.nn.xw_plus_b(output, output_w, output_b, name="predictions")
            self.binary_preds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binary_preds")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        with tf.name_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # 对梯度进行梯度截断
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(tf.global_variables())

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """

        # 获得最后一层LSTM的神经元数量
        hidden_size = self.hidden_sizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.get_variable("W", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        new_M = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restore_M = tf.reshape(new_M, [-1, self.max_len])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restore_M)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.max_len, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeeze_R = tf.squeeze(r)

        sentence_repren = tf.tanh(sequeeze_R)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentence_repren, self.keep_prob)

        return output

    def train(self, sess, batch, keep_prob):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        feed_dict = {self.input_x: batch["x"],
                     self.sequence_len: batch["x_length"],
                     self.input_y: batch["label"],
                     self.keep_prob: keep_prob}

        # 训练模型
        _, loss, preds, binary_preds = sess.run([self.train_op, self.loss, self.predictions, self.binary_preds],
                                                feed_dict=feed_dict)
        return loss, preds, binary_preds

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.input_x: batch["x"],
                     self.sequence_len: batch["x_length"],
                     self.input_y: batch["label"],
                     self.keep_prob: 1.0}

        # 验证模型
        loss, preds, binary_preds = sess.run([self.loss, self.predictions, self.binary_preds],
                                             feed_dict=feed_dict)
        return loss, preds, binary_preds

    def infer(self, sess, batch):
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.input_x: batch["x"],
                     self.sequence_len: batch["x_length"],
                     self.keep_prob: 1.0}

        # 预测结果
        preds, binary_preds = sess.run([self.predictions, self.binary_preds],
                                       feed_dict=feed_dict)
        return preds, binary_preds

