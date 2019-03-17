import tensorflow as tf


# 构建模型
class MutAttenCosine(object):
    """
    构建模型
    """

    def __init__(self, embedding_size, hidden_sizes, word_embedding, learning_rate,
                 max_gradient_norm=3, l2_reg_lambda=0.0):
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.l2_reg_lambda = l2_reg_lambda

        # 定义模型的输入
        self.input_a = tf.placeholder(tf.int32, [None, None], name="input_a")
        self.input_b = tf.placeholder(tf.int32, [None, None], name="input_b")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.sequence_len_a = tf.placeholder(tf.int32, [None], name="sequence_len_a")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.max_len_a = tf.reduce_max(self.sequence_len_a, name="max_len_a")
        self.mask_a = tf.sequence_mask(self.sequence_len_a, self.max_len_a, dtype=tf.float32, name='mask_a')

        self.sequence_len_b = tf.placeholder(tf.int32, [None], name="sequence_len_b")
        self.max_len_b = tf.reduce_max(self.sequence_len_b, name="max_len_b")
        self.mask_b = tf.sequence_mask(self.sequence_len_b, self.max_len_b, dtype=tf.float32, name="mask_b")

        # 定义l2损失
        l2_loss = tf.constant(0.0)

        with tf.name_scope("output_a"):
            output_a = self.bilstm(word_embedding, self.input_a, self.sequence_len_a, "a")

        with tf.name_scope("output_b"):
            output_b = self.bilstm(word_embedding, self.input_b, self.sequence_len_b, "b")

        # 利用句子a最后的输入对句子b进行attention，利用句子b最后的输出对句子a进行attention
        with tf.name_scope("attention_a"):
            b_atten = self.attention(output_a, output_b, output_b, self.sequence_len_a, self.sequence_len_b)

        with tf.name_scope("attention_b"):
            a_atten = self.attention(output_b, output_a, output_a, self.sequence_len_b, self.sequence_len_a)

        print(a_atten)
        print(b_atten)
        with tf.name_scope("cosine"):
            cosine_vec = tf.multiply(a_atten[:, -1, :], b_atten[:, -1, :])
        print(cosine_vec)
        # 全连接层的输出
        with tf.name_scope("finale_output"):
            output_size = self.hidden_sizes[-1]
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[1]), name="output_b")
            l2_loss += tf.nn.l2_loss(output_w)
            l2_loss += tf.nn.l2_loss(output_b)
            self.predictions = tf.nn.xw_plus_b(cosine_vec, output_w, output_b, name="predictions")
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

    def bilstm(self, word_embedding, x, sequence_len, sent_name):
        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            embedding_w = tf.Variable(tf.cast(word_embedding, dtype=tf.float32, name="word_embedding"),
                                      name="embedding_w")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded_words = tf.nn.embedding_lookup(embedding_w, x)

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
                                                                                   sequence_length=sequence_len,
                                                                                   dtype=tf.float32,
                                                                                   scope="blstm" + sent_name + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embedded_words = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embedded_words, 2, -1)
        output = outputs[0] + outputs[1]

        return output

    def padding_and_softmax(self, logits, query_len, key_len):
        """
        对attention权重归一化处理
        :param logits: 未归一化的attention权重 [batch_size, de_seq_len, en_seq_len]
        :param query_len: decoder的输入的真实长度 [batch_size]
        :param key_len: encoder的输入的真实长度 [batch_size]
        :return:
        """
        with tf.name_scope("padding_aware_softmax"):
            # 获得序列最大长度值
            de_seq_len = tf.shape(logits)[1]
            en_seq_len = tf.shape(logits)[2]

            # masks
            # [batch_size, de_seq_len]
            query_mask = tf.sequence_mask(lengths=query_len, maxlen=de_seq_len, dtype=tf.int32)
            # [batch_size, en_seq_len]
            key_mask = tf.sequence_mask(lengths=key_len, maxlen=en_seq_len, dtype=tf.int32)

            # 扩展一维
            query_mask = tf.expand_dims(query_mask, axis=2)  # [batch_size, de_seq_len, 1]
            key_mask = tf.expand_dims(key_mask, axis=1)  # [batch_size, 1, en_seq_len]

            # 将query和key的mask相结合 [batch_size, de_seq_len, en_seq_len]
            joint_mask = tf.cast(tf.matmul(query_mask, key_mask), tf.float32, name="joint_mask")

            # Padding should not influence maximum (replace with minimum)
            logits_min = tf.reduce_min(logits, axis=2, keepdims=True, name="logits_min")  # [batch_size, de_seq_len, 1]
            logits_min = tf.tile(logits_min, multiples=[1, 1, en_seq_len])  # [batch_size, de_seq_len, en_seq_len]
            logits = tf.where(condition=joint_mask > .5,
                              x=logits,
                              y=logits_min)

            # 获得最大值
            logits_max = tf.reduce_max(logits, axis=2, keepdims=True, name="logits_max")  # [batch_size, de_seq_len, 1]
            # 所有的元素都减去最大值  [batch_size, de_seq_len, en_seq_len]
            logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")

            # 导出未缩放的值
            weights_unscaled = tf.exp(logits_shifted, name="weights_unscaled")

            # mask 部分权重  [batch_size, de_seq_len, en_seq_len]
            weights_unscaled = tf.multiply(joint_mask, weights_unscaled, name="weights_unscaled_masked")

            # 得到每个时间步的总值 [batch_size, de_seq_len, 1]
            weights_total_mass = tf.reduce_sum(weights_unscaled, axis=2,
                                               keepdims=True, name="weights_total_mass")

            # 避免除数为0
            weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                          x=weights_total_mass,
                                          y=tf.ones_like(weights_total_mass))

            # 对权重进行正规化  [batch_size, de_seq_len, en_seq_len]
            weights = tf.divide(weights_unscaled, weights_total_mass, name="normalize_attention_weights")

            return weights

    def attention(self, query, key, value, query_len, key_len):
        """
        计算encoder decoder之间的attention
        :param query: decoder 的输入 [batch_size, de_seq_len, embedding_size]
        :param key: encoder的输出 [batch_size, en_seq_len, embedding_size]
        :param value: encoder的输出 [batch_size, en_seq_len, embedding_size]
        :param query_len: decoder的输入的真实长度 [batch_size]
        :param key_len: encoder的输入的真实长度 [batch_size]
        :return:
        """
        with tf.name_scope("attention"):
            # 通过点积的方法计算权重, 得到[batch_size, de_seq_len, en_seq_len]
            attention_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))

            # 对权重进行归一化处理
            attention_scores = self.padding_and_softmax(logits=attention_scores,
                                                        query_len=query_len,
                                                        key_len=key_len)
            # 对source output进行加权平均 [batch_size, de_seq_len, embedding_size]
            weighted_output = tf.matmul(attention_scores, value)

            return weighted_output

    def train(self, sess, batch, keep_prob):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        feed_dict = {self.input_a: batch["a"],
                     self.input_b: batch["b"],
                     self.sequence_len_a: batch["a_len"],
                     self.sequence_len_b: batch["b_len"],
                     self.input_y: batch["label"],
                     self.keep_prob: keep_prob}

        # 训练模型
        _, loss, preds, binary_preds = sess.run([self.train_op, self.loss, self.predictions, self.binary_preds],
                                                feed_dict=feed_dict)
        return loss, preds, binary_preds

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.input_a: batch["a"],
                     self.input_b: batch["b"],
                     self.sequence_len_a: batch["a_len"],
                     self.sequence_len_b: batch["b_len"],
                     self.input_y: batch["label"],
                     self.keep_prob: 1.0}

        # 验证模型
        loss, preds, binary_preds = sess.run([self.loss, self.predictions, self.binary_preds],
                                             feed_dict=feed_dict)
        return loss, preds, binary_preds

    def infer(self, sess, batch):
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.input_a: batch["a"],
                     self.input_b: batch["b"],
                     self.sequence_len_a: batch["a_len"],
                     self.sequence_len_b: batch["b_len"],
                     self.keep_prob: 1.0}

        # 预测结果
        preds, binary_preds = sess.run([self.predictions, self.binary_preds],
                                       feed_dict=feed_dict)
        return preds, binary_preds

