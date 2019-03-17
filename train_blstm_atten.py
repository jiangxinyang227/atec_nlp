"""
训练bilstm attention模型
"""
import os

import numpy as np
import tensorflow as tf
from data_helper import DataSet
from blstm_atten import BiLSTMAttention
from metrics import mean, cal_acc, cal_auc, cal_f1


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_list("hidden_sizes", [256, 128], "Number of hidden units in each layer")
flags.DEFINE_integer("num_layers", 1, "Number of layers in each encoder and decoder")
flags.DEFINE_integer("embedding_size", 200, "Embedding dimensions of encoder and decoder inputs")

flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_float("dropout_prob", 0.5, "keep dropout prob")
flags.DEFINE_integer("epochs", 10, "Maximum # of training epochs")
flags.DEFINE_string("word_embedding_path", "word2vec/blstm_atten/word_embedding.npy",
                    "word embedding numpy store path")
flags.DEFINE_string("data_path", "data/tokens.txt", "raw data path")
flags.DEFINE_integer("steps_per_checkpoint", 100, "Save model checkpoint every this iteration")
flags.DEFINE_string("model_dir", "model/blstm_atten/", "Path to save model checkpoints")
flags.DEFINE_string("model_name", "atec.ckpt", "File name used for model checkpoints")

flags.DEFINE_float("ratio", 0.1, "eval data ratio")

dataSet = DataSet(filename=FLAGS.data_path, embedding_size=FLAGS.embedding_size, model="blstm_atten")

# 生成训练集和测试集
train_data, eval_data = dataSet.gen_train_eval_data()
print("train number: {}".format(len(train_data)))

word_embedding = np.load(FLAGS.word_embedding_path)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)


with tf.Session(config=config) as sess:
    model = BiLSTMAttention(embedding_size=FLAGS.embedding_size, hidden_sizes=FLAGS.hidden_sizes,
                            word_embedding=word_embedding, learning_rate=FLAGS.learning_rate)

    sess.run(tf.global_variables_initializer())

    current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

    for epoch in range(FLAGS.epochs):
        print("----- Epoch {}/{} -----".format(epoch + 1, FLAGS.epochs))

        for batch in dataSet.next_batch(train_data, FLAGS.batch_size):
            loss, preds, binary_preds = model.train(sess, batch, FLAGS.dropout_prob)
            acc = cal_acc(batch["label"], binary_preds)
            auc = cal_auc(batch["label"], preds)
            f1 = cal_f1(batch["label"], binary_preds)
            current_step += 1
            print("train: step: {}, loss: {}, acc: {}, auc: {}, f1: {}".format(current_step, loss, acc, auc, f1))
            if current_step % FLAGS.steps_per_checkpoint == 0:

                eval_losses = []
                eval_accs = []
                eval_aucs = []
                eval_f1s = []
                for eval_batch in dataSet.next_batch(eval_data, FLAGS.batch_size):
                    eval_loss, eval_preds, eval_binary_preds = model.eval(sess, eval_batch)
                    eval_acc = cal_acc(eval_batch["label"], eval_binary_preds)
                    eval_auc = cal_auc(eval_batch["label"], eval_preds)
                    eval_f1 = cal_f1(eval_batch["label"], eval_binary_preds)
                    eval_losses.append(eval_loss)
                    eval_accs.append(eval_acc)
                    eval_aucs.append(eval_auc)
                    eval_f1s.append(eval_f1)
                print("\n")
                print("train: step: {}, loss: {}, acc: {}, auc: {}, f1: {}".format(current_step,
                                                                                   mean(eval_losses),
                                                                                   mean(eval_accs),
                                                                                   mean(eval_aucs),
                                                                                   mean(eval_f1s)))
                print("\n")
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)