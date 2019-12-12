import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 6))
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None, None))
        self.model_dic = './model/ADNet'
        self.pic_dic = './pic/ADNet'
        self.x_att = self.AttentionLayer(self.inputs)
        self.x_dense = tf.layers.dense(self.x_att, 128, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.x_dense, 4)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=-1), tf.argmax(self.labels, axis=-1)),
                                          dtype=tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def AttentionLayer(self, x):
        x1 = tf.expand_dims(x, axis=2)
        x2 = tf.expand_dims(x, axis=1)
        score = tf.matmul(x1, x2)
        alpha = tf.nn.softmax(score)
        x_att = tf.matmul(alpha, x1)
        x_att = tf.reshape(x_att, shape=(-1, x_att.shape[1]))
        return x_att

    def train(self, x, y, x_dev, y_dev, batch_size, epoch):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            acc_train = []
            acc_dev = []
            loss_train = []
            loss_dev = []
            stop = False
            n = 0
            es_step = 0
            loss_stop = 99999
            while step < epoch and stop is False:
                print('Epoch:{}'.format(step))
                acc_total = 0
                loss_total = 0
                train_step = 0
                train_num = 0
                for x_batch, y_batch in self.getBatch(batch_size, x, y):
                    _, loss, acc = sess.run([self.train_op, self.loss, self.acc], feed_dict={self.inputs: x_batch,
                                                                                             self.labels: y_batch})
                    acc_total += acc
                    train_num += len(x_batch)
                    loss_total += loss
                    print('step{} [{}/{}] --acc:{:.5f}, --loss:{:.5f}'.format(train_step, train_num, len(x), acc, loss))
                    train_step += 1
                acc_t = acc_total / train_step
                loss_t = loss_total / train_step
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                acc_total = 0
                loss_total = 0
                dev_step = 0
                for x_batch, y_batch in self.getBatch(batch_size, x_dev, y_dev):
                    loss, acc = sess.run([self.loss, self.acc], feed_dict={self.inputs: x_batch, self.labels: y_batch})
                    acc_total += acc
                    loss_total += loss
                    dev_step += 1
                acc_d = acc_total / dev_step
                loss_d = loss_total / dev_step
                acc_dev.append(acc_d)
                loss_dev.append(loss_d)
                print('Epoch{}----acc:{:.5f},loss:{:.5f},val_acc:{:.5f},val_loss:{:.5f}'.format(step, acc_t, loss_t,
                                                                                                acc_d, loss_d))
                loss_ = loss_d
                acc_ = acc_d
                if loss_ > loss_stop:
                    if n >= 3:
                        stop = True
                    else:
                        n += 1
                else:
                    if not os.path.exists(self.model_dic):
                        os.makedirs(self.model_dic)
                    saver.save(sess, os.path.join(self.model_dic, 'model'))
                    es_step = step
                    n = 0
                    loss_stop = loss_
                step += 1
            if stop:
                print('Early Stop at Epoch{}'.format(es_step))

            if not os.path.exists(self.pic_dic):
                os.makedirs(self.pic_dic)
            plt.plot(acc_train)
            plt.plot(acc_dev)
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(self.pic_dic, 'acc.png'))
            plt.close()

            plt.plot(loss_train)
            plt.plot(loss_dev)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(self.pic_dic, 'loss.png'))
            plt.close()

    def predict(self, x_test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.model_dic)
            saver.restore(sess, ckpt)
            result = []
            index = 0
            for x_batch in self.getBatch(64, x_test):
                index += x_batch.shape[0]
                print('predict ==> [{}/{}]'.format(index, len(x_test)))
                temp = sess.run(self.logits, feed_dict={self.inputs: x_batch})
                result.append(temp)
        score = np.concatenate(result, axis=0)
        result = np.argmax(score, axis=-1)
        return result

    def getBatch(self, batch_size, x, y=None):
        if len(x) % batch_size == 0:
            steps = len(x) // batch_size
        else:
            steps = len(x) // batch_size + 1
        begin = 0
        for i in range(steps):
            end = begin + batch_size
            if end > len(x):
                end = len(x)
            x_batch = x[begin: end]
            if y is not None:
                y_batch = y[begin: end]
                yield x_batch, y_batch
            else:
                yield x_batch
            begin = end
