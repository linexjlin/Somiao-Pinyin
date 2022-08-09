
'''
Training.
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
from data_load import get_batch, load_vocab
import tensorflow as tf
from modules import *
from tqdm import tqdm

class Graph():
    '''Builds a model graph'''

    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch()
            else:  # Evaluation
                self.x = tf.compat.v1.placeholder(tf.int32, shape=(None, hp.maxlen,))
                self.y = tf.compat.v1.placeholder(tf.int32, shape=(None, hp.maxlen,))

            # Load vocabulary
            pnyn2idx, _, hanzi2idx, _ = load_vocab()

            # Character Embedding for x
            enc = embed(self.x, len(pnyn2idx), hp.embed_size, scope="emb_x")

            # Encoder pre-net
            prenet_out = prenet(enc,
                                num_units=[hp.embed_size, hp.embed_size // 2],
                                is_training=is_training)  # (N, T, E/2)

            # Encoder CBHG
            ## Conv1D bank
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.embed_size // 2,
                               is_training=is_training)  # (N, T, K * E / 2)

            ## Max pooling
            enc = tf.compat.v1.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

            ## Conv1D projections
            enc = conv1d(enc, hp.embed_size // 2, 5, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                            activation_fn=tf.nn.relu, scope="norm1")
            enc = conv1d(enc, hp.embed_size // 2, 5, scope="conv1d_2")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                            activation_fn=None, scope="norm2")
            enc += prenet_out  # (N, T, E/2) # residual connections

            ## Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc, num_units=hp.embed_size // 2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            ## Bidirectional GRU
            enc = gru(enc, hp.embed_size // 2, True, scope="gru1")  # (N, T, E)

            ## Readout
            self.outputs = tf.compat.v1.layers.dense(enc, len(hanzi2idx), use_bias=False)
            self.preds = tf.cast(tf.argmax(self.outputs, axis=-1), dtype=tf.int32)

            if is_training:
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
                self.istarget = tf.cast(tf.not_equal(self.y, tf.zeros_like(self.y)), dtype=tf.float32)  # masking
                self.hits = tf.cast(tf.equal(self.preds, self.y), dtype=tf.float32) * self.istarget
                self.acc = tf.reduce_sum(input_tensor=self.hits) / tf.reduce_sum(input_tensor=self.istarget)
                self.mean_loss = tf.reduce_sum(input_tensor=self.loss * self.istarget) / tf.reduce_sum(input_tensor=self.istarget)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.compat.v1.summary.scalar('mean_loss', self.mean_loss)
                tf.compat.v1.summary.scalar('acc', self.acc)
                self.merged = tf.compat.v1.summary.merge_all()


def train():
    g = Graph(); print("Training Graph loaded")

    with g.graph.as_default():
        # Training
        sv = tf.compat.v1.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))


if __name__ == '__main__':
    train(); print("Done")
