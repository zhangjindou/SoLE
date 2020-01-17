from model import Model
import tensorflow as tf
import datetime
import numpy as np
import config
from utils.batch_utils import Batch_Loader_with_groundings


class Complex(Model):
    def __init__(self, n_entities, n_relations, hparams):
        super(Complex, self).__init__(n_entities, n_relations, hparams)
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.build()

    def add_params(self):
        self.entity_embedding1 = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding1")
        self.entity_embedding2 = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding2")
        self.relation_embedding1 = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding1")
        self.relation_embedding2 = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding2")

    def add_prediction_op(self):
        self.e1_1 = tf.nn.embedding_lookup(self.entity_embedding1, self.heads)
        self.e1_2 = tf.nn.embedding_lookup(self.entity_embedding2, self.heads)
        self.e2_1 = tf.nn.embedding_lookup(self.entity_embedding1, self.tails)
        self.e2_2 = tf.nn.embedding_lookup(self.entity_embedding2, self.tails)
        self.r_1 = tf.nn.embedding_lookup(self.relation_embedding1, self.relations)
        self.r_2 = tf.nn.embedding_lookup(self.relation_embedding2, self.relations)

        self.score = tf.subtract(tf.reduce_sum(self.e1_1 * self.r_1 * self.e2_1, -1) +
                                tf.reduce_sum(self.e1_2 * self.r_1 * self.e2_2, -1) +
                                tf.reduce_sum(self.e1_1 * self.r_2 * self.e2_2, -1),
                                tf.reduce_sum(self.e1_2 * self.r_2 * self.e2_1, -1), name="score")
        self.pred = tf.nn.sigmoid(self.score, name="pred")

    def add_loss_op(self):
        losses = tf.nn.softplus(-self.labels * self.score)
        self.l2_loss = tf.reduce_mean(tf.square(self.e1_1)) \
            + tf.reduce_mean(tf.square(self.e1_2)) \
            + tf.reduce_mean(tf.square(self.e2_1)) \
            + tf.reduce_mean(tf.square(self.e2_2)) \
            + tf.reduce_mean(tf.square(self.r_1)) \
            + tf.reduce_mean(tf.square(self.r_2))
        self.loss = tf.add(tf.reduce_mean(losses), self.l2_reg_lambda * self.l2_loss, name="loss")


class SoLE_Complex(Model):
    def __init__(self, n_entities, n_relations, hparams):
        super(SoLE_Complex, self).__init__(n_entities, n_relations, hparams)
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.NNE_enable = hparams.NNE_enable
        #self.theta = hparams.theta
        self.build()

    def add_placeholders(self):
        self.heads = tf.placeholder(tf.int32, [None], name="head_entities")
        self.tails = tf.placeholder(tf.int32, [None], name="tail_entities")
        self.relations = tf.placeholder(tf.int32, [None], name="relations")
        self.labels = tf.placeholder(tf.float32, [None], name="labels")
        self.groundings = tf.placeholder(tf.int32, [None,None,3], name="grounding_instances")
        self.glabels = tf.placeholder(tf.float32, [None,None], name="soft_rule_labels")

    def create_feed_dict(self, heads, relations, tails, groundings, glabels, labels=None):
        feed_dict = {
            self.heads: heads,
            self.relations: relations,
            self.tails: tails,
            self.groundings: groundings,
            self.glabels: glabels,
        }
        if labels is not None:
            feed_dict[self.labels] = labels
        return feed_dict

    def add_params(self):

        minVal = 0.0
        maxVal = 1.0

        self.entity_embedding1 = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], minVal, maxVal, dtype=tf.float32, seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding1")
        self.entity_embedding2 = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], minVal, maxVal, dtype=tf.float32, seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding2")
        self.relation_embedding1 = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], minVal, maxVal, dtype=tf.float32, seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding1")
        self.relation_embedding2 = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], minVal, maxVal, dtype=tf.float32, seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding2")


    def add_prediction_op(self):
        self.e1_1 = tf.nn.embedding_lookup(self.entity_embedding1, self.heads)
        self.e1_2 = tf.nn.embedding_lookup(self.entity_embedding2, self.heads)
        self.e2_1 = tf.nn.embedding_lookup(self.entity_embedding1, self.tails)
        self.e2_2 = tf.nn.embedding_lookup(self.entity_embedding2, self.tails)
        self.r_1 = tf.nn.embedding_lookup(self.relation_embedding1, self.relations)
        self.r_2 = tf.nn.embedding_lookup(self.relation_embedding2, self.relations)

        self.pred = tf.nn.sigmoid(tf.reduce_sum(self.e1_1 * self.r_1 * self.e2_1, -1) +
                                  tf.reduce_sum(self.e1_2 * self.r_1 * self.e2_2, -1) +
                                  tf.reduce_sum(self.e1_1 * self.r_2 * self.e2_2, -1) -
                                  tf.reduce_sum(self.e1_2 * self.r_2 * self.e2_1, -1), name="pred")

    def add_loss_op(self):
        embedding_last_row = tf.constant(1.0, dtype=tf.float32, shape=[1, self.embedding_size])
        new_ent_embedding1 = tf.concat([self.entity_embedding1, embedding_last_row], 0)
        new_ent_embedding2 = tf.concat([self.entity_embedding2, embedding_last_row], 0)
        new_rel_embedding1 = tf.concat([self.relation_embedding1, embedding_last_row], 0)
        new_rel_embedding2 = tf.concat([self.relation_embedding2, embedding_last_row], 0)

        e1_1 = tf.nn.embedding_lookup(new_ent_embedding1, self.heads)
        e1_2 = tf.nn.embedding_lookup(new_ent_embedding2, self.heads)
        e2_1 = tf.nn.embedding_lookup(new_ent_embedding1, self.tails)
        e2_2 = tf.nn.embedding_lookup(new_ent_embedding2, self.tails)
        r_1 = tf.nn.embedding_lookup(new_rel_embedding1, self.relations)
        r_2 = tf.nn.embedding_lookup(new_rel_embedding2, self.relations)

        score = tf.reduce_sum(e1_1 * r_1 * e2_1, -1) + \
                tf.reduce_sum(e1_2 * r_1 * e2_2, -1) + \
                tf.reduce_sum(e1_1 * r_2 * e2_2, -1) - \
                tf.reduce_sum(e1_2 * r_2 * e2_1, -1)

        if self.groundings is not None:
            groundings_e1_1 = tf.nn.embedding_lookup(new_ent_embedding1, self.groundings[:, :, 0])
            groundings_e1_2 = tf.nn.embedding_lookup(new_ent_embedding2, self.groundings[:, :, 0])
            groundings_e2_1 = tf.nn.embedding_lookup(new_ent_embedding1, self.groundings[:, :, 2])
            groundings_e2_2 = tf.nn.embedding_lookup(new_ent_embedding2, self.groundings[:, :, 2])
            groundings_r_1 = tf.nn.embedding_lookup(new_rel_embedding1, self.groundings[:, :, 1])
            groundings_r_2 = tf.nn.embedding_lookup(new_rel_embedding2, self.groundings[:, :, 1])

            mask = tf.cast((self.glabels <= 0.0), dtype=tf.float32)

            grounding_pred_func = tf.nn.sigmoid(tf.reduce_sum(groundings_e1_1 * groundings_r_1 * groundings_e2_1, -1) +
                                tf.reduce_sum(groundings_e1_2 * groundings_r_1 * groundings_e2_2, -1) +
                                tf.reduce_sum(groundings_e1_1 * groundings_r_2 * groundings_e2_2, -1) -
                                tf.reduce_sum(groundings_e1_2 * groundings_r_2 * groundings_e2_1, -1)) - np.array([0.0, 0.0, 1.0])

            grounding_prob = tf.log(tf.clip_by_value((tf.reduce_prod(grounding_pred_func, axis=-1) + 1.0),
                                                     tf.exp(-16.0),
                                                     1.0))
            shape = tf.shape(self.glabels)
            rule_num = shape[1]
            values = tf.tile(tf.expand_dims(grounding_prob,1), [1,rule_num])
            values = values * mask
            groundings_loss = tf.square(tf.reduce_sum(values, axis=0) - tf.reduce_min(self.glabels, axis=0))

        else:
            groundings_loss = tf.constant(0.0)

        losses = tf.nn.softplus(-self.labels * score)

        self.l2_loss = tf.reduce_mean(tf.square(self.e1_1)) \
            + tf.reduce_mean(tf.square(self.e1_2)) \
            + tf.reduce_mean(tf.square(self.e2_1)) \
            + tf.reduce_mean(tf.square(self.e2_2)) \
            + tf.reduce_mean(tf.square(self.r_1)) \
            + tf.reduce_mean(tf.square(self.r_2))

        self.loss = tf.add(tf.reduce_mean(losses) + tf.reduce_mean(groundings_loss),
                           self.l2_reg_lambda * self.l2_loss, name="loss")


    def predict(self, sess, test_triples):
        test_batch_loader = Batch_Loader_with_groundings(
            None, test_triples, self.n_entities, self.n_relations,
            batch_size=5000, neg_ratio=0, contiguous_sampling=True)
        preds = []
        for i in range(len(test_triples) // 5000 + 1):
            input_batch = test_batch_loader()
            feed = self.create_feed_dict(**input_batch)
            pred = sess.run(self.pred, feed_dict=feed)
            preds = np.concatenate([preds, pred])
        return preds

    def add_training_op(self):
        """
        Define the training operator: self.train_op.
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.AdagradOptimizer(self.lr)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [(tf.clip_by_norm(g, 1.), v) for g, v in self.grads_and_vars]
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        if self.NNE_enable :
            self.get_varibles_op = tf.trainable_variables("entity")
            self.clip_varibles_op = [tf.assign(v, tf.clip_by_value(v, 0., 1.))
                                     for v in self.get_varibles_op]

    def train_on_batch(self, sess, input_batch):
        feed = self.create_feed_dict(**input_batch)
        _, step, loss = sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed)
        if self.NNE_enable:
            sess.run([self.clip_varibles_op])
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

    def save_embeddings(self, sess, path="./"):
        ent_em1, ent_em2, rel_em1, rel_em2 = sess.run([self.entity_embedding1, self.entity_embedding2,
                                                       self.relation_embedding1, self.relation_embedding2])
        np.savetxt(path + "entity2vec1"+str(self.embedding_size)+".init", ent_em1, delimiter="\t")
        np.savetxt(path + "entity2vec2" + str(self.embedding_size) + ".init", ent_em2, delimiter="\t")
        np.savetxt(path + "relation2vec1" + str(self.embedding_size) + ".init", rel_em1, delimiter="\t")
        np.savetxt(path + "relation2vec2" + str(self.embedding_size) + ".init", rel_em2, delimiter="\t")
