import numpy as np
import os
import tensorflow as tf

from .data_utils import minibatches, piece_split, bags_split, pad_sequences
from .general_utils import Progbar
from .base_model import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class PCNNModel(BaseModel):
    """Specialized class of Model for PCNN"""

    def __init__(self, config):
        super(PCNNModel, self).__init__(config)
        self.idx_to_relation = {idx: rel for rel, idx in
                        self.config.vocab_relations.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of left part of sentence in batch)
        self.word_ids_left = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_left")

        # shape = (batch size, max length of left part of sentence in batch)
        self.pos1_ids_left = tf.placeholder(tf.int32, shape=[None, None],
                        name="pos1_ids_left")

        # shape = (batch size, max length of left part of sentence in batch)
        self.pos2_ids_left = tf.placeholder(tf.int32, shape=[None, None],
                        name="pos2_ids_left")

        # shape = (batch size, max length of middle part of sentence in batch)
        self.word_ids_mid = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_mid")

        # shape = (batch size, max length of middle part of sentence in batch)
        self.pos1_ids_mid = tf.placeholder(tf.int32, shape=[None, None],
                        name="pos1_ids_mid")

        # shape = (batch size, max length of middle part of sentence in batch)
        self.pos2_ids_mid = tf.placeholder(tf.int32, shape=[None, None],
                        name="pos2_ids_mid")

        # shape = (batch size, max length of right part of sentence in batch)
        self.word_ids_right = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_right")

        # shape = (batch size, max length of right part of sentence in batch)
        self.pos1_ids_right = tf.placeholder(tf.int32, shape=[None, None],
                        name="pos1_ids_right")

        # shape = (batch size, max length of right part of sentence in batch)
        self.pos2_ids_right = tf.placeholder(tf.int32, shape=[None, None],
                        name="pos2_ids_right")

        self.maxlen_left = tf.placeholder(tf.int32, shape=[1],
                        name="maxlen_left")

        self.maxlen_mid = tf.placeholder(tf.int32, shape=[1],
                        name="maxlen_mid")

        self.maxlen_right = tf.placeholder(tf.int32, shape=[1],
                        name="maxlen_right")

        # shape = (batch size, 1)
        self.relations = tf.placeholder(tf.int32, shape=[None, 1],
                        name="relations")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, word_ids, pos1_ids, pos2_ids, pos, relations=None, lr=None, dropout=None):
        """Given some data, build a feed dictionary

        Args:
            word_ids: list of sentences. A sentence is a list of ids of words.
            pos1_ids: list of sentences. A sentence is a list of positions from words to entity1.
            pos2_ids: list of sentences. A sentence is a list of positions from words to entity2.
            pos: list of 3 length lists, containing the positions of entity1, entity2 and final word in sentences.
            relations: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        width = self.config.window_size - 1
        word_ids_left, word_ids_mid, word_ids_right = piece_split(word_ids, pos, width)
        pos1_ids_left, pos1_ids_mid, pos1_ids_right = piece_split(pos1_ids, pos, width)
        pos2_ids_left, pos2_ids_mid, pos2_ids_right = piece_split(pos2_ids, pos, width)

        blk = self.config.nposition - 1
        word_ids_left, maxlen_left = pad_sequences(word_ids_left)
        pos1_ids_left, _ = pad_sequences(pos1_ids_left, pad_tok=blk)
        pos2_ids_left, _ = pad_sequences(pos2_ids_left, pad_tok=blk)

        word_ids_mid, maxlen_mid = pad_sequences(word_ids_mid)
        pos1_ids_mid, _ = pad_sequences(pos1_ids_mid, pad_tok=blk)
        pos2_ids_mid, _ = pad_sequences(pos2_ids_mid, pad_tok=blk)

        word_ids_right, maxlen_right = pad_sequences(word_ids_right)
        pos1_ids_right, _ = pad_sequences(pos1_ids_right, pad_tok=blk)
        pos2_ids_right, _ = pad_sequences(pos2_ids_right, pad_tok=blk)


        # build feed dictionary
        feed = {
            self.word_ids_left:  word_ids_left,
            self.pos1_ids_left:  pos1_ids_left,
            self.pos2_ids_left:  pos2_ids_left,
            self.word_ids_mid:   word_ids_mid,
            self.pos1_ids_mid:   pos1_ids_mid,
            self.pos2_ids_mid:   pos2_ids_mid,
            self.word_ids_right: word_ids_right,
            self.pos1_ids_right: pos1_ids_right,
            self.pos2_ids_right: pos2_ids_right,
            self.maxlen_left:    maxlen_left,
            self.maxlen_mid:     maxlen_mid,
            self.maxlen_right:   maxlen_right
        }

        if relations is not None:
            feed[self.relations] = relations

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_sentence_embeddings_op(self, word_ids, pos1_ids, pos2_ids, maxlen):
        """Defines sentence_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words", reuse=tf.AUTO_REUSE):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_word_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, \
                    word_ids, name="word_embeddings")


        with tf.variable_scope("pos1", reuse=tf.AUTO_REUSE):
            # self.logger.info("randomly initializing pos1 vectors")
            _pos1_embeddings = tf.get_variable(
                    name="_pos1_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nposition, self.config.dim_pos])

            pos1_embeddings = tf.nn.embedding_lookup(_pos1_embeddings, \
                    pos1_ids, name="pos1_embeddings")

        with tf.variable_scope("pos2", reuse=tf.AUTO_REUSE):
            # self.logger.info("randomly initializing pos2 vectors")
            _pos2_embeddings = tf.get_variable(
                    name="_pos2_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nposition, self.config.dim_pos])

            pos2_embeddings = tf.nn.embedding_lookup(_pos2_embeddings, \
                    pos2_ids, name="pos2_embeddings")

        word_emb_shape = word_embeddings.get_shape().as_list()
        pos1_emb_shape = pos1_embeddings.get_shape().as_list()
        pos2_emb_shape = pos2_embeddings.get_shape().as_list()
        assert word_emb_shape[0] == pos1_emb_shape[0] == pos2_emb_shape[0]
        assert word_emb_shape[1] == pos1_emb_shape[1] == pos2_emb_shape[1]
        assert word_emb_shape[2] == self.config.dim_word
        assert pos1_emb_shape[2] == self.config.dim_pos
        assert pos2_emb_shape[2] == self.config.dim_pos

        sentence_embeddings = tf.concat([word_embeddings, \
            pos1_embeddings, pos2_embeddings], 2)

        sen_emb_shape = sentence_embeddings.get_shape().as_list()
        assert sen_emb_shape[2] == self.config.dim
        # (batch_size, max length of sentences in batch, vector representation dimension, 1)
        sentence_embeddings = tf.expand_dims(sentence_embeddings, -1)
        return sentence_embeddings

    def add_convolution_op(self, sentence_embeddings):
        """Defines conv and maxpool

        Args:
            sentence_embeddings:
        Returns:
            maxpool:

        """
        with tf.variable_scope("conv", reuse=tf.AUTO_REUSE) as scope:
            _conv = tf.layers.conv2d(
                inputs=sentence_embeddings,
                filters=self.config.feature_maps,
                kernel_size=[self.config.window_size, self.config.dim],
                strides=(1, self.config.dim),
                padding="same",
                name=scope.name
            )

        _conv_shape = _conv.get_shape().as_list()
        assert _conv_shape[2] == 1
        sen_emb_shape = sentence_embeddings.get_shape().as_list()
        conv = tf.squeeze(_conv, [2])
        maxpool = tf.reduce_max(conv, axis=1, keepdims=True)
        maxpool_shape = maxpool.get_shape().as_list()
        assert maxpool_shape[1] == 1
        maxpool = tf.squeeze(maxpool)
        # shape = (batch_size, feature_maps, 1)
        maxpool = tf.expand_dims(maxpool, -1)
        return maxpool


    def add_concat_op(self):
        """Defines self.concat
        First, concat left, middle, right parts.
        Second, concat different channels or feature maps.
        """
        sentence_embeddings_left  = self.add_sentence_embeddings_op(self.word_ids_left, \
                                self.pos1_ids_left, self.pos2_ids_left, self.maxlen_left)
        sentence_embeddings_mid   = self.add_sentence_embeddings_op(self.word_ids_mid, \
                                self.pos1_ids_mid, self.pos2_ids_mid, self.maxlen_mid)
        sentence_embeddings_right = self.add_sentence_embeddings_op(self.word_ids_right, \
                                self.pos1_ids_right, self.pos2_ids_right, self.maxlen_right)

        # shape = (batch_size, feature_maps, 1)
        maxpool_left  = self.add_convolution_op(sentence_embeddings_left)
        maxpool_mid   = self.add_convolution_op(sentence_embeddings_mid)
        maxpool_right = self.add_convolution_op(sentence_embeddings_right)
        # shape = (batch_size, feature_maps, 3)
        _maxpool = tf.concat([maxpool_left, maxpool_mid, maxpool_right], 2)
        # shape = (batch_size, 3*feature_maps)
        maxpool_flat = tf.reshape(_maxpool, [-1, 3*self.config.feature_maps])

        _gvector = tf.tanh(maxpool_flat)
        self.gvector = tf.nn.dropout(_gvector, self.config.dropout)


    def add_pred_op(self):
        """Defines self.logits and self.relations_pred
        """
        with tf.variable_scope("proj"):
            W1 = tf.get_variable("W1", dtype=tf.float32,
                    shape=[3*self.config.feature_maps, self.config.nrelations])

            b = tf.get_variable("b", dtype=tf.float32,
                    shape=[self.config.nrelations], initializer=tf.zeros_initializer())

        pred = tf.matmul(self.gvector, W1) + b
        self.logits = tf.reshape(pred, [-1, self.config.nrelations])

        relations_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        self.relations_pred = tf.reshape(relations_pred, [-1])


    def add_loss_op(self):
        """Defines the loss"""
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.relations)
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def log_trainable(self):
        """Print out trainable variables
        """
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            self.logger.info("Variable: {}".format(k))
            self.logger.info("Shape: {}".format(v.shape))
            # self.logger.info(v)


    def build(self):
        # PCNN specific functions
        self.add_placeholders()
        self.add_concat_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init
        self.log_trainable()


    def predict_batch(self, word_ids, pos1_ids, pos2_ids, pos):
        """
        Args:
            word_ids: list of sentences. A sentence is a list of ids of words.
            pos1_ids: list of sentences. A sentence is a list of positions from words to entity1.
            pos2_ids: list of sentences. A sentence is a list of positions from words to entity2.
            pos: list of 3 length lists, containing the positions of entity1, entity2 and final word in sentences.

        Returns:
            relations_pred: list of relations for each instance

        """
        fd = self.get_feed_dict(word_ids, pos1_ids, pos2_ids, pos, dropout=1.0)
        relations_pred = self.sess.run(self.relations_pred, feed_dict=fd)
        return relations_pred


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields list of tuple (word_idx, pos1, pos2, relation)
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, data in enumerate(minibatches(train, batch_size)):

            if self.config.MIL:
                # multi-instances learning
                word_ids, pos1_ids, pos2_ids, pos, relations = [], [], [], [], []
                word_bags, pos1_bags, pos2_bags, pos_bags, y_bags, num_bags = bags_split(data)
                for j in range(num_bags):
                    rel = y_bags[j][0]
                    fd = self.get_feed_dict(word_bags[j], pos1_bags[j], pos2_bags[j], pos_bags[j])
                    logits = self.sess.run(self.logits, feed_dict=fd)
                    scores = logits[:, rel]
                    idx = scores.index(max(scores))

                    word_ids.append(word_bags[j][idx])
                    pos1_ids.append(pos1_ids[j][idx])
                    pos2_ids.append(pos2_ids[j][idx])
                    pos.append(pos[j][idx])
                    relations.append(relations[j][idx])

            else:
                word_ids, pos1_ids, pos2_ids, pos, relations = data

            fd = self.get_feed_dict(word_ids, pos1_ids, pos2_ids, pos, relations, \
                        self.config.lr, self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, relation tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        y_true, y_pred = [], []
        for data in minibatches(test, self.config.batch_size):
            word_batch, pos1_batch, pos2_batch, pos_batch, y_batch = data
            relations_pred = self.predict_batch(word_batch, pos1_batch, pos2_batch, pos_batch)
            assert len(relations_pred) == len(y_batch)
            y_true += y_batch
            y_pred += relations_pred

        acc = accuracy_score(y_true, y_pred)
        p   = precision_score(y_true, y_pred, average='macro')
        r   = recall_score(y_true, y_pred, average='macro')
        f1  = f1_score(y_true, y_pred, average='macro')

        return {"acc":acc, "p":p, "r":r, "f1":f1}


    def predict(self, words, pos1_ids, pos2_ids, pos):
        """Returns list of tags

        Args:
            words: list of words (string), just one sentence (no batch)
            pos1_ids:
            pos2_ids:
            pos:

        Returns:
            preds: str, relation.

        """
        words = [self.config.processing_word(w) for w in words]

        pred_ids = self.predict_batch([words], [pos1_ids], [pos2_ids], [pos])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
