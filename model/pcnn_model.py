import numpy as np
import os
import tensorflow as tf

from .data_utils import minibatches, pad_sequences, piece_split
from .general_utils import Progbar
from .base_model import BaseModel



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

        word_ids_left = tf.keras.preprocessing.sequence.pad_sequences(word_ids_left, padding='post', value=0)
        pos1_ids_left = tf.keras.preprocessing.sequence.pad_sequences(pos1_ids_left, padding='post', value=0)
        pos2_ids_left = tf.keras.preprocessing.sequence.pad_sequences(pos2_ids_left, padding='post', value=0)

        word_ids_mid = tf.keras.preprocessing.sequence.pad_sequences(word_ids_mid, padding='post', value=0)
        pos1_ids_mid = tf.keras.preprocessing.sequence.pad_sequences(pos1_ids_mid, padding='post', value=0)
        pos2_ids_mid = tf.keras.preprocessing.sequence.pad_sequences(pos2_ids_mid, padding='post', value=0)

        word_ids_right = tf.keras.preprocessing.sequence.pad_sequences(word_ids_right, padding='post', value=0)
        pos1_ids_right = tf.keras.preprocessing.sequence.pad_sequences(pos1_ids_right, padding='post', value=0)
        pos2_ids_right = tf.keras.preprocessing.sequence.pad_sequences(pos2_ids_right, padding='post', value=0)

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
            self.pos2_ids_right: pos2_ids_right
        }

        if relations is not None:
            feed[self.relations] = relations

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_sentence_embeddings_op(self, word_ids, pos1_ids, pos2_ids):
        """Defines self.sentence_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words", reuse=True) as scope:
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

        # self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("pos1", reuse=True) as scope:
            self.logger.info("randomly initializing pos1 vectors")
            _pos1_embeddings = tf.get_variable(
                    name="_pos1_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nposition, self.config.dim_dim_pos])

            pos1_embeddings = tf.nn.embedding_lookup(_pos1_embeddings, \
                    pos1_ids, name="pos1_embeddings")

        with tf.variable_scope("pos2", reuse=True) as scope:
            self.logger.info("randomly initializing pos2 vectors")
            _pos2_embeddings = tf.get_variable(
                    name="_pos2_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nposition, self.config.dim_dim_pos])

            pos2_embeddings = tf.nn.embedding_lookup(_pos2_embeddings, \
                    pos2_ids, name="pos2_embeddings")

        # batch size
        assert tf.shape(word_embeddings)[0] == \
            tf.shape(pos1_embeddings)[0] == tf.shape(pos2_embeddings)[0]
        # max length of sentence part
        assert tf.shape(word_embeddings)[1] == \
            tf.shape(pos1_embeddings)[1] == tf.shape(pos2_embeddings)[1]
        assert tf.shape(word_embeddings)[2] == self.config.dim_word
        assert tf.shape(pos1_embeddings)[2] == self.config.dim_pos
        assert tf.shape(pos2_embeddings)[2] == self.config.dim_pos

        sentence_embeddings = tf.concat([word_embeddings, \
            pos1_embeddings, pos2_embeddings], 2)

        assert tf.shape(sentence_embeddings)[2] == self.config.dim
        return sentence_embeddings

    def add_convolution_op(self, sentence_embeddings):
        """Defines conv
        """
        with tf.variable_scope("conv", reuse=True) as scope:
            _conv = tf.layers.conv2d(
                inputs=sentence_embeddings,
                filters=self.config.feature_maps,
                kernel_size=[self.config.window_size, self.config.dim],
                strides=(1, self.config.dim),
                padding="same",
                name=scope.name
            )
        assert tf.shape(_conv)[2] == 1

        conv = tf.reshape(_conv, [-1, \
            tf.shape(sentence_embeddings)[1], self.config.feature_maps])
        maxpool = tf.layers.max_pooling1d(
            inputs=conv,
            pool_size=tf.shape(sentence_embeddings)[1],
            strides=tf.shape(sentence_embeddings)[1])
        assert tf.shape(maxpool)[1] == 1
        return maxpool


    def add_concat_op(self):
        """Defines self.concat
        First, concat left, middle, right parts.
        Second, concat different channels or feature maps.
        """
        sentence_embeddings_left  = self.add_sentence_embeddings_op(self.word_ids_left, \
                                    self.pos1_ids_left, self.pos2_ids_left)
        sentence_embeddings_mid   = self.add_sentence_embeddings_op(self.word_ids_mid, \
                                    self.pos1_ids_mid, self.pos2_ids_mid)
        sentence_embeddings_right = self.add_sentence_embeddings_op(self.word_ids_right, \
                                    self.pos1_ids_right, self.pos2_ids_right)

        # shape = (batch_size, 1, feature_maps)
        maxpool_left  = self.add_convolution_op(sentence_embeddings_left)
        maxpool_mid   = self.add_convolution_op(sentence_embeddings_mid)
        maxpool_right = self.add_convolution_op(sentence_embeddings_right)
        # shape = (batch_size, 3, feature_maps)
        _maxpool = tf.concat([maxpool_left, maxpool_mid, maxpool_right], 1)
        assert tf.shape(_maxpool)[1] == 3
        # shape = (batch_size, 3*feature_maps)
        maxpool  = tf.concat(_maxpool, 2)
        assert tf.shape(maxpool)[1] == 3 * self.feature_maps

        _gvector = tf.tanh(maxpool)
        self.gvector = tf.nn.dropout(_gvector, self.config.dropout)



    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # PCNN specific functions
        self.add_placeholders()
        self.add_concat_op()


        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
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
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

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
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                                    sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    # def piece_split(data, pos):
    #     """Split each sentence in batch into three piece
    #     accodring to entity1, entity2 position and sentence length.
    #
    #     Args:
    #         data: output matrix of convolution layer, representing batch of sentences.
    #         pos: list of positions, containing entity1, entity2 position and
    #                 sentence length of corresponding sentence in data.
    #     Return:
    #         piecewise_max: list of max pooling from sentence piece.
    #     """
    #     assert data.shape[0] == pos.shape[0]
    #     assert pos.shape[1] == 3
    #     num = data.shape[0]
    #     splited = [[] for i in range(num)]
    #     for i in range(num):
    #         splited[i].append(data[i][0:pos[i][0]])
    #         splited[i].append(data[i][pos[i][0]:pos[i][1]])
    #         splited[i].append(data[i][pos[i][1]:pos[i][2]])
    #
    #     piecewise_max = list()
    #     for i in splited:
    #         for j in i:
    #             piecewise_max.append(max(j))
    #
    #     assert len(piecewise_max) == pos.shape[0]*pos.shape[1]
    #     piecewise_max = np.asarray(piecewise_max, np.float32)
    #     return piecewise_max


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
