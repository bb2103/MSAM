#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: lea.cgh
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import tensorflow as tf
import numpy as np
import collections
import config
import model
import data
import os
import utils


class Application(object):
    def __init__(self):
        super(Application, self).__init__()
        self.config = config.Config()
        self.model = model.Model(**self.config.__dict__)
        print(self.config.__dict__)

    def train(self):
        def preprocess_fn(inputs):
            for name in self.config.embedding_config:
                value_size = self.config.embedding_config[name][0]
                if inputs[name].dtype.is_integer:
                    inputs[name] = tf.mod(inputs[name], value_size)
                else:
                    inputs[name] = tf.string_to_hash_bucket_fast(inputs[name], value_size)
            inputs['context'] = tf.reshape(tf.to_float(inputs['context']), [-1, 1])
            inputs['position'] = tf.reshape(tf.to_float(inputs['position']), [-1, 1])
            inputs['label'] = tf.reshape(tf.to_float(inputs['label']), [-1, 1])
            return inputs
        with tf.Graph().as_default():
            print("Build Graph")
            data_sources = data.DataSourceMananger([self.config.train_data_source, self.config.test_data_source])
            inputs = data_sources.get_features(self.config.batch_size, preprocess_fn)
            scores, labels, context, global_step, losses, learning_rate, train_op = self.model.build_graph(inputs)
            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(config=config) as sess:
                print("Init")
                saver = tf.train.Saver()
                summary_writer = tf.summary.FileWriterCache.get(self.config.task_dir)
                sess.run(tf.global_variables_initializer())
                test_auc, test_logloss, test_scores, test_labels = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
                # Generators
                def train_generator():
                    data_sources.set_source(sess, self.config.train_data_source)
                    try:
                        while True:
                            _, step_value, learning_rate_value, losses_value = sess.run([train_op, global_step, learning_rate, losses])
                            yield step_value, learning_rate_value, losses_value
                            for _ in range(999):
                                sess.run(train_op)
                    except tf.errors.OutOfRangeError:
                        pass
                def test_generator():
                    data_sources.set_source(sess, self.config.test_data_source)
                    try:
                        while True:
                            context_arr, scores_arr, labels_arr = sess.run((context, scores, labels))
                            yield context_arr, scores_arr, labels_arr
                    except tf.errors.OutOfRangeError:
                        pass
                # Flow
                step_per_epoch = self.config.step_per_epoch
                while True:
                    # Train
                    print("[Train] Start")
                    for step_value, learning_rate_value, losses_value in train_generator():
                        epoch_value = 0 if step_per_epoch is None else int(step_value / step_per_epoch)
                        step_value = step_value if step_per_epoch is None else int(step_value % step_per_epoch)
                        losses_keys_str, losses_values_str = '+'.join(map(str, losses_value.keys())), '+'.join(map(str, losses_value.values()))
                        print("[Train] Epoch {epoch_value} Batch {step_value}/{step_per_epoch} Learning Rate {learning_rate_value:.5f} Loss({losses_keys_str}) {losses_values_str}".format(**locals()))
                    # Test
                    print("[Test] Start")
                    for idx, (context_arr, scores_arr, labels_arr) in enumerate(test_generator()):
                        print("[Test] Step %d" % idx) if idx % 100 == 0 else None
                        for _context, _score, _label in zip(context_arr, scores_arr, labels_arr):
                            test_scores[_context].append(_score)
                            test_labels[_context].append(_label)
                    names = sorted(test_scores)
                    print("[Test] Metrics Names%s" % names)
                    records = []
                    for name in names:
                        test_auc[name].append(roc_auc_score(test_labels[name], test_scores[name]))
                        test_logloss[name].append(log_loss(test_labels[name], test_scores[name]))
                        print("Context %s, AUC%s, LogLoss%s" % (name, test_auc[name], test_logloss[name]))
                        records.extend([test_auc[name][-1], test_logloss[name][-1]])
                        del test_scores[name][:], test_labels[name][:]
                    print("[Test] Early Stopping Check")
                    if len(test_auc[names[0]]) >= 2 and test_auc[names[0]][-1] < test_auc[names[0]][-2]:
                        break
                    else:
                        print('\t'.join(map(str, records)))
                        print(self.config.__dict__)
                        saver.save(sess, os.path.join(self.config.task_dir, 'model'))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(0)
    app = Application()
    app.train()
