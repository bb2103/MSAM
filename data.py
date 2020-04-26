#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: lea.cgh
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import tensorflow as tf

class DataSourceMananger(object):
    def __init__(self, names, *args, **kwargs):
        super(DataSourceMananger, self).__init__(*args, **kwargs)
        mapping = {
            "avazu_train": AvazuTrainDataSource,
            "avazu_test": AvazuTestDataSource,
        }
        self.data_sources = {name : mapping[name]() for name in names}
        self.init_ops = None
        self.features = None

    def get_features(self, batch_size, preprocess_fn=None):
        if self.features is None:
            with tf.variable_scope("DataSouces"):
                datasets = {name : self.data_sources[name].get_processed_dataset(batch_size, preprocess_fn) for name in self.data_sources}
                example_dataset = datasets.values()[0]
                iterator = tf.data.Iterator.from_structure(example_dataset.output_types, example_dataset.output_shapes)
                self.features = iterator.get_next()
                self.init_ops = {name : iterator.make_initializer(datasets[name]) for name in datasets}
        return self.features

    def set_source(self, sess, name):
        assert self.init_ops is not None, "Please Get Feature First"
        assert name in self.init_ops, "Dataset[%s] Is Not Init" % name
        sess.run(self.init_ops[name])
    
    def get_tables(self):
        outputs = set()
        for data_source in self.data_sources:
            for filename in data_source.filenames:
                outputs.add(filename)
        return list(outputs)

class DataSource(object):
    def __init__(self, filenames, record_defaults, record_names=None, *args, **kwargs):
        super(DataSource, self).__init__(*args, **kwargs)
        self.filenames = filenames
        self.record_defaults = record_defaults
        self.record_names = record_names

    def get_dataset(self, *args, **kwargs):
        return tf.data.experimental.CsvDataset(self.filenames, record_defaults=self.record_defaults, use_quote_delim=False, header=True)

    def get_processed_dataset(self, batch_size, preprocess_fn=None):
        def combined_preprocess_fn(*inputs):
            outputs = dict(zip(self.record_names, inputs))
            if callable(preprocess_fn):
                outputs = preprocess_fn(outputs)
            return outputs
        return self.get_dataset().batch(batch_size).map(combined_preprocess_fn, num_parallel_calls=8).prefetch(1)

class AvazuTrainDataSource(DataSource):
    def __init__(self, *args, **kwargs):
        filenames = [""]
        record_names = "label,context,position,hour,C1,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21".split(",")
        record_defaults = [tf.constant(0, dtype=tf.int64)] * 5 + [tf.constant('', dtype=tf.string)] * 9 + [tf.constant(0, dtype=tf.int64)] * 10
        super(AvazuTrainDataSource, self).__init__(filenames, record_defaults, record_names, *args, **kwargs)

class AvazuTestDataSource(DataSource):
    def __init__(self, *args, **kwargs):
        filenames = [""]
        record_names = "label,context,position,hour,C1,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21".split(",")
        record_defaults = [tf.constant(0, dtype=tf.int64)] * 5 + [tf.constant('', dtype=tf.string)] * 9 + [tf.constant(0, dtype=tf.int64)] * 10
        super(AvazuTestDataSource, self).__init__(filenames, record_defaults, record_names, *args, **kwargs)
