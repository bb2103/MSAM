#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: lea.cgh
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import datetime

class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        # Data
        self.train_data_source = "avazu_train"
        self.test_data_source = "avazu_test"
        self.train_data_source_size = 32342973
        # Model
        self.context_size = 3
        self.embedding_size = 8
        self.embedding_config = {
            'hour': (240, self.embedding_size),
            'C1': (7, self.embedding_size),
            'site_id': (4000, self.embedding_size),
            'site_domain': (7000, self.embedding_size),
            'site_category': (30, self.embedding_size),
            'app_id': (8000, self.embedding_size),
            'app_domain': (550, self.embedding_size),
            'app_category': (40, self.embedding_size),
            'device_id': (2000000, self.embedding_size),
            'device_ip': (4000000, self.embedding_size),
            'device_model': (8000, self.embedding_size),
            'device_type': (5, self.embedding_size),
            'device_conn_type': (4, self.embedding_size),
            'C14': (2626, self.embedding_size),
            'C15': (8, self.embedding_size),
            'C16': (9, self.embedding_size),
            'C17': (435, self.embedding_size),
            'C18': (4, self.embedding_size),
            'C19': (68, self.embedding_size),
            'C20': (172, self.embedding_size),
            'C21': (60, self.embedding_size),
        }
        self.hidden_size_base = 256
        self.layer_count = {
            'common_dnn': 2,
            'common_dcn': 2,
            'context_dnn': 1,
            'context_dcn': 1,
        }
        self.composition = {
            'js': True,
            'context_emb': True,
            'softmax': True,
        }
        # Train
        self.loss_scaler = {
            'l1_regularization': 0.0,
            'l2_regularization': 5.00E-06,
            'js': 0.001,
        }
        self.batch_size = 2048 # 20000
        self.step_per_epoch = self.train_data_source_size // self.batch_size + 1
        self.learning_rate_base = 0.002
        self.learning_rate_values = [self.learning_rate_base, self.learning_rate_base / 2, self.learning_rate_base / 4, self.learning_rate_base / 8]
        self.learning_rate_boundaries = [self.step_per_epoch, self.step_per_epoch * 2, self.step_per_epoch * 4]
        self.task_name = "MSAM"
        self.task_ds = datetime.date.today().strftime("%Y%m%d")
        self.task_dir = "./%s/%s/" % (self.task_name, self.task_ds)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        key, sub_key = sys.argv[1], sys.argv[2]
        value = getattr(Config(), key, '')[sub_key]
    elif len(sys.argv) == 2:
        key = sys.argv[1]
        value = getattr(Config(), key, '')
    else:
        raise NotImplementedError("Unknown Args: %s" % sys.argv)
    print(value)
