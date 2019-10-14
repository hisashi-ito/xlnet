#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【predict_xlnet】
# 
# 概要:
#      XLNet を分類問題に適用した場合の推論器
#      run_classifier.py のevalを実現するように実装する。
#
import sys
sys.path.append(".")
from run_classifier import LivedoorProcessor
from run_classifier import get_model_fn
from run_classifier import file_based_input_fn_builder
from run_classifier import file_based_convert_examples_to_features
from utils import str_to_value

# xlnet
import modeling
import tensorflow as tf
import configparser
import json
import glob
import os
import pandas as pd
import tempfile
import re

# report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 設定類
CURDIR = os.getcwd()
CONFIGPATH = os.path.join(CURDIR, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)
FILEURL = config['FINETUNING-DATA']['FILEURL']
FILEPATH = config['FINETUNING-DATA']['FILEPATH']
EXTRACTDIR = config['FINETUNING-DATA']['TEXTDIR']
# model files
PRETRAINED_MODEL_PATH = '*'
FINETUNE_OUTPUT_DIR = '*'

class FLAGS(object):
    '''Parameters.'''
    def __init__(self, finetuned_model_path):
        # sentencepiece model
        self.model_file = "../../bert-japanese/model/wiki-ja.model"
        self.vocab_file = "../../bert-japanese/model/wiki-ja.vocab"
        self.do_lower_case = True
        self.use_tpu = False
        self.output_dir = "/dummy"
        self.data_dir = "/work/data/livedoor"
        self.max_seq_length = 512
        self.init_checkpoint = finetuned_model_path
        self.predict_batch_size = 8
        
        # The following parameters are not used in predictions.
        # Just use to create RunConfig.
        self.master = None
        self.save_checkpoints_steps = 1
        self.iterations_per_loop = 1
        self.num_tpu_cores = 1
        self.learning_rate = 0
        self.num_warmup_steps = 0
        self.num_train_steps = 0
        self.train_batch_size = 0
        self.eval_batch_size = 0

def latest_ckpt_model():
    models = {}
    for name in glob.glob("{}/model.ckpt*data*".format(FINETUNE_OUTPUT_DIR)):
        m = re.search(r'model.ckpt\-(\d+)\.data', name)
        if m:
            models[int(m.group(1))] = name
    # 一番epoch数が大きいモデルを返却
    latest_key = sorted(models, reverse=True)[0]
    return models[latest_key]

def accracy(result, label_list):
    import pandas as pd
    test_df = pd.read_csv("/work/data/livedoor/test.tsv", sep='\t')
    test_df['predict'] = [ label_list[elem['probabilities'].argmax()] for elem in result ]
    acc = sum( test_df['label'] == test_df['predict'] ) / len(test_df)

    # 正答率(accuracy)を表示
    print("*** accuracy: {} ***".format(acc))

    # 詳細なレポートが簡単に見れる
    # support は正解データの数
    print("*** classification_report ***")
    print(classification_report(test_df['label'], test_df['predict']))
    print("*** 混合行列 ***")
    print(confusion_matrix(test_df['label'], test_df['predict']))

def main():
    # logger
    tf.logging.set_verbosity(tf.logging.INFO)
    if not tf.gfile.Exists(predict_dir):
        tf.gfile.MakeDirs(predict_dir)
    processors = {
        "livedoor": LivedoorProcessor
    }
    task_name = FLAGS.task_name.lower()
    # processorを作成
    processor = processors[task_name]()
    label_list = processor.get_labels()

    # sentencepice の初期化
    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    # 工事中...