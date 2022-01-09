import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
import argparse
import random
import pickle
import os
import sys
import re
import copy
import time
import argument_parser
import copy
import numpy as np
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from base_data_loader import BaseDataLoader
from util.threaded_iterator import ThreadedIterator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
from keras_radam.training import RAdamOptimizer
from util.network.tbcnn import TBCNN
from util import util_functions
np.set_printoptions(threshold=sys.maxsize)
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

def main(test_opt):
    test_opt.model_path = os.path.join(test_opt.model_path, 'rust_classification_task')
    checkfile = os.path.join(test_opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(test_opt.model_path)
    # print("The model path : " + str(checkfile))
    if not (ckpt and ckpt.model_checkpoint_path):
        print('Failed to upload the pretrained model')   
    tbcnn_model = TBCNN(test_opt)
    tbcnn_model.feed_forward()
    test_data_loader = BaseDataLoader(test_opt.batch_size, test_opt.label_size, test_opt.tree_size_threshold_upper, test_opt.tree_size_threshold_lower, test_opt.test_path, False)

    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)  
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            # print("Checkpoint path : " + str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        correct_labels = []
        predictions = []
        test_batch_iterator = ThreadedIterator(test_data_loader.make_minibatch_iterator(), max_queue_size=test_opt.worker)
        for test_step, test_batch_data in enumerate(test_batch_iterator):
            scores = sess.run(
                    [tbcnn_model.softmax],
                    feed_dict={
                        tbcnn_model.placeholders["node_type"]: test_batch_data["batch_node_type_id"],
                        tbcnn_model.placeholders["node_token"]:  test_batch_data["batch_node_sub_tokens_id"],
                        tbcnn_model.placeholders["children_index"]:  test_batch_data["batch_children_index"],
                        tbcnn_model.placeholders["children_node_type"]: test_batch_data["batch_children_node_type_id"],
                        tbcnn_model.placeholders["children_node_token"]: test_batch_data["batch_children_node_sub_tokens_id"],
                        tbcnn_model.placeholders["dropout_rate"]: 0.0
                    }
                )

            file = test_batch_data['batch_files']
            batch_predictions = list(np.argmax(scores[0],axis=1))
            confidence = np.amax(scores[0],axis=1)[0]
            if batch_predictions[0]==0:
                print('%s: Safe (%.3f).'%(file[0],confidence))
            elif batch_predictions[0]==1:
                print('%s: Unsafe (%.3f).'%(file[0],confidence))
            else:
                print('Uknown category')
if __name__ == "__main__":
    test_opt = argument_parser.parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = test_opt.cuda
    main(test_opt)
