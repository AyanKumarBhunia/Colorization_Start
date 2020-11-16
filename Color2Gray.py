from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from ops import *
import cv2
import os
from vgg import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from natsort import natsorted

pathTrain_DomainA = './places365standard_easyformat/places365_standard/train'
pathTrain_DomainB = './places365standard_easyformat/places365_standard/train'
pathTest_DomainA = './places365standard_easyformat/places365_standard/val'
pathTest_DomainB = './places365standard_easyformat/places365_standard/val'

global params
params = {'pathTrain_DomainA': pathTrain_DomainA,
          'pathTrain_DomainB': pathTrain_DomainB,
          'pathTest_DomainA': pathTest_DomainA,
          'pathTest_DomainB': pathTest_DomainB,
          'batch_size': 64,
          'output_size': 256,
          'gf_dim': 32,
          'df_dim': 32,
          'model_path': './model',
          'L1_lambda': 100,
          'lr': 0.0001,
          'beta_1': 0.5,
          'epochs': 500}


def get_file_paths(path):
    img_paths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith('.jpg')])
    print("NUMBER OF PATHS :", len(img_paths))
    return img_paths

def load_data_DomainA(path):
    img_ = cv2.resize(cv2.imread(path, 1), (256, 256))
    return img_ / 127.5 - 1.

def load_data_DomainB(path):
    img_ = cv2.resize(cv2.imread(path, 0), (256, 256))
    return img_ / 127.5 - 1.

# Functions to load and save weights
def load_weights(saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        print("MODEL LOADED SUCCESSFULLY")
    else:
        print("LOADING MODEL FAILED")


def save(saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(sess, dir, step)


global d_bn1, d_bn2, d_bn3

global g_bn_e2, g_bn_e3, g_bn_e4, g_bn_e5, g_bn_e6, g_bn_e7, g_bn_e8

global g_bn_d1, g_bn_d2, g_bn_d3, g_bn_d4, g_bn_d5, g_bn_d6, g_bn_d7

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn_e2 = batch_norm(name='g_bn_e2')
g_bn_e3 = batch_norm(name='g_bn_e3')
g_bn_e4 = batch_norm(name='g_bn_e4')
g_bn_e5 = batch_norm(name='g_bn_e5')
g_bn_e6 = batch_norm(name='g_bn_e6')
g_bn_e7 = batch_norm(name='g_bn_e7')
g_bn_e8 = batch_norm(name='g_bn_e8')

g_bn_d1 = batch_norm(name='g_bn_d1')
g_bn_d2 = batch_norm(name='g_bn_d2')
g_bn_d3 = batch_norm(name='g_bn_d3')
g_bn_d4 = batch_norm(name='g_bn_d4')
g_bn_d5 = batch_norm(name='g_bn_d5')
g_bn_d6 = batch_norm(name='g_bn_d6')
g_bn_d7 = batch_norm(name='g_bn_d7')


def Generator_1(image, reuse = False):
    s = params['output_size']
    output_c_dim = 1
    s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(
        s / 128)
    gf_dim = params['gf_dim']

    with tf.variable_scope("generator1") as scope:

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x gf_dim)
        e2 = g_bn_e2(conv2d(lrelu(e1), gf_dim * 2, name='g_e2_conv'))
        # e2 is (64 x 64 x gf_dim*2)
        e3 = g_bn_e3(conv2d(lrelu(e2), gf_dim * 4, name='g_e3_conv'))
        # e3 is (32 x 32 x gf_dim*4)
        e4 = g_bn_e4(conv2d(lrelu(e3), gf_dim * 8, name='g_e4_conv'))
        # e4 is (16 x 16 x gf_dim*8)
        e5 = g_bn_e5(conv2d(lrelu(e4), gf_dim * 8, name='g_e5_conv'))
        # e5 is (8 x 8 x gf_dim*8)
        e6 = g_bn_e6(conv2d(lrelu(e5), gf_dim * 8, name='g_e6_conv'))
        # e6 is (4 x 4 x gf_dim*8)
        e7 = g_bn_e7(conv2d(lrelu(e6), gf_dim * 8, name='g_e7_conv'))
        # e7 is (2 x 2 x gf_dim*8)
        e8 = g_bn_e8(conv2d(lrelu(e7), gf_dim * 8, name='g_e8_conv'))
        # e8 is (1 x 1 x gf_dim*8)

        batch_size = params['batch_size']
        d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8),
                                  [batch_size, s128, s128, gf_dim * 8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(g_bn_d1(d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
                                  [batch_size, s64, s64, gf_dim * 8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(g_bn_d2(d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
                                  [batch_size, s32, s32, gf_dim * 8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(g_bn_d3(d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x gf_dim*8*2)

        d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
                                  [batch_size, s16, s16, gf_dim * 8], name='g_d4', with_w=True)
        d4 = g_bn_d4(d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x gf_dim*8*2)

        d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
                                  [batch_size, s8, s8, gf_dim * 4], name='g_d5', with_w=True)
        d5 = g_bn_d5(d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x gf_dim*4*2)

        d6, d6_w, sd6_b = deconv2d(tf.nn.relu(d5),
                                   [batch_size, s4, s4, gf_dim * 2], name='g_d6', with_w=True)
        d6 = g_bn_d6(d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x gf_dim*2*2)

        d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
                                  [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
        d7 = g_bn_d7(d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x gf_dim*1*2)

        d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7),
                                  [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def Discriminator_1(image, y=None, reuse=False):
    df_dim = params['df_dim']
    batch_size = params['batch_size']
    with tf.variable_scope("discriminator1") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x df_dim)
        h1 = lrelu(d_bn1(conv2d(h0, df_dim * 2, name='d_h1_conv')))
        # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(d_bn2(conv2d(h1, df_dim * 4, name='d_h2_conv')))
        # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(d_bn3(conv2d(h2, df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x df_dim*8)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4


def train():
    Domain_A = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'], 256, 256, 3], name='Color')
    Domain_B = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'], 256, 256, 1], name='Gray')

    Domain_A2B = Generator_1(tf.image.rgb_to_grayscale(Domain_A), reuse=False)
    Domain_B2B = Generator_1(Domain_B, reuse=True)

    tf.summary.image("DomainA_Input", Domain_A, max_outputs=4)
    tf.summary.image("DomainB_Input", Domain_B, max_outputs=4)
    tf.summary.image("DomainA_Input_GRAY", tf.image.rgb_to_grayscale(Domain_A), max_outputs=4)
    tf.summary.image("Output_Color2Gray", Domain_A2B, max_outputs=4)
    tf.summary.image("Output_Gray2Gray", Domain_B2B, max_outputs=4)

    D_real, D_real_logits = Discriminator_1(Domain_B, reuse=False)
    D_fake, D_fake_logits = Discriminator_1(Domain_A2B, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
    Reconstruction_Loss = tf.reduce_mean(tf.abs(Domain_B - Domain_B2B))
    g_loss = g_loss + 50 * Reconstruction_Loss

    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'discriminator1' in var.name]
    g_vars = [var for var in t_vars if 'generator1' in var.name]

    d_optim = tf.train.AdamOptimizer(params['lr'], beta1=params['beta_1']).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(params['lr'], beta1=params['beta_1']).minimize(g_loss, var_list=g_vars)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver(max_to_keep=5)
    load_weights(saver, params['model_path'])

    summary = tf.summary.merge_all()
    trainwriter = tf.summary.FileWriter("./logs/train", sess.graph)
    #testwriter = tf.summary.FileWriter("./logs/test", sess.graph)

    counter = 1
    pathTrain_DomainA = np.array(get_file_paths(params['pathTrain_DomainA']))
    pathTrain_DomainB = np.array(get_file_paths(params['pathTrain_DomainB']))
    pathTest_DomainA = np.array(get_file_paths(params['pathTest_DomainA']))
    pathTest_DomainB = np.array(get_file_paths(params['pathTest_DomainB']))


    np.random.shuffle(pathTrain_DomainA)
    np.random.shuffle(pathTrain_DomainB)
    np.random.shuffle(pathTest_DomainA)
    np.random.shuffle(pathTest_DomainB)

    num_DomainA = len(pathTrain_DomainA)
    num_DomainB = len(pathTrain_DomainB)
    print(num_DomainA // params['batch_size'])


    for epoch in range(params['epochs']):
        print("Epoch:{}".format(epoch))

        for idx in range(num_DomainA // params['batch_size']):

            batch_DomainA_path = pathTrain_DomainA[idx * params['batch_size']: (idx + 1) * params['batch_size']]
            batch_DomainA_data = np.array([load_data_DomainA(path) for path in batch_DomainA_path])
            batch_DomainB_path = pathTrain_DomainB[np.random.choice(range(num_DomainB), params['batch_size'], replace=False)]
            batch_DomainB_data = np.expand_dims([load_data_DomainB(path) for path in batch_DomainB_path], -1)

            feed_dict = {Domain_A: batch_DomainA_data, Domain_B: batch_DomainB_data}

            _, d_loss_ = sess.run([d_optim, d_loss], feed_dict)  # update D network
            _, g_loss_ = sess.run([g_optim, g_loss], feed_dict)  # update G network
            _, summary_, g_loss_ = sess.run([g_optim, summary, g_loss], feed_dict)  # update G network

            trainwriter.add_summary(summary_, counter)

            print(
                '#epoch : ' + str(epoch) + ' idx : ' + str(idx) + ' Dis loss : ' + str(d_loss_) + ' Gen loss : ' + str(
                    g_loss_))

            #if idx % 10 == 0:
                #batch_DomainA_path = pathTest_DomainA[np.random.choice(range(len(pathTest_DomainA)), params['batch_size'], replace=False)]
                #batch_DomainA_data = np.array([load_data_DomainA(path) for path in batch_DomainA_path])
                #batch_DomainB_data = np.expand_dims([load_data_DomainB(path) for path in batch_DomainA_path], -1)

                #feed_dict = {Domain_A: batch_DomainA_data, Domain_B: batch_DomainB_data}

                #test_summary_ = sess.run(summary, feed_dict)
                #testwriter.add_summary(test_summary_, counter)

            counter = counter + 1
            if counter % 1000 == 0:
                save(saver, params['model_path'], counter)
                print('Model Saved!!')

            counter = counter + 1


from tensorflow.python.framework import ops

ops.reset_default_graph()

global sess

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session()
graph = tf.get_default_graph()
train()
