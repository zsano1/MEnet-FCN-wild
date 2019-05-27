# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import ceil

import os
import logging
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

# in BGR order
DataSet_Mean = {'Taipei':[104.03, 104.93, 103.30],
                'Tokyo':[120.04, 121.09, 119.94],
                'Denmark':[126.77, 130.34, 127.37],
                'Roma':[113.44, 115.97, 114.38],
                'Rio':[115.81, 118.83, 116.11],
                'Cityscapes':[72.39, 82.91, 73.16],
                'Synthia':[63.31, 70.81, 80.35]}

class FCN8VGG:
    def __init__(self):
        '''
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "../cscap_dlvgg16.npy")
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
        
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4
        print("npy file loaded")
        '''
    def bn_scale_relu_conv(self, bottom, train, nout, conv_type = 'conv', name='BatchNorm', ks=3, stride=1, pad='SAME'):
        param_shape = bottom.get_shape()[-1]
        moving_decay = 0.9
        eps = 1e-5
        with tf.variable_scope(name):
            gamma1 = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1))
            beta1 = tf.get_variable('beat1', param_shape, initializer=tf.constant_initializer(0))
            ema1 = tf.train.ExponentialMovingAverage(moving_decay)
            axes1 = list(range(len(bottom.get_shape()) - 1))
            batch_mean1, batch_var1 = tf.nn.moments(bottom, axes1, name='moments')

            # 采用滑动平均更新均值与方差
            ema1 = tf.train.ExponentialMovingAverage(moving_decay)

            def mean_var_with_update():
                ema_apply_op1 = ema1.apply([batch_mean1, batch_var1])
                with tf.control_dependencies([ema_apply_op1]):
                    return tf.identity(batch_mean1), tf.identity(batch_var1)

            # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
            mean1, var1 = tf.cond(tf.equal(train, True), mean_var_with_update,
                                lambda: (ema1.average(batch_mean1), ema1.average(batch_var1)))

            # 最后执行batch normalization
            bn=tf.nn.batch_normalization(bottom, mean1, var1, beta1, gamma1, eps)

            relu1=tf.nn.relu(bn)

            kernel_size=[ks,ks]
            return tf.cond(tf.equal(conv_type, 'conv'), lambda :layers.conv2d(relu1,nout,kernel_size,stride,pad),
                           lambda :layers.conv2d_transpose(relu1,nout,kernel_size,stride,pad))


    def conv_bn_scale_relu(self, bottom, train, nout, conv_type = 'conv', name='BatchNorm', ks=3, stride=1, pad='SAME'):
        moving_decay = 0.9
        eps = 1e-5
        with tf.variable_scope(name):
            kernel_size=[ks,ks]

            conv=tf.cond(tf.equal(conv_type, 'conv'), lambda: layers.conv2d(bottom, nout, kernel_size, stride, pad),
                    lambda: layers.conv2d_transpose(bottom, nout, kernel_size, stride, pad))

            param_shape = conv.get_shape()[-1]

            ema = tf.train.ExponentialMovingAverage(moving_decay)
            axes = list(range(len(conv.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(conv, axes, name='moments')
            gamma = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1))
            beta = tf.get_variable('beat', param_shape, initializer=tf.constant_initializer(0))

            # 采用滑动平均更新均值与方差
            ema = tf.train.ExponentialMovingAverage(moving_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
            mean, var = tf.cond(tf.equal(train, True), mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))

            # 最后执行batch normalization
            bn=tf.nn.batch_normalization(conv, mean, var, beta, gamma, eps)

            relu=tf.nn.relu(bn)

            return relu

    def rs(self, data, target_size):#resize
        if data.shape==target_size.shape:
            return data
        else:
            original_size = data.shape
            resize_data = np.zeros([original_size[0], original_size[1], target_size[2], target_size[3]])
            x_step = target_size[2] / original_size[2]
            y_step = target_size[3] / original_size[3]
            for i in range(original_size[2]):
                for j in range(original_size[3]):
                    resize_data[:, :, i * x_step : (i + 1)* x_step, j * y_step : (j + 1) * y_step] = data[:, :, i, j, np.newaxis, np.newaxis]
            return resize_data

    #def build(self, rgb, task_labels, domain_label, batch_size, train=False, num_classes=19, city='Taipei', 
    def build(self, batch_size, img_w=512, img_h=256, train=False, num_classes=19, city='Taipei', 
              random_init_fc8=False, random_init_adnn=False, debug=False):
        
        if city=='syn2real':
            src_mean = DataSet_Mean['Synthia']
            tgt_mean = DataSet_Mean['Cityscapes']
        else:
            src_mean = DataSet_Mean['Cityscapes']
            tgt_mean = DataSet_Mean[city]
    
        self.rgb = tf.placeholder(tf.int32, shape = [batch_size, img_h, img_w, 3], name = 'rgb')
        self.task_labels = tf.placeholder(tf.float32, shape = [batch_size, img_h, img_w, 1], name = 'task_labels') 
        
        self.domain_labels = tf.placeholder(tf.float32, shape = [batch_size, int(img_h/8), int(img_w/8), 1], name = 'domain_labels')
        domain_label = tf.cast(tf.squeeze(self.domain_labels, [3]), tf.int32)
        # Convert RGB to BGR
        with tf.name_scope('Processing'):

            if train:
                src_planes = tf.split(self.rgb[:int(batch_size/2), :, :, :], 3, 3)
                r_src, g_src, b_src = [tf.cast(plane, tf.float32) for plane in src_planes]
                
                tgt_planes = tf.split(self.rgb[int(batch_size/2):, :, :, :], 3, 3)
                r_tgt, g_tgt, b_tgt = [tf.cast(plane, tf.float32) for plane in tgt_planes]
                bgr_src = tf.concat([b_src - src_mean[0],
                                     g_src - src_mean[1],
                                     r_src - src_mean[2]], 3)
            
                bgr_tgt = tf.concat([b_tgt - tgt_mean[0],
                                     g_tgt - tgt_mean[1],
                                     r_tgt - tgt_mean[2]], 3)
                self.bgr = tf.concat([bgr_src, bgr_tgt], 0)
            else:
                tgt_planes = tf.split(self.rgb, 3, 3)
                r_tgt, g_tgt, b_tgt = [tf.cast(plane, tf.float32) for plane in tgt_planes]
                
                self.bgr = tf.concat([b_tgt - tgt_mean[0],
                                      g_tgt - tgt_mean[1],
                                      r_tgt - tgt_mean[2]], 3)
        
        with tf.variable_scope('feature_extractor'):
            nout=32
            ks=3
            repeat=2
            result = self.conv_bn_scale_relu(self.bgr,train,nout,ks=ks)

            data_f=self.bn_scale_relu_conv(self.bgr,train,nout=1,ks=ks,name="conv_0")
            for i in range(repeat):
                result1=self.bn_scale_relu_conv(result,train,nout,ks=ks,name="conv_0_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout,ks=ks,name="conv_0_%s_1" %i)
                result=tf.add(result2,result)
            scale0=result
            scale0_m= self.bn_scale_relu_conv(scale0,train,nout=1,ks=ks,name="scale0")

            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=2*nout,ks=ks,stride=2,name="conv_1_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=2*nout,ks=1,stride=1,name="conv_1_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=2*nout,ks=ks,name="conv_1_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=2*nout,ks=ks,name="conv_1_%s" %i )
                result=tf.add(result2,result)
            scale1=result
            scale1_m =self.bn_scale_relu_conv(scale1,train,nout=1,ks=ks,name="scale1")

            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*4,ks=ks,stride=2,name="conv_2_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=4*nout,ks=1,stride=2,name="conv_2_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*4,ks=ks,stride=1,name="conv_2_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=4*nout,ks=ks,name="conv_2_%s" %i)
                result=tf.add(result2,result)
            scale2=result
            scale2_m=self.bn_scale_relu_conv(scale2,train,nout=1,ks=ks,name="scale2")

            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*8,ks=ks,stride=2,name="conv_3_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=8*nout,ks=1,stride=2,name="conv_3_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*8,ks=ks,stride=1,name="conv_3_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=8*nout,ks=ks,name="conv_3_%s" %i)
                result=tf.add(result2,result)
            scale3=result
            scale3_m=self.bn_scale_relu_conv(scale3,train,nout=1,ks=ks,name="scale3")

            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*16,ks=ks,stride=2,name="conv_4_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=16*nout,ks=1,stride=2,name="conv_4_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*16,ks=ks,stride=1,name="conv_4_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=16*nout,ks=ks,name="conv_4_%s" %i)
                result=tf.add(result2,result)
            scale4=result
            scale4_m=self.bn_scale_relu_conv(scale4,train,nout=1,ks=ks,name="scale4")

            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*32,ks=ks,stride=2,name="conv_5_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=32*nout,ks=1,stride=2,name="conv_5_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout*32,ks=ks,stride=1,name="conv_5_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=32*nout,ks=ks,name="conv_5_%s" %i)
                result=tf.add(result2,result)
            scale5=result
            scale5_m=self.bn_scale_relu_conv(scale3,train,nout=1,ks=ks,name="sclae5")

            result=self.bn_scale_relu_conv(result,train,nout=32*nout,ks=7,name="middle_white")
            # zsa：上行为白色，1*1的中间部分


            """
            self.conv1_1 = self._conv_layer(self.bgr, "conv1_1")
            self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)
            
            self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)
            
            self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)
        
            self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
            
            self.conv5_1 = self._dilated_conv_layer(self.conv4_3, "conv5_1", 2)
            self.conv5_2 = self._dilated_conv_layer(self.conv5_1, "conv5_2", 2)
            self.conv5_3 = self._dilated_conv_layer(self.conv5_2, "conv5_3", 2) 
            
            self.fc6 = self._dilated_conv_layer(self.conv5_3, "fc6", 4)
            if train:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5)
            
            self.fc7 = self._fc_layer(self.fc6, "fc7")
            if train:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5)
                """
        
        with tf.variable_scope('label_predictor'):
            result=self.bn_scale_relu_conv(result,train,nout=32*nout,conv_type='dconv',ks=7,name="dconv_1")
            scale5_u=self.bn_scale_relu_conv(result,train,nout=1,ks=ks,name="scale5_u")
            result=tf.concat(result,scale5)
            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=16*nout,conv_type='dconv',ks=4,stride=2,name="dconv_1_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=16*nout,conv_type='dconv',ks=2,stride=2,name="dconv_1_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=16*nout,ks=ks,name="dconv_1_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=16*nout,ks=ks,name="dconv_1_%s" %i)
                result=tf.add(result2,result)
            scale4_u=self.bn_scale_relu_conv(result,train,nout=1,name="scale4_u")

            result=tf.concat(result,scale4)
            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=8*nout,conv_type='dconv',ks=4,stride=2,name="dconv_2_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=8*nout,conv_type='dconv',ks=2,stride=2,name="dconv_2_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=8*nout,ks=ks,name="dconv_2_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=8*nout,ks=ks,name="dconv_1_%s" %i)
                result=tf.add(result2,result)
            scale3_u=self.bn_scale_relu_conv(result,train,nout=1,name="scale3_u")

            result=tf.concat(result,scale3)
            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=4*nout,conv_type='dconv',ks=4,stride=2,name="dconv_3_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=4*nout,conv_type='dconv',ks=2,stride=2,name="dconv_3_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=4*nout,ks=ks,name="dconv_3_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=4*nout,ks=ks,name="dconv_3_%s" %i)
                result=tf.add(result2,result)
            scale2_u=self.bn_scale_relu_conv(result,train,nout=1,name="scale2_u")

            result=tf.concat(result,scale2)
            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=2*nout,conv_type='dconv',ks=4,stride=2,name="dconv_4_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=2*nout,conv_type='dconv',ks=2,stride=2,name="dconv_4_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=2*nout,ks=ks,name="dconv_4_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=2*nout,ks=ks,name="dconv_4_%s" %i)
                result=tf.add(result2,result)
            scale1_u=self.bn_scale_relu_conv(result,train,nout=1,name="scale1_u")

            result=tf.concat(result,scale1)
            for i in range(repeat):
                if i==0:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout,conv_type='dconv',ks=4,stride=2,name="dconv_5_%s_0" %i)
                    result=self.bn_scale_relu_conv(result,train,nout=nout,conv_type='dconv',ks=2,stride=2,name="dconv_5_%s_1" %i)
                else:
                    result1=self.bn_scale_relu_conv(result,train,nout=nout,ks=ks,name="dconv_5_%s_0" %i)
                result2=self.bn_scale_relu_conv(result1,train,nout=nout,ks=ks,name="dconv_5_%s" %i)
                result=tf.add(result2,result)
            scale0_u=self.bn_scale_relu_conv(result,train,nout=1,name="scale0_u")

            #计算两层神经元的输出
            target_size=data_f.shape


            data=np.concatenate((self.rs(data_f,target_size),self.rs(scale0_m,target_size),
                                 self.rs(scale1_m,target_size), self.rs(scale2_m,target_size),
                                 self.rs(scale3_m,target_size), self.rs(scale4_m,target_size),
                                 self.rs(scale5_m,target_size),
                                 self.rs(scale0_u,target_size), self.rs(scale1_u,target_size),
                                 self.rs(scale2_u, target_size), self.rs(scale3_u,target_size),
                                 self.rs(scale4_u,target_size), self.rs(scale5_u,target_size)),axis=1)

            #先只看saliency部分
            self.metric=self.conv_bn_scale_relu(result,train,nout=16)
            #sal=self.conv_bn_scale_relu(result,train,nout=2)

            sal=self._fc_layer(self.fc7, "final2",
                                            relu=False)
            self.upsample=self._upscore_layer(sal,shape=self.bgr.get_shape(),
                                                num_classes=2,
                                                debug=debug, name='upsample',
                                                ksize=16, stride=8)
            self.pred_prob = tf.nn.softmax(self.upsample, name='pred_prob')
            self.pred_up = tf.argmax(self.upsample, dimension=3) # for inference




            """
            if random_init_fc8:
                self.final = self._score_layer(self.fc7, "final", num_classes)
            else:
                self.final = self._fc_layer(self.fc7, "final",
                                            num_classes=num_classes,
                                            relu=False)
            self.upsample = self._upscore_layer(self.final,
                                                shape=self.bgr.get_shape(),
                                                num_classes=num_classes,
                                                debug=debug, name='upsample',
                                                ksize=16, stride=8)
            self.pred_prob = tf.nn.softmax(self.upsample, name='pred_prob')
            self.pred_up = tf.argmax(self.upsample, dimension=3) # for inference
            """

        if train:
            
            #####################################
            ##############Task loss##############
            #####################################
        
            # 255 stands for ignored label
            task_labels = tf.cast(tf.gather(self.task_labels, tf.range(int(batch_size/2))), tf.int32)
            task_labels = tf.squeeze(task_labels, [3]) 
            mask = tf.constant(255, shape=task_labels.get_shape().as_list())
            mask = tf.cast(tf.not_equal(task_labels, mask), tf.int32)
            valid_pixel_num = tf.cast(tf.reduce_sum(mask), tf.float32)
            
            # calculating the accuracy
            prediction = tf.cast(tf.gather(self.pred_up, tf.range(int(batch_size/2))), tf.int32) 
            self.task_accur = tf.div(tf.reduce_sum(tf.cast(tf.equal(prediction,task_labels),tf.float32)), valid_pixel_num)
            
            # calculating the loss
            task_labels = tf.multiply(task_labels, mask) # remove those pixels labeled as 255
            epsilon = tf.constant(value=1e-4)
            logits = tf.gather(self.upsample, tf.range(int(batch_size/2))) + epsilon
                        
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = task_labels, logits = logits, name = 'cross_entropy_per_example')
            cross_entropy = tf.multiply(cross_entropy, tf.cast(mask, dtype=tf.float32)) # igonre the loss of pixels labeled by 255
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean') # average over the batch
            tf.add_to_collection('losses', cross_entropy_mean)
            self.task_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        ######################################
        ###########adversarial net############
        ######################################

        # GA stands for global alignment    
        # CA stands for class-specific alignment
        
        if random_init_adnn and train:
            
            with tf.variable_scope('global_alignment'):
                    
                # discriminator for global alignment
                self.GA_adnn1 = self._score_layer(self.fc7, "GA_adnn1")
                self.GA_adnn1 = tf.nn.relu(self.GA_adnn1)
                self.GA_adnn1 = tf.nn.dropout(self.GA_adnn1, 0.5)
                
                self.GA_adnn2 = self._score_layer(self.GA_adnn1, "GA_adnn2")
                self.GA_adnn2 = tf.nn.relu(self.GA_adnn2)
                self.GA_adnn2 = tf.nn.dropout(self.GA_adnn2, 0.5)

                self.GA_adnn3 = self._score_layer(self.GA_adnn2, "GA_adnn3")
                GA_domain_pred = tf.cast(tf.argmax(self.GA_adnn3, dimension=3), tf.int32)
                self.GA_domain_accur = tf.reduce_mean(tf.cast(tf.equal(GA_domain_pred, domain_label), tf.float32))
                GA_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     labels = domain_label,
                                     logits = self.GA_adnn3)
                GA_entropy_inv = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     labels = 1-domain_label,
                                     logits = self.GA_adnn3)
                
                self.GA_domain_loss = tf.reduce_mean(GA_entropy, name='GA_loss')
                self.GA_domain_loss_inv = tf.reduce_mean(GA_entropy_inv, name='GA_loss_inv')


        t_vars = tf.trainable_variables()
        
        self.f_vars = [var for var in t_vars if 'feature_extractor' in var.name]
        self.y_vars = [var for var in t_vars if 'label_predictor' in var.name]
        
        if train:
            self.ga_vars = [var for var in t_vars if 'global_alignment' in var.name]

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool
    '''
    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu
    
    def _dilated_conv_layer(self, bottom, name,rate):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            #conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.atrous_conv2d(bottom, filt, rate, padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    '''
    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'final': #'score_fr':
                #name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 19],
                                                  num_classes=num_classes)
            elif name =="final2":
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 2])
            else: #fc7
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes=20):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name.split('_')[-1] in ['adnn1', 'adnn2']:
                shape = [1, 1, in_features, 1024]
            elif name.split('_')[-1] == 'adnn3':
                shape = [1, 1, in_features, 2]
            elif name.split('_')[-1] == 'wgan':
                shape = [1, 1, in_features, 1]

            if name == "final": #"score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            elif name.split('_')[-1] in ['adnn1', 'adnn2', 'adnn3', 'wgan']:
                stddev = 0.001
            #elif name.split('_')[-1] in ['adnn2', 'wgan']:
            #    stddev = 0.001
            # Apply convolution
            w_decay = self.wd

            weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                       decoder=True)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([shape[3]], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [int(shape[0]), int(shape[1]), int(shape[2]), num_classes]
            
            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter('upsample',f_shape)
            self._add_wd_and_summary(weights, self.wd, "fc_wlosses")
            deconv = tf.nn.conv2d_transpose(bottom, weights, new_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv
    '''
    def get_deconv_filter(self, name, f_shape):
        """
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear
        """
        weights_2 = np.zeros(f_shape)
        w_in = self.data_dict[name][0]
        for i in range(f_shape[2]):
            weights_2[:, :, i, i] = w_in[:,:,i]
        #weights_2 = np.tile(np.expand_dims(weights_2, 2), (1, 1, 19, 1))
        
        weights = weights_2.reshape(f_shape)

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.nn.l2_loss(var)* self.wd
            tf.add_to_collection('losses', weight_decay)
        
        return var

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        #value = self.data_dict[name][0][0][0][0][0]
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        #print('Layer value: %s' % str(value))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.nn.l2_loss(var)* self.wd
            tf.add_to_collection('losses', weight_decay)
        _variable_summaries(var)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'final': #'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        _variable_summaries(var)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.nn.l2_loss(var)* self.wd
            tf.add_to_collection('losses', weight_decay)
        _variable_summaries(var)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight
    '''
    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.nn.l2_loss(var)* wd
            if not decoder:
                tf.add_to_collection('losses', weight_decay)
            else:
                tf.add_to_collection('dec_losses', weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, wd, collection_name="losses"):
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.nn.l2_loss(var)* wd
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var
    


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #tf.histogram_summary(tensor_name + '/activations', x)
    #tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            #tf.scalar_summary(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            #tf.scalar_summary(name + '/sttdev', stddev)
            #tf.scalar_summary(name + '/max', tf.reduce_max(var))
            #tf.scalar_summary(name + '/min', tf.reduce_min(var))
            #tf.histogram_summary(name, var)
