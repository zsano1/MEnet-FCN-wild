import tensorflow as tf
#from config import *
import numpy as np
import tensorflow as tf
import time
import sys
from model import FCN8VGG
"""

class MatricLoss():
    def _init_(selfself,):
        pass
    def calculate(self,x,weight_path):
        model=FCN8VGG.(weight_path)

        self.num_cases, self.channels, self.height, self.width = model.metric.shape
        '''
        tmp_scale0_data = self.scale0_data.transpose(2, 3, 0, 1)
        tmp_scale0_labels = self.scale0_labels.transpose(2, 3, 0, 1)
        '''
        tmp_scale0_data=model.metric
        tmp_scale0_labels=model.task_labels
        
        num_foreground_example = np.sum(self.scale0_labels.reshape(self.num_cases, self.height * self.width), axis = 1)
        num_foreground_example[np.where(num_foreground_example == 0)] = 1
        ............
        tmp_foreground_example = self.scale0_data * \
                                    np.tile(self.scale0_labels, (1, self.channels, 1, 1))      #what?????
        tmp_foreground_example = tmp_foreground_example.reshape(self.num_cases, self.channels, self.height * self.width)
        tmp_foreground_example = tmp_foreground_example.sum(axis = 2).transpose(1, 0) * 1. / num_foreground_example
        tmp_foreground_example = tmp_foreground_example.transpose(1, 0)
        
        #foreground
        num_foreground_example=np.sum(model.task_labels.reshape(self.num_cases, self.height * self.width), axis = 1)
        num_foreground_example[np.where(num_foreground_example == 0)] = 1

        tmp_foreground_example=model.metric * \
                                    np.tile(model.task_labels, (1, self.channels, 1, 1))      #what?????
        tmp_foreground_example=tmp_foreground_example.reshape(self.num_cases,self.channels, self.height * self.width)
        tmp_foreground_example=tmp_foreground_example.sum(axis=2)*1./num_foreground_example



        #background
        num_background_example=np.sum(1-model.task_labels.reshape(self.num_cases, self.height * self.width), axis = 1)
        num_background_example[np.where(num_foreground_example == 0)] = 1

        tmp_background_example=model.metric * \
                                    np.tile(1-model.task_labels, (1, self.channels, 1, 1))      #what?????
        tmp_background_example=tmp_foreground_example.reshape(self.num_cases,self.channels, self.height * self.width)
        tmp_background_example=tmp_foreground_example.sum(axis=2)*1./num_foreground_example

        #fore_back_loss
        foreground_loss = ((model.metric - tmp_foreground_example) ** 2 - (
                            model.metric - tmp_background_example) ** 2) * \
                            np.tile(model.task_labels, (1, 1, 1, self.channels))

        foreground_loss = foreground_loss.sum(axis=3)

        background_loss = ((model.metric - tmp_background_example) ** 2 - (
                            model.metric - tmp_foreground_example) ** 2) * \
                            np.tile(1-model.task_labels, (1, 1, 1, self.channels))

        background_loss = background_loss.sum(axis=3)

        total_loss=foreground_loss+background_loss
        total_loss = total_loss[:,:,:,np.newaxis].transpose(2, 3, 0, 1)
        return total_loss.mean()


from ccnn import constraintloss
#from tensorflow.python.framework import ops
import pdb

class WeakLoss():
    def __init__(self, src_data_name):

    def calculate(self,x): # forward
        #x is in type numpy 
        D = x.shape[-1]; H = x.shape[1]; W = x.shape[2];
        ds = self.downsample_rate 
        x = x[:, 0:H:ds, 0:W:ds, :] #subsample for coarse output to reduce computing
        batch_size = int(x.shape[0]/2) 
        #bottom = bottom[batch_size:,...] # only get target
        self.diff = []
        loss,w = 0,0
        
        for i in range(batch_size, 2*batch_size): #iter over batch_size

            if (not self.semi_supervised):         # weakly-supervised downsampled training
                # Setup bottoms
                f = np.ascontiguousarray(x[i].reshape((-1,D)))       # f : height*width x channels
                q = np.exp(f-np.max(f,axis=1)[:,None])                              # expAndNormalize across channels
                q/= np.sum(q,axis=1)[:,None]

                # Setup the constraint softmax
                csm = constraintloss.ConstraintSoftmax(self.hardness)
                # calculate image_level label
                pred = np.argmax(x[i], axis=-1)
                pred_avg = np.zeros((D))
                for cla in range(D):
                    pred_avg[cla] = np.mean(pred == cla)
                L = pred_avg > (0.1* self.lower_ten) 
                
                csm.addZeroConstraint( (~L).astype(np.float32) )
                
                # Add Positive label constraints
                for cla in np.flatnonzero(L):
                    #if cla > 6:
                    #    break # early stop for debug
                    v_onehot = np.zeros((D)).astype(np.float32)
                    v_onehot[cla] = 1 
                    csm.addLinearConstraint(  v_onehot, float(self.avg[cla]), self.fg_slack ) # lower bound
                    csm.addLinearConstraint( -v_onehot, float(-self.upper_ten[cla]) ) # upper bound
                
                # Run constrained optimization
                #start_time = time.clock()
                p = csm.compute(f)
                #print('opt time', time.clock()-start_time) 
                
                if self.normalization:
                    self.diff.append( ((q-p).reshape(x[i].shape)) / np.float32(f.shape[0]))      # normalize by (f.shape[0])
                else:
                    self.diff.append( ((q-p).reshape(x[i].shape)) )      # unnormalize
            
            if self.normalization:          
                loss += (np.sum(p*np.log(np.maximum(p,1e-10))) - np.sum(p*np.log(np.maximum(q,1e-10))))/np.float32(f.shape[0])    # normalize by (f.shape[0])
            else:
                loss += (np.sum(p*np.log(np.maximum(p,1e-10))) - np.sum(p*np.log(np.maximum(q,1e-10))))    # unnormalize

        loss /= batch_size

        self.diff = np.array(self.diff)
        #pdb.set_trace()
        self.diff = self.diff.repeat(ds, axis=1).repeat(ds, axis=2)
        ### mapping to original image size
        patch = np.zeros((batch_size,ds,ds,D)).astype(np.float32); patch[:,0,0,:] = 1;
        valid_mask = np.tile(patch, (1,H/ds,W/ds,1))
        self.diff = self.diff * valid_mask
        #pdb.set_trace()
        self.diff = np.concatenate((np.zeros_like(self.diff), self.diff),0)
        
        # process grad for size constrain: reduce by 0.1 for major class
        grad_reduce_map = np.ones(self.diff.shape).astype(np.float32)
        grad_reduce_map[:,:,:,0] = 0.1; grad_reduce_map[:,:,:,2]=0.1; grad_reduce_map[:,:,:,8]=0.1;
        self.diff = self.diff * grad_reduce_map # gradient for backward
        return loss.astype(np.float32)
"""

