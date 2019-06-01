from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import numpy as np
import cv2
import pickle

class Reader(object):
    
    def __init__(self, src_data_path, tgt_data_path, input_width=512, input_height=256, batch_size = 64, suffle=True):
        

        if not os.path.isfile(src_data_path) or not os.path.isfile(tgt_data_path):
            print('No data!!!')
            exit(1)

        self.src_images, self.src_labels = self.read_file_list(src_data_path)
        self.tgt_images, self.tgt_labels = self.read_file_list(tgt_data_path)
        self.src_ind = np.arange(len(self.src_images))
        self.tgt_ind = np.arange(len(self.tgt_images))
        self.src_step = 0
        self.tgt_step = 0
        self.IMG_W = input_width
        self.IMG_H = input_height
        self.batch_size = batch_size
        self.shuffle = True
    
    def read_file_list(self, path): #path is .txt in format: image_path label_path 
        images = []
        labels = []
        for line in open(path):
            images.append(line.split(' ')[0])
            labels.append(line.split(' ')[1].rstrip('\n'))
        
        return images, labels

    def next_train(self):
        
        if (self.src_step+1)*(self.batch_size/2) > len(self.src_images):
            self.src_step = 0
        
        if (self.tgt_step+1)*(self.batch_size/2) > len(self.tgt_images):
            self.tgt_step = 0

        if self.shuffle and self.src_step == 0:
            print('randomly shuffle the source data') 
            np.random.shuffle(self.src_ind)
        
        if self.shuffle and self.tgt_step == 0:
            print('randomly shuffle the target data')   
            np.random.shuffle(self.tgt_ind)

        src_ind_this_batch = self.src_ind[int(self.src_step*(self.batch_size/2)): 
                                          int((self.src_step+1)*(self.batch_size/2))]
        tgt_ind_this_batch = self.tgt_ind[int(self.tgt_step*(self.batch_size/2)): 
                                          int((self.tgt_step+1)*(self.batch_size/2))]
        
        self.src_step += 1
        self.tgt_step += 1

        src_images_this_batch = []
        src_labels_this_batch = []
        
        tgt_images_this_batch = []
        tgt_labels_this_batch = []

        for i in range(len(src_ind_this_batch)):
            src_image = cv2.imread(self.src_images[src_ind_this_batch[i]])[:,:,::-1]
            src_image = cv2.resize(src_image, (self.IMG_W, self.IMG_H))
            src_label = cv2.imread(self.src_labels[src_ind_this_batch[i]], 0)[:,:,np.newaxis]
            src_label = cv2.resize(src_label, (self.IMG_W, self.IMG_H),interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]
            src_images_this_batch.append(src_image)
            src_labels_this_batch.append(src_label)
            
            # load the target image
            tgt_image = cv2.imread(self.tgt_images[tgt_ind_this_batch[i]])[:,:,::-1]
            tgt_image = cv2.resize(tgt_image, (self.IMG_W, self.IMG_H))
            # create the dummy labels for target domain images
            tgt_label = np.zeros((tgt_image.shape[0], tgt_image.shape[1], 1))
                        
            tgt_images_this_batch.append(tgt_image)
            tgt_labels_this_batch.append(tgt_label)
        
        src_images_this_batch = np.array(src_images_this_batch).astype(np.float32)
        src_labels_this_batch = np.array(src_labels_this_batch).astype(np.float32)
        src_domain_labels_this_batch = np.zeros((src_labels_this_batch.shape[0],
                                                 int(src_labels_this_batch.shape[1]),
                                                 int(src_labels_this_batch.shape[2]),
                                                 src_labels_this_batch.shape[3]))

        tgt_images_this_batch = np.array(tgt_images_this_batch).astype(np.float32)
        tgt_labels_this_batch = np.array(tgt_labels_this_batch).astype(np.float32)

        tgt_domain_labels_this_batch = np.ones((tgt_labels_this_batch.shape[0],
                                                 int(tgt_labels_this_batch.shape[1]),
                                                 int(tgt_labels_this_batch.shape[2]),
                                                 tgt_labels_this_batch.shape[3]))
        images_this_batch = np.vstack((src_images_this_batch, tgt_images_this_batch))
        task_labels_this_batch = np.vstack((src_labels_this_batch, tgt_labels_this_batch))
        domain_labels_this_batch = np.vstack((src_domain_labels_this_batch, tgt_domain_labels_this_batch))
        
        return images_this_batch, task_labels_this_batch, domain_labels_this_batch

def main(argv = None):
    reader = Reader('data/Cityscapes.txt', 'data/Taipei.txt', batch_size = 8)
    for i in range(100):
        A,B,C = reader.next_train()
if __name__ == '__main__':
    main()

