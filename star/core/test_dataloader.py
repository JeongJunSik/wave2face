"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import shutil
import os, random, cv2, argparse
import time


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

def get_test_loader(root, img_size=256,
                     batch_size=1,  num_workers=0):

    dataset = lrs2_Dataset()

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True,
                           shuffle= False,
                           drop_last=False)

class InputFetcher:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            #print("@@@ fetch_input try")
            x, y, z, w = next(self.iter)
            #print("@@ x.shape: y.shape: ", x.shape, y.shape)
        except (AttributeError, StopIteration):
            #print("@@@ fetch_input except")
            #print("@@@ loader length: ", len(self.loader))
            self.iter = iter(self.loader)
            #print("@@@@ type(self.iter): ",type(self.iter))
            x, y, z, w = next(self.iter)
            #print("@@ x.shape: y.shape: ", x.shape, y.shape)
        return x, y, z, w

    def __len__(self):
        return len(self.loader)

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y, z, w = self._fetch_inputs()
        #print("@@ x.shape: y.shape: ", x.shape, y.shape)
        inputs = Munch(gt_land=x, gt=y, gt_mask=z, prior=w)
        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})

class lrs2_Dataset(Dataset):
    def __init__(self):
        # self.all_videos = get_image_list('/root/project/public/data/lrs2/lrs2_landmark_210406', split) # make video file list ex)00001
        # self.all_videos2 = get_image_list('/root/project/public/data/lrs2/lrs2_preprocessed', split)
        # basename: basename('/root/project/.../landmark/.../0010.jpg') = 0010.jpg
        self.all_landmark_root_path = '/root/project/public/data/lrs2/lrs2_landmark_rasterized/6220056249377872413/00016/'
        self.all_videos_root_path = '/root/project/public/data/lrs2/lrs2_preprocessed/6220056249377872413/00016/'
        self.all_videos_mask_root_path = '/root/project/public/data/lrs2/lrs2_frame_mask/6220056249377872413/00016/'
        self.img_size=128
        self.imgResize=True
        self.syncnet_T = 5



        if not os.path.isdir(self.all_landmark_root_path):
            print("wrong path lrs2_Dataset.all_lanmarks. the path doesn't exist : ",self.all_landmark_root_path )
        if not os.path.isdir(self.all_videos_root_path):
            print("wrong path lrs2_Dataset.all_videos_root_path. the path doesn't exist: ", self.all_videos_root_path)
        if not os.path.isdir(self.all_videos_mask_root_path):
            print("wrong path lrs2_Dataset.all_videos_root_path. the path doesn't exist: ", self.all_videos_mask_root_path)

    
        """ extract landmark and image random base t'th frame """
        self.landmark_names = list(glob(join(self.all_landmark_root_path , '*.jpg'))) # landmark file

        #print(os.listdir(vidname))
        if len(self.landmark_names) <= 3 * self.syncnet_T: #이미지 개수가 15개 이하면 패스
            print("Thhis clip is too short (more than 15.)")
            exit()

        #임의의 비디오에서 임의의 landmark 프레임을 추출
        full_frame_len = len(self.landmark_names)
        number_list = []
        for i in range(full_frame_len):
            number_list.append(self.get_frame_id(self.landmark_names[i]))
        number_list = np.array(number_list)
        number_list = np.argsort(number_list)

        self.landmark_names = np.array(self.landmark_names)[number_list] # sorted array.

    def get_frame_id(self, frame):
        # dirname(path): forepart of basename.
        #print("@@ get_frame_id: ", basename(frame).split('.')[0])
        return int(basename(frame).split('.')[0]) # makek a frame list ex) 01 02 03

    def get_window_rasterized_landmark(self, start_frame): # start frame + syncnet_T 만큼 window list안에 경로 저우 랜드마크
        start_id = self.get_frame_id(start_frame)
        #print("@@@ rasterized start_frame: ", start_frame)
        vidname = dirname(start_frame)
        #print("@@@ vidname: ", vidname)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            #print("#### frame: ", frame)
            if not isfile(frame):
                #print("@@@@@@@@@@@@@@@@@@@@@@@ rasterized frame is None: ", frame)
                return None
            window_fnames.append(frame)
        return window_fnames
    
    def get_window_image(self,start_frame):# image에대한 윈도우
        #print("#### iamge start_frame: ", start_frame)
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            #print("#### frame: ", frame)
            if not isfile(frame):
                #print("############### image frame is None: ", frame)
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window_rasterized_landmark(self, window_fnames): #read a image according to the window frame (5 frames)
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)[::,::,::-1] #  BGR to RGB

            if img is None:
                #print("@@ ratster image none")
                return None
            else:
                if self.imgResize:
                    img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                else:
                    if img.shape[0]<256:
                        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

            window.append(img)
        return window
    
    def read_window_image(self, window_fnames): #read a image according to the window frame (5 frames)
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)[::,::,::-1] #  BGR to RGB
            if img is None:
                return None
            else:
                if self.imgResize:
                    img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                else:
                    if img.shape[0]<256:
                        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
            window.append(img)
        return window

    def prepare_window(self, window, window_image):
        # 3 x T x H x W
        # 5 x 68 x 2  -> x y 순서임
        # window_image 5 x h x w x c 
        #normalize 시켜주는데 reference image shape 으로
        b=len(window_image)
        x=[]
        for i in range(b):
            window[i][:,0]=window[i][:,0] / window_image[i].shape[1]
            window[i][:,1]=window[i][:,1] / window_image[i].shape[0]
            x.append(window[i])
        x=np.asarray(x)
        #x = np.transpose(x, (3, 0, 1, 2))# W x 3 x T x H

        return x

    def __len__(self):
        return len( list(glob(join(self.all_landmark_root_path , '*.jpg'))) )//self.syncnet_T 

    def __getitem__(self, idx):
        #start = time.time()
        while 1:
            landname = self.all_landmark_root_path #그 해당 index의 vidname
            vidname= self.all_videos_root_path
            vidname_mask = self.all_videos_mask_root_path
            
            landmark_name = self.landmark_names[idx*self.syncnet_T ]
            
            ID = self.get_frame_id(landmark_name)#landmark id 찾고
            img_name = vidname +'/'+str(ID)+'.jpg' #경로 저장
            img_name_mask = vidname_mask +'/'+str(ID)+'.jpg' #경로 저장

            """ get window """
            landmark_window = self.get_window_rasterized_landmark(landmark_name)
            image_window=self.get_window_image(img_name)
            image_window_mask =self.get_window_image(img_name_mask)

            if landmark_window is None or image_window  is None or image_window_mask is None: # 둘다 None 이면 생략         
                print("@@@ 1 window error")      
                exit()


            """ read landmark window """
            landmarks = self.read_window_rasterized_landmark(landmark_window) #read window in the window path 
            if landmarks is None:
                print("@@@ 2 landmark load continue")    
                exit()
                
            """ read frame window """
            images = self.read_window_image(image_window)
            images_mask = self.read_window_image(image_window_mask)
            if images is None or  images_mask is None:
                print("@@@ 3 image load continue") 
                exit()


            """ prepare for return """
            b=len(landmark_window)
            
            A=[]
            A.append(img_name)

            landmark_GT = np.asarray(landmarks)

            img_GT = np.asarray(images)
            img_GT_mask = np.asarray(images_mask)

            GT =  torch.FloatTensor( img_GT ).permute(0,3,1,2)/255.0
            GT_landmark =  torch.FloatTensor( landmark_GT ).permute(0,3,1,2)/255.0
            GT_mask =  torch.FloatTensor( img_GT_mask ).permute(0,3,1,2)/255.0
            
            """ normalize """
            GT = 2*GT -1 # (landmark, GT, GT_mask)
            GT_landmark = 2*GT_landmark -1
            GT_mask = 2*GT_mask -1
            
            return GT_landmark, GT, GT_mask, torch.Tensor(0)