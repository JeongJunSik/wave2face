from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from L2L6 import *
#from models import SyncNet_color as SyncNet
#from models import Wav2Lip as Wav2Lip
#import audio
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from glob import glob
import shutil
import os, random, cv2, argparse
import time
from hparams import hparams, get_image_list
from random import randrange

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')



parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)


parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
class lrs(Dataset):
    def __init__(self, split):
        self.all_videos = get_image_list('/root/project/public/data/lrs2/lrs2_landmark_210406', split) # make video file list ex)00001
        self.all_videos2 = get_image_list('/root/project/public/data/lrs2/lrs2_preprocessed', split)
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0]) # makek a frame list ex) 01 02 03

    def get_window(self, start_frame): # start frame + syncnet_T 만큼 window list안에 경로 저우 랜드마크
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.npy'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
    
    def get_window_image(self,start_frame):# image에대한 윈도우
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames): #read a image according to the window frame (5 frames)
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            landmark = np.load(fname)
            if landmark is None:
                return None
            

            window.append(landmark)

        return window
    
    def read_window2(self, window_fnames): #read a image according to the window frame (5 frames)
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame): 
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(25))) # 80 * (frame_num / 25)
        
        end_idx = start_idx + syncnet_mel_step_size # start_idx + 16

        return spec[start_idx : end_idx, :] #spec -> melspectogram.T

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels
    def mask_landmark(self,landmark):
        
        
        x_lip_argmin = np.argmin(landmark[48:68,0])
        x_lip_argmax = np.argmax(landmark[48:68,0])
        y_lip_argmin = np.argmin(landmark[48:68,1])
        y_lip_argmax = np.argmax(landmark[48:68,1])

        # np.where(  landmark[0:17,0][  np.where((landmark[0:17,0] >= x_min))  ] <= x_max    )
        #print(np.where( landmark[0:17,1] >= landmark[48:68,1][y_lip_argmin], landmark[0:17,0], 10000) )
        x_chin_argmin = np.argmin(  np.where( landmark[0:17,1] >= landmark[48:68,1][y_lip_argmin], landmark[0:17,0], 10000)   )
        x_chin_argmax = np.argmax(  np.where( landmark[0:17,1] >= landmark[48:68,1][y_lip_argmin], landmark[0:17,0], -10000 )   )
        y_chin_argmin = np.argmin(  np.where( landmark[0:17,1] >= landmark[48:68,1][y_lip_argmin], landmark[0:17,1], 10000 )   )
        y_chin_argmax = np.argmax(  np.where( landmark[0:17,1] >= landmark[48:68,1][y_lip_argmin], landmark[0:17,1], -10000 )   )


        """ x axis """
        if landmark[0:17,0][x_chin_argmin] <= landmark[48:68,0][x_lip_argmin]  :
            x_min = int(landmark[0:17,0][x_chin_argmin])
        else:
            x_min = int(landmark[48:68,0][x_lip_argmin])

        if landmark[0:17,0][x_chin_argmax] >= landmark[48:68,0][x_lip_argmax]  :
            x_max = int(landmark[0:17,0][x_chin_argmax])
        else:
            x_max = int(landmark[48:68,0][x_lip_argmax])

        """ y axis """
        y_min = int(landmark[48:68,1][y_lip_argmin])
        y_max = int(landmark[0:17,1][y_chin_argmax])

        chin_zero_x_index = np.where(  np.where( (landmark[0:17,0] >= x_min), landmark[0:17,0], +10000   ) <= x_max   )
        chin_zero_y_index = np.where(  np.where( (landmark[0:17,1] >= y_min), landmark[0:17,1], +10000   ) <= y_max  )
        lip_zero_x_index = np.where(   np.where( (landmark[48:68,0] >= x_min), landmark[48:68,0], +10000 ) <= x_max  )
        lip_zero_y_index = np.where(   np.where( (landmark[48:68,1] >= y_min), landmark[48:68,1], +10000 ) <= y_max  )

        landmark[0:17,:][chin_zero_x_index,:] = 0
        landmark[0:17,:][chin_zero_y_index,:] = 0
        landmark[48:68,:][lip_zero_x_index,:] = 0
        landmark[48:68,:][lip_zero_y_index,:] = 0
        
        return landmark
    def prepare_window(self, window,window_image):
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
        return len(self.all_videos)

    def __getitem__(self, idx):
        #start = time.time()
        while 1:
            
            idx = random.randint(0, len(self.all_videos) - 1) # all videos의 개수 중에서 랜덤의 index
            
            vidname = self.all_videos[idx] #그 해당 index의 vidname
            mfccname= self.all_videos2[idx]
            landmark_names = list(glob(join(vidname, '*.npy'))) # landmark file
            
            #print(os.listdir(vidname))
            if len(landmark_names) <= 3 * syncnet_T: #이미지 개수가 15개 이하면 패스
                
                continue
            
             #임의의 비디오에서 임의의 이미지 프레임을 추출
            
            Q = random.choice(range(len(landmark_names)))
            landmark_name=landmark_names[Q]
            
            ID = self.get_frame_id(landmark_name)#landmark id 찾고
            img_name=mfccname+'/'+str(ID)+'.jpg' #경로 저장
            
            
            
            wrong_landmark_name = random.choice(landmark_names) #틀린 이미지 프레임도 새롭게 추출
            ID2=self.get_frame_id(wrong_landmark_name)
            img_name2=mfccname+'/'+str(ID2)+'.jpg'
            
            
            while wrong_landmark_name == landmark_name:
                #print("1")
                wrong_landmark_name = random.choice(landmark_names)
                ID2=self.get_frame_id(wrong_landmark_name)# 만약 운이 좋아 같다면 새롭게 랜덤 추출
                img_name2=mfccname+'/'+str(ID2)+'.jpg'
            
            window_fnames = self.get_window(landmark_name)
            
            image_fnames=self.get_window_image(img_name)
            
            image_fnames2=self.get_window_image(img_name2)
            #해당 이미지하나가 start_frame이 되어 window 추출 (5)
            wrong_window_fnames = self.get_window(wrong_landmark_name) #wrong도 똑같음
            
           
            if window_fnames is None or wrong_window_fnames is None or image_fnames is None or image_fnames2 is None: # 둘다 None 이면 생략               
                
                
                continue

            window = self.read_window(window_fnames) #read window in the window path 
            
            if window is None: 
                
                continue
            
            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                
                continue
                
                
            window_image=self.read_window2(image_fnames)
            if window_image is None:
                
                continue
                
            window_image2=self.read_window2(image_fnames2)
            if window_image2 is None:
                
                continue
            
            try:
                #start = time.time()
                mfccpath = join(mfccname, "mfcc.npy")
                
                mfcc = np.load(mfccpath)
                #print("audo load time :", time.time() - start)
                orig_mel=mfcc
                #start = time.time()
                #orig_mel = audio.melspectrogram(mfcc).T #make melspectogram
                # print("melsepctogram, time :", time.time() - start)
            except Exception as e:
                
                continue


            mel = self.crop_audio_window(orig_mel.copy(), landmark_name) #crop according to the window
            
            if (mel.shape[0] != syncnet_mel_step_size): #mel.shape[0] != 16
                
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), landmark_name)
            if indiv_mels is None:
                

                continue
            
            
            
            b=len(window)
            
            A=[]
            A.append(img_name)

            landmark_GT=np.asarray(deepcopy(window))
            landmark_prior=np.asarray(deepcopy(wrong_window))
            
            for i in range(b):
                window[i][48:68,:] = 0
                
                
            
        
            
            window = self.prepare_window(deepcopy(window),window_image)
            Masked=np.asarray(deepcopy(window))
            GT =  self.prepare_window(deepcopy(landmark_GT),window_image)
            
            
                
            #window[:, :, window.shape[2]//2:] = 0.
            
            
            wrong_window = self.prepare_window(deepcopy(landmark_prior),window_image2)
            x = np.concatenate([window, wrong_window], axis=2)
            
            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = GT
            #y = np.concatenate([GT, wrong_window], axis=1)
            y = torch.FloatTensor(y)
            
            #print("total get item, time :", time.time() - start)
            return x, indiv_mels, mel, y,landmark_prior,landmark_GT,window_image[0].shape,A[0],Masked


################################################################################

def save_sample_images(landmark_pred,landmark_GT,image, landmark_prior,GT,Masked,global_step, checkpoint_dir):
    #print(landmark_pred)
    #print(landmark_GT)
    #import pdb;pdb.set_trace()
    landmark_pred=landmark_pred.detach().cpu()
    plt.scatter(landmark_pred[0][0:68,0], -landmark_pred[0][0:68,1])
    plt.savefig('/root/project/sh/checkpoint/eval/pred1_{}.jpg'.format(global_epoch))
    plt.close()
    
    plt.scatter(landmark_pred[0][68:,0], -landmark_pred[0][68:,1])
    plt.savefig('/root/project/sh/checkpoint/eval/pred2_{}.jpg'.format(global_epoch))
    plt.close()
    #import pdb;pdb.set_trace()
    landmark_GT=landmark_GT.detach().cpu()
    plt.scatter(landmark_GT[0][:,0], -landmark_GT[0][:,1])
    plt.savefig('/root/project/sh/checkpoint/eval/GT_{}.jpg'.format(global_epoch))
    plt.close()
    GT=GT.detach().cpu()
    plt.scatter(GT[0][:,0], -GT[0][:,1])
    plt.savefig('/root/project/sh/checkpoint/eval/GT_unnorm_{}.jpg'.format(global_epoch))
    plt.close()
    landmark_prior=landmark_prior.detach().cpu()
    plt.scatter(landmark_prior[0][:,0], -landmark_prior[0][:,1])
    plt.savefig('/root/project/sh/checkpoint/eval/prior_{}.jpg'.format(global_epoch))
    plt.close()
    Masked=Masked.detach().cpu()
    plt.scatter(Masked[0][:,0], -Masked[0][:,1])
    plt.savefig('/root/project/sh/checkpoint/eval/Masked_{}.jpg'.format(global_epoch))
    plt.close()
    img = cv2.imread(image)
    cv2.imwrite('/root/project/sh/checkpoint/eval/image_{}.jpg'.format(global_epoch), img)
    a=landmark_pred[0]
    b=landmark_GT[0]
    a=torch.round(a)
    b=torch.round(b)
    print((a-b).mean())
def save_sample_npy(landmark_pred,landmark_GT,y,x,image, global_step, checkpoint_dir):
    #print(landmark_pred)
    #print(landmark_GT)
    img = cv2.imread(image)
    cv2.imwrite('/root/project/sh/checkpoint/nump/img_{}.jpg'.format(global_step), img)

    
    landmark_pred=landmark_pred.detach().cpu()
    a=landmark_pred[0]
    
    a[:,0]=a[:,0].clone()*x
    a[:,1]=a[:,1].clone()*y
    #a=torch.round(a)

    np.save('/root/project/sh/checkpoint/nump/pred_{}'.format(global_step),a)
    np.save('/root/project/sh/checkpoint/nump/GT_{}'.format(global_step),landmark_GT[0])
    
    
    

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss




recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    Check=True
    is_best=False
    L1=987654321.
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_l1_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt,landmark_prior,landmark_GT,shape,image,Masked) in prog_bar:
            #import pdb;pdb.set_trace()
            
            #print(len(image))
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            
            g = model(indiv_mels, x)
            
           
          
            
            l1loss = recon_loss(g, gt)

            loss = l1loss
            loss.backward()
            optimizer.step()
            #save_sample_npy(g[0],landmark_GT[0], shape[0][0],shape[1][0],image[0],global_step, checkpoint_dir)
            if global_step % checkpoint_interval == 0:
                save_sample_images(g[1],gt[1],image[1],landmark_prior[1], landmark_GT[1],Masked[1],step, checkpoint_dir)
            #save_sample_images(g[0],gt[0],image[0], step, checkpoint_dir)
            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            

            
            
            if global_step == 1 or global_step % hparams.eval_interval == 0  :
                
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                    
                    if average_sync_loss<L1:
                        L1=average_sync_loss
                        is_best=True
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch,is_best)
                is_best=False
                    
                        
            prog_bar.set_description('L1: {}'.format(running_l1_loss / (step + 1)))

        global_epoch += 1
        

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    recon_losses = []
    step = 0
    i=0
    while 1:
        start = 0 # start
        for x, indiv_mels, mel, gt,landmark_prior,landmark_GT,shape,image,Masked in test_data_loader:
            #print("pass the model. dataloader, time :", time.time() - start)
            
            i+=1
            #print("step : {}".format(i))
            step += 1

            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)
            #print("start")

            g = model(indiv_mels, x)
            #print("pass the model. g, time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
            Masked=Masked.to(device)
    
            l1loss = recon_loss(g, gt)
            
            
            recon_losses.append(l1loss.item())
            #print("pass the model, time :", time.time() - start)
            #save_sample_images(g[0],gt[0],image[0],landmark_prior[0], landmark_GT[0],step, checkpoint_dir)
            #print("end")
            #start = time.time() 
            if step > eval_steps: 
                
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)

                print('EVALUATION L1: {}'.format(averaged_recon_loss))

                return averaged_recon_loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch,best):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    if best:
        shutil.copyfile(checkpoint_path, os.path.join(checkpoint_dir,'model_best.pth'))
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    random_seed = 1006
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    #train_dataset = Dataset('train') 
    #test_dataset = Dataset('val')

    train_dataset = lrs('train')
    test_dataset = lrs('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=False,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=16)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Model
    model = L2L().to(device)
    #model = nn.DataParallel(model)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        
    

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
