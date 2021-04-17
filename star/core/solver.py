"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.dataloader import InputFetcher
import core.utils as utils
import sys
from torch.utils.tensorboard import SummaryWriter
import cv2

#from metrics.eval import calculate_metrics
import random
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]  = "1"
#
random_seed = 1
os.environ['PYTHONHASHSEED'] = str(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.summary_writer = SummaryWriter('./runs/experiment_1')

        #self.nets, self.nets_ema = build_model(args)
        self.nets = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # for name, module in self.nets_ema.items():
        #     setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr= args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{0}_nets.ckpt'), **self.nets),
                #CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{0}_optims.ckpt'), **self.optims)]

            #""" load the pretrained checkpoint """
            #self._load_checkpoint(step="", fname='./checkpoints/git_nets_ema.ckpt')

        if args.mode == 'eval':
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{0}_nets.ckpt'), **self.nets),
                #CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                #CheckpointIO(ospj(args.checkpoint_dir, '{0}_optims.ckpt'), **self.optims)]
            ]
        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    
    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step, fname=None):
        for ckptio in self.ckptios:
            ckptio.load(step, fname)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        #nets_ema = self.nets_ema
        optims = self.optims

        # resume training if necessary
        #if args.resume_iter > 0:
        #    self._load_checkpoint( str(e) + '_' + str(args.resume_iter) )

        """ define the fetcher for dataloading """
        fetcher_tr = InputFetcher(loaders.train, 'train')
        fetcher_val = InputFetcher(loaders.val, 'val')

        print('Start training...')
        start_time = time.time()

        for e in range(args.epoch):
            for i in range (args.resume_iter, len(fetcher_tr) ):#args.total_iters ):

                """ get input from training and validation from fetcher """
                inputs = next(fetcher_tr)
                inputs_val = next(fetcher_val)
                gt_land, gt, gt_mask, prior = inputs.gt_land, inputs.gt, inputs.gt_mask, inputs.prior
                gt_land = gt_land.detach()
                gt = gt.detach()
                gt_mask = gt_mask.detach()
                prior = prior.detach()
                #gt.shape ...: (batch, sync_t, c, h, w)
                #prior.shape: (batch, sync_t, c*2, h, w)

                gt_land = gt_land.flatten(0,1) # (batch*sync_t, c*3, h, w)
                gt = gt.flatten(0,1)
                gt_mask = gt_mask.flatten(0,1)
                prior = prior.flatten(0,1) # (batch*sync_t, c*2, h, w)
                #utils.save_image(  torch.cat( (gt[:,:3], gt[:,3:]), dim=0 ) , './sample_gt.jpg')
                # utils.save_image(prior.view(5,2,3,256,256).view(10,3,256,256), './sample_prior.jpg')

                """ train the discriminator """
                # d_losses.key(): real, fake, reg
                d_loss, d_losses = compute_d_loss( nets, args, gt_land, gt, gt_mask, prior ) 
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

                """ train the generator """
                # g_losses.key(): adv, recon
                g_loss, g_losses = compute_g_loss(nets, args, gt_land, gt, gt_mask, prior )
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.style_encoder.step() ############# style encoder update

                # print out log info
                if (i+1) % args.print_every == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                    log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, len(fetcher_tr))
                    all_losses = dict()
                    for loss, prefix in zip([d_losses, g_losses],
                                            ['D', 'G']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                    log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                    print(log)

                    for k, v in all_losses.items():
                        self.summary_writer.add_scalar( k , v, e*i+i) # add


                # save model checkpoints
                if (i+1) % args.save_every == 0:
                    self._save_checkpoint(step= str(e) + '_' + str(i+1))

                # generate images for debugging
                with torch.no_grad():
                    if (i+1) % args.sample_every == 0:
                        os.makedirs(args.sample_dir, exist_ok=True)
                        utils.debug_image(nets, args, inputs=inputs_val, step= str(e) + '_' + str(i+1) )
                    # compute FID and LPIPS if necessary
                    # if (i+1) % args.eval_every == 0:
                    #     calculate_metrics(nets, args, i+1, mode='latent')
                    #     calculate_metrics(nets, args, i+1, mode='reference')

    @torch.no_grad()
    def eval(self, loaders):
        args = self.args
        nets = self.nets
        self._load_checkpoint(step="", fname=args.chkpt_path)

        """ define the fetcher for dataloading """
        fetcher_eval = InputFetcher(loaders.eval, 'eval')
        gen_frame_list = []
        gt_frame_list = []
        for i in range (args.resume_iter, len(fetcher_eval) ):#args.total_iters ):
                """ get input from eval from fetcher """
                inputs = next(fetcher_eval)
                gt_land, gt, gt_mask, _ = inputs.gt_land, inputs.gt, inputs.gt_mask, inputs.prior
                #gt.shape ...: (batch, sync_t, c, h, w)
                #prior.shape: (batch, sync_t, c*2, h, w)

                gt_land = gt_land.flatten(0,1) # (batch*sync_t, c*3, h, w)
                gt = gt.flatten(0,1)
                gt_mask = gt_mask.flatten(0,1)

                s_gt = nets.style_encoder( gt ) #(batch*sync_t, 512)
                gen_frames = nets.generator( torch.cat( (gt_land, gt_mask), dim=1 ) , s_gt, gt)

                gen_frame_list.append(gen_frames) # (5,3,128,128)
                gt_frame_list.append(gt)

        result = torch.cat(gen_frame_list, dim=0)
        result_gt = torch.cat(gt_frame_list, dim=0)
        utils.save_image(result, filename="./result_fake.jpg")
        utils.save_image(result_gt, filename="./result_gt.jpg")

        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        out = cv2.VideoWriter('result.mp4', fourcc, 25.0, (128,128))
        for gen_frames in gen_frame_list:
            for gen_frame in gen_frames:
                # gen_frame: [3,128,128]
                gen_frame = utils.denormalize(gen_frame)
                #print("@@@  gen_frame.permute(1,2,0).detach().cpu().numpy().shape: ",  gen_frame.permute(1,2,0).detach().cpu().numpy().shape)
                out.write(  np.uint8(gen_frame.permute(1,2,0).detach().cpu().numpy()*255.0)[::,::,::-1]  )
        out.release()

        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        out = cv2.VideoWriter('result_gt.mp4', fourcc, 25.0, (128,128))
        for gt_frames in gt_frame_list:
            for gt_frame in gt_frames:
                # gen_frame: [3,128,128]
                gt_frame = utils.denormalize(gt_frame)
                #print("@@@  gen_frame.permute(1,2,0).detach().cpu().numpy().shape: ",  gen_frame.permute(1,2,0).detach().cpu().numpy().shape)
                out.write(  np.uint8(gt_frame.permute(1,2,0).detach().cpu().numpy()*255.0)[::,::,::-1]  )
        out.release()

        return 0

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, gt_land, gt, gt_mask, prior):
    gt_land = gt_land.detach()
    gt = gt.detach()
    gt_mask = gt_mask.detach()
    prior = prior.detach()
    # with real images
    """
    gt.shape:    (batch*sync_t, c*2, h, w)
    prior.shape: (batch*sync_t, c*2, h, w)
    """
    gt.requires_grad_()
    #print("@@@@@ compute_d_loss @@@@@@")
    out = nets.discriminator( torch.cat((gt_land, gt), dim=1) ) # (batch,sync_t, 1)
    loss_real = adv_loss(out, 1)
    #print("@@@ compute_d_loss: loss_real.shape: ", loss_real.shape)
    loss_reg = r1_reg(out, gt ) # 이 부분 질문하기!  [3:6]이 되어야 할 거 같음. 즉 진짜 이미지만 들어가게, 그런데 전체가 다 들어가야함  retain graph error occured !!!!
    #print("@@@ compute_d_loss: loss_reg.shape: ", loss_reg.shape)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder( prior ) #(batch*sync_t, 512)
        gt_fake = nets.generator( torch.cat( (gt_land, gt_mask), dim=1 ) , s_trg, gt)

    out = nets.discriminator( torch.cat( (gt_land, gt_fake), dim=1 ) ) ## To Do. concat the gt landmark.
    loss_fake = adv_loss(out, 0)

    loss = loss_real + args.lambda_reg * loss_reg + loss_fake 
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

def compute_g_loss(nets, args, gt_land, gt, gt_mask, prior):
    """  generate fake image and compute adversarial loss """ 
    gt_land = gt_land.detach()
    gt = gt.detach()
    gt_mask = gt_mask.detach()
    prior = prior.detach()

    s_trg = nets.style_encoder( prior )
    gt_fake = nets.generator( torch.cat( (gt_land, gt_mask), dim=1 ), s_trg, gt )
    out = nets.discriminator( torch.cat( (gt_land, gt_fake), dim=1 ) )  # To Do. concat the gt landmark.
    loss_adv = adv_loss(out, 1)

    """ l1 recon loss """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zero_tensor = torch.cuda.FloatTensor( [-1.] ).to(device)
    gt_reverse_masked = torch.where(  gt==gt_mask, zero_tensor , gt)
    fake_reverse_masked = torch.where( gt==gt_mask, zero_tensor, gt_fake )

    loss_recon = F.l1_loss( gt_reverse_masked, fake_reverse_masked)#/gt.shape[0] 
    loss = loss_adv + loss_recon*10

    return loss, Munch(adv=loss_adv.item(),
                       recon=loss_recon.item(),
                      )


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    """
    logits: (batch*sync_t, 1)
    targets: (batch*sync_t, 1)
    """
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    #print("@@ adv_loss: ", loss)
   
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True, allow_unused = True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    #print("@@ reg_loss: ", reg)
    return reg