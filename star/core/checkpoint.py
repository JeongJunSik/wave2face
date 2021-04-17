"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        torch.save(outdict, fname)

    def load(self, step, fname=None):
        if fname is None:
            fname = self.fname_template.format(step)
        else:
            fname = fname
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        # for name, module in self.module_dict.items():
        #     print("@@ names: ", name)
        # for name, module in module_dict.items():
        #     print("## names: ", name)
        for name, module in self.module_dict.items():
            print("@@@@@@@@@@@@@@@@ name: ", name)
            if name == 'generator' or name == 'style_encoder':
                self.load_model( module_dict[name], module  )
            elif name == 'discriminator' or name == 'fan':
                continue
            else:
                continue

    def load_model(self, pretrained_dict, model):
        model_dict = model.state_dict()
        model_dict_keys = model_dict.keys()
        #model_dict = pretrained_dict
        
        # 1. filter out unnecessary keys
        skip_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict_keys:
                pretrained_dict_feat_shape = v.shape
                model_dict_feat_shape = model_dict[k].shape
                if pretrained_dict_feat_shape != model_dict_feat_shape:
                    skip_keys.append(k)

        filtered_dict = {(k): v for k, v in pretrained_dict.items() if (k) in model_dict and not any([sk in k for sk in skip_keys])}
        skipped_keys = [ k for k in pretrained_dict if (k) not in filtered_dict]
        


        # 2. overwrite entries in the existing state dict
        model_dict.update(filtered_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)

        print("######## loaded_keys ##########")
        print("@@@@@@@@@@@ num_filtered_dict: ", len(filtered_dict))
        print(filtered_dict.keys())
        print("----------------------------------------------------------------")
        print("######## skipped_keys ##########")
        print("@@@@@@@@@@@ num_skipped_dict: ", len(skipped_keys))
        print(skipped_keys)
        print("-..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..-")