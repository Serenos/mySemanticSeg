import os.path as osp
import fcn

import torch
import torchvision

def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file = _get_vgg16_pretrained_model()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model



def _get_vgg16_pretrained_model():
    # return fcn.data.cached_download(
    #     url='',
    #     path=osp.expanduser('~/data/models/pytorch/vgg16-397923af.pth'),
    #     md5='',
    # 
    return osp.expanduser('~/data/models/pytorch/vgg16-397923af.pth')

