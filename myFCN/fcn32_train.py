import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import torch
import yaml

import myfcn

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        myfcn.models.FCN32,
        myfcn.models.FCN16,
        myfcn.models.FCN8,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-10, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    args = parser.parse_args()
    args.model = 'FCN32s'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))


    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


    #1.datasets
    train_loader = torch.utils.data.DataLoader(
        myfcn.dataset.SBDClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        myfcn.dataset.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)
    #2.models
    start_epoch = 0
    start_iteration = 0
    model = myfcn.models.FCN32(n_class=21)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = myfcn.models.VGG16(pretrained=False)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()
    
    #3.optimizer
    optim = torch.optim.SGD([
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True), 
                'lr': args.lr * 2, 'weigth_decay': 0},
            ], 
            lr=args.lr, 
            momentum=args.momentum)
    if args.resume:
        optim.load_state_dict(['optim_state_dict'])

    trainer = myfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()