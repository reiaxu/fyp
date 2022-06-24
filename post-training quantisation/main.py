from models.mobilenetv2 import mobilenetv2, InvertedResidual
from models.resnet import resnet18
from models.vgg import vgg11
from models.alexnet import alexnet
from quant_ops import *

import random
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import argparse

def replace_quant_ops(args, model):
    prev_module = None
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_op = QuantConv(args, child)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, torch.nn.Linear):
            new_op = QuantLinear(args, child)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, (torch.nn.ReLU, torch.nn.ReLU6)):
            # prev_module.activation_function = child
            prev_module.activation_function = torch.nn.ReLU()
            setattr(model, child_name, PassThroughOp())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, PassThroughOp())
        else:
            replace_quant_ops(args, child)

def get_input_sequences(model):
    layer_bn_pairs = []
    def hook(name):
        def func(m, i, o):
            if m in (torch.nn.Conv2d, torch.nn.Linear):
                if not layer_bn_pairs:
                    layer_bn_pairs.append((m, name))
                else:
                    if layer_bn_pairs[-1][0] in (torch.nn.Conv2d, torch.nn.Linear):
                        layer_bn_pairs.pop()
            else:
                layer_bn_pairs.append((m, name))
        return func

    handlers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn([1,3,32,32]).cuda()
    model(dummy)
    for handle in handlers:
        handle.remove()
    return layer_bn_pairs

def register_bn_params_to_prev_layers(model, layer_bn_pairs):
    idx = 0
    while idx + 1 < len(layer_bn_pairs):
        conv, bn = layer_bn_pairs[idx], layer_bn_pairs[idx + 1]
        conv, conv_name = conv
        bn, bn_name = bn
        bn_state_dict = bn.state_dict()
        conv.register_buffer('eps', torch.tensor(bn.eps))
        conv.register_buffer('gamma', bn_state_dict['weight'].detach())
        conv.register_buffer('beta', bn_state_dict['bias'].detach())
        conv.register_buffer('mu', bn_state_dict['running_mean'].detach())
        conv.register_buffer('var', bn_state_dict['running_var'].detach())
        idx += 2

def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(data_loader, batch_size=128):
    def eval_func(model, arguments):
        top1_acc = 0.0
        total_num = 0
        idx = 0
        iterations , use_cuda = arguments[0], arguments[1]
        print(iterations)
        if use_cuda:
            model.cuda()
        for sample, label in tqdm(data_loader):
            total_num += sample.size()[0]
            if use_cuda:
                sample = sample.cuda()
                label = label.cuda()
            logits = model(sample)
            pred = torch.argmax(logits, dim = 1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            top1_acc += correct
            idx += 1
            if idx > iterations:
                break
        avg_acc = top1_acc * 100. / total_num
        print("Top 1 ACC : {:0.2f}".format(avg_acc))
        return avg_acc
    return eval_func


def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def arguments():
    parser = argparse.ArgumentParser(description='Cross Layer Equalization in MV2')

    parser.add_argument('--images-dir',                 help='CIFAR10 eval image', default='./data/cifar-10-python.tar.gz/', type=str)
    parser.add_argument('--seed',                       help='Seed number for reproducibility', type=int, default=0)
    parser.add_argument('--model',                      help='model name', default='resnet18', type=str,
                                                        choices=['alexnet', 'vgg11', 'resnet18', 'mobilenetv2'])
    parser.add_argument('--quant-scheme',               help='Quantization scheme', default='mse', type=str, choices=['mse', 'minmax'])
    parser.add_argument('--bitwidth',                   help='bitwidth', type=int, default=8)
    parser.add_argument('--batch-size',                 help='Data batch size for a model', type = int, default=128)
    parser.add_argument('--num-workers',                help='Number of workers to run data loader in parallel', type = int, default=2)

    args = parser.parse_args()
    return args

def load_model(args, pretrained = True):
    if args.model == 'resnet18':
        model = resnet18(pretrained)
    elif args.model == 'vgg11':
        model = vgg11(pretrained)
    if args.model == 'mobilenetv2':
        model = mobilenetv2(pretrained)
    if args.model == 'alexnet':
        model = alexnet(pretrained)
    model.eval()
    return model

def get_loaders(args):
    image_size = 32
    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : args.num_workers}
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2471, 0.2435, 0.2616])
    val_transforms = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    val_data = CIFAR10(root='content/drive/MyDrive/Quantizations/data/', download=True, train=False, transform=val_transforms)
    train_data = CIFAR10(root='content/drive/MyDrive/Quantizations/data/', download=True, transform=train_transforms)
    val_dataloader = DataLoader(val_data, args.batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
    train_dataloader = DataLoader(train_data, args.batch_size, shuffle = True, pin_memory = True, **data_loader_kwargs)
    return val_dataloader,train_dataloader

def get_conv_layers(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantConv):
            conv_layers.append(module)
    return conv_layers

def set_quant_mode(quantized):
    def set_precision_mode(module):
        if isinstance(module, (Quantizers)):
            module.set_quantize(quantized)
            module.estimate_range(flag = False)
    return set_precision_mode


def run_calibration(calibration):
    def estimate_range(module):
        if isinstance(module, Quantizers):
            module.estimate_range(flag = calibration)
    return estimate_range


def main():
    args = arguments()
    seed(args)
    model = load_model(args, pretrained = True)

    val_dataloader = get_loaders(args)[0]
    train_dataloader = get_loaders(args)[1]

    calibrate_func = model_eval(train_dataloader, batch_size=args.batch_size)
    eval_func = model_eval(val_dataloader, batch_size=args.batch_size)

    model.cuda()
    
    layer_bn_pairs = get_input_sequences(model)
    if args.model == 'mobilenetv2' or args.model == 'resnet18':
        register_bn_params_to_prev_layers(model, layer_bn_pairs)

    def bn_fold(module):
        if isinstance(module, (QuantConv)):
            module.batchnorm_folding()

    replace_quant_ops(args, model)
    model.apply(bn_fold)

    model.apply(run_calibration(calibration = True))
    calibrate_func(model, (args.batch_size, True))

    model.apply(set_quant_mode(quantized = True))

    eval_func(model, (10000, True))

if __name__ == '__main__':
    main()
