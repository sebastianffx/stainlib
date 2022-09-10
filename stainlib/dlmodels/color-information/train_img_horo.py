import argparse
import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc
import sys
import pdb
from glob import glob
from sklearn.utils import shuffle
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image,ImageStat
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
import pyvips
import random
import torch.utils.data.distributed
import horovod.torch as hvd
import cv2
import torch.multiprocessing as mp
import pprint
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets
from torchsummary import summary
from data_utils import make_dataset
from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
from lib.GMM import GMM_model as gmm
import lib.image_transforms as imgtf
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts
import difflib

# Arguments
parser = argparse.ArgumentParser(description='Residual Flow Model Color Information', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## GMM ##
parser.add_argument('--nclusters', type=int, default=4,help='The amount of tissue classes trained upon')

## DATA ##
parser.add_argument('--slide_path', type=str, help='Folder of where the training data whole slide images are located', default=None)
parser.add_argument('--label_path', type=str, help='Folder of where the training data whole slide images masks are located', default=None)
parser.add_argument('--valid_slide_path', type=str, help='Folder of where the validation data whole slide images are located', default=None)
parser.add_argument('--valid_label_path', type=str, help='Folder of where the validation data whole slide images masks are located', default=None)
parser.add_argument('--test_path', type=str, help='Folder of where the testing data whole slide images are located', default=None)
parser.add_argument('--slide_format', type=str, help='In which format the whole slide images are saved.', default='tif')
parser.add_argument('--label_format', type=str, help='In which format the masks are saved.', default='xml')
parser.add_argument('--bb_downsample', type=int, help='OpenSlide(<slide_path>).level_dimensions[bb_downsample] is used for the contour construction as downsampling level of whole slide image', default=7)
parser.add_argument('--log_dir', type=str, help='Path of savepath of downsampled image with processed rectangles on it.', default='.')
parser.add_argument('--verbose', type=str, help='Verbosity of the data sampler', default='info',choices=['info','debug'])
parser.add_argument('--batch_tumor_ratio', type=int, help='round(batch_tumor_ratio*batch_size) images will have tumor in the mini batch, set to 0 if no tumor', default=0)
parser.add_argument('--val_split', type=float, default=0.15)
parser.add_argument('--debug', action='store_true', help='If running in debug mode')

## MEMORY / TIME ##
parser.add_argument('--steps_per_epoch', type=int, help='The hard - coded amount of iterations in one epoch.', default=1000)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=1)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=5)
parser.add_argument('--save_every', help='Save model every so epochs', type=int, default=1)
parser.add_argument('--fp16_allreduce', action='store_true', help='If all reduce in fp16')
parser.add_argument('--img_size', type=int, help='The Field of View (patch size) to use', default=32)

## EVALUATION ##
parser.add_argument('--evaluate',action='store_true', help='If running evaluation')
parser.add_argument('--resume', type=str, help='File of checkpoint (.pth file) if resuming from checkpoint',default=None)
parser.add_argument('--save_conv', action='store_true',help='Save converted images.')
parser.add_argument('--deploy_samples', type=int,help='How many samples of images should be used as templates in evaluation.', default=2)
parser.add_argument('--begin-epoch', type=int, default=0)
parser.add_argument('--gather_metrics', action='store_true', help='If gathering NMI metrics')

    

## INVERTIBLE RESIDUAL NETWORK ##
parser.add_argument('--nbits', type=int, default=10)  # Only used for celebahq.
parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False,help='Learn Lipschitz norms, see paper')

parser.add_argument('--n-power-series', type=int, default=None, help='Amount of power series evaluated, see paper')
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False,help='Factorize dimensions, see paper')
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2,help='Exact terms computed in series estimation, see paper')
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True,help='Neumann gradients, see paper')
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True,help='Memory efficient backprop, see paper')

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=128)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval, choices=[True, False], default=False)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-idim', type=int, default=8)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=False)
parser.add_argument('--cdim', type=int, default=128)

## OPTIMIZER ##
parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batch_size', help='Minibatch size', type=int, default=64)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)

parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, help='Use exponential moving averages of parameters at validation.', choices=[True, False], default=False)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid','gmm'], default='gmm')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')




args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)


# Horovod: initialize library.
hvd.init()
print(f"hvd.size {hvd.size()} hvd.rank {hvd.rank()} hvd.local_rank {hvd.local_rank()}")


# logger
try:
    if hvd.rank() == 0:
        utils.makedirs(args.log_dir)
        logger = utils.get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
except:
    utils.makedirs(args.log_dir)
    logger = utils.get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

if hvd.rank() == 0:
    logger.info(args)
    
if device.type == 'cuda':
    if hvd.rank() == 0:
        logger.info(f'Found {hvd.size()} CUDA devices.')
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    
    if hvd.rank() == 0:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}


    
def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rescale(tensor):
    """
    Parameters
    ----------
    tensor : Pytorch tensor
        Tensor to be rescaled to [0,1] interval.

    Returns
    -------
    Rescaled tensor.

    """
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor

def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x
    

im_dim = args.nclusters
n_classes = args.nclusters
init_layer = layers.LogitTransform(0.05)


train_dataset = make_dataset(args,mode='train')
test_dataset  = make_dataset(args,mode='validation')
# # Horovod: use DistributedSampler to partition the training data.
# train_sampler = torch.utils.data.distributed.DistributedSampler(
#     train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
# # Horovod: use DistributedSampler to partition the test data.
# test_sampler = torch.utils.data.distributed.DistributedSampler(
#     test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

if args.task in ['classification', 'hybrid','gmm']:
    try:
        n_classes
    except NameError:
        raise ValueError('Cannot perform classification without n_classes')
else:
    n_classes = 1

if hvd.rank() == 0:
    logger.info('Dataset loaded.')
    logger.info('Creating model.')

# input_size = (args.batch_size, 1, (im_dim + args.padding) //  2 * args.img_size, (im_dim + args.padding) //  2 * args.img_size)
input_size = (args.batch_size, (im_dim + args.padding), args.img_size, args.img_size)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

# Model
model = ResidualFlow(
    input_size,
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=args.idim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    classification=args.task in ['classification', 'hybrid'],
    classification_hdim=args.cdim,
    n_classes=n_classes,
    block_type=args.block,
)



# Custom
gmm = gmm(input_size,args,num_clusters=args.nclusters)
gmm.to(device)
model.to(device)
ema = utils.ExponentialMovingAverage(model)

def parallelize(model):
    return torch.nn.DataParallel(model)

if hvd.rank() == 0:
    logger.info(model)
    logger.info('EMA: {}'.format(ema))

# Optimization
def tensor_in(t, a):
    for a_ in a:
        if t is a_:
            return True
    return False


scheduler = None
params = [par for par in model.parameters()] + [par for par in gmm.parameters()]

# params = [par for par in gmm.parameters()]
if args.optimizer == 'adam':
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    if args.scheduler: scheduler = CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2, last_epoch=args.begin_epoch - 1)
elif args.optimizer == 'adamax':
    optimizer = optim.Adamax(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(params, lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1
        )
else:
    raise ValueError('Unknown optimizer {}'.format(args.optimizer))




# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none


optimizer = hvd.DistributedOptimizer(optimizer,
                                      backward_passes_per_step=args.update_freq,
                                      named_parameters=model.named_parameters(),
                                      compression=compression,
                                      op=hvd.Adasum)

# Horovod: broadcast parameters & optimizer state.

best_test_bpd = math.inf


if (args.resume is not None):
    
    if hvd.rank() == 0: logger.info('Resuming model from {}'.format(args.resume))
    
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    
    checkpt = torch.load(args.resume,map_location='cpu')
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    
    try:
        model.load_state_dict(state, strict=True)
    except ValueError("Model mismatch, check args.nclusters and args.nblocks"):
        sys.exit(1)
    
    # ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state



hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
hvd.join()



if hvd.rank()== 0:
    logger.info(optimizer)

fixed_z = standard_normal_sample([min(32, args.batch_size),
                                  (im_dim + args.padding) * args.img_size * args.img_size]).to(device)



def compute_loss(x, model,gmm, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)


    nvals = 2**args.nbits
    # print(f"max {torch.max(x)} {torch.min(x)}")
    
        
    if args.task == 'gmm' :
        # Get the D from the HSD (Actually the HSD consists of D,cx,cy see lib.image_transforms)
        D = x[:,0,...].unsqueeze(0).clone()
        # You need to rescale to process by model, however use HSD for other
        D = rescale(D) # rescaling to [0,1]
        if args.squeeze_first:
            D = squeeze_layer(D)
        
        # Is for uniform or Gaussian padding, default is with zeros, and args.padding = 0 is default
        D, logpu = add_padding(D, nvals)
        
        # Repeat D because we need args.nclusters for the GMM input shape (thus model output shape)
        D = D.repeat(1, args.nclusters, 1, 1)
        z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
        z, delta_logp = z_logp
        # log p(z)
        # logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        logpz, params = gmm(z.view(-1,args.nclusters,args.img_size,args.img_size), x)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (args.img_size * args.img_size * (im_dim + args.padding)) - logpu
        bits_per_dim = -torch.mean(logpx) / (args.img_size * args.img_size * im_dim) / np.log(2)

        logpz = torch.mean(logpz).detach()
        delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp, params


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.iResBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


batch_time = utils.RunningAverageMeter(0.97)
bpd_meter = utils.RunningAverageMeter(0.97)
ll_meter = utils.RunningAverageMeter(0.97)
logpz_meter = utils.RunningAverageMeter(0.97)
deltalogp_meter = utils.RunningAverageMeter(0.97)
firmom_meter = utils.RunningAverageMeter(0.97)
secmom_meter = utils.RunningAverageMeter(0.97)
gnorm_meter = utils.RunningAverageMeter(0.97)
ce_meter = utils.RunningAverageMeter(0.97)




def train(epoch,model,gmm):

    model.train()
    gmm.train()

    print("Starting Training...")
    end = time.time()
    step = 0
    while step < args.steps_per_epoch:
        for idx, (x, y) in enumerate(train_loader):
    
            x = x.to(device)
            global_itr = epoch*args.steps_per_epoch + step
            update_lr(optimizer, global_itr)
            
            if hvd.rank() == 0: print(f'Step {step}')
            # Training procedure:
            # for each sample x:
            #   compute z = f(x)
            #   maximize log p(x) = log p(z) - log |det df/dx|
            #   see https://arxiv.org/abs/1906.02735
            beta = 1
            bpd, logits, logpz, neg_delta_logp, params = compute_loss(x, model,gmm, beta=beta)
    
            firmom, secmom = estimator_moments(model)
    
            bpd_meter.update(bpd.item())
            logpz_meter.update(logpz.item())
            deltalogp_meter.update(neg_delta_logp.item())
            firmom_meter.update(firmom)
            secmom_meter.update(secmom)
            
                    
            loss = bpd
            loss.backward()
            
            if global_itr % args.update_freq == args.update_freq - 1:
                
                if args.update_freq > 1:
                    with torch.no_grad():
                        for p in model.parameters():
                            

                            if p.grad is not None:
                                p.grad /= args.update_freq
 
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
                if args.learn_p: compute_p_grads(model)
    
                
                optimizer.step()
                optimizer.zero_grad()
                update_lipschitz(model)
                ema.apply()
    
                gnorm_meter.update(grad_norm)
            

                
            # measure elapsed time
            steps_per_epoch = args.steps_per_epoch
            batch_time.update(time.time() - end)
            end = time.time()
            if global_itr % args.print_freq == 0:
                s = (
                    '\n\nEpoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                    'GradNorm {gnorm_meter.avg:.2f}'.format(
                        epoch, step, steps_per_epoch, batch_time=batch_time, gnorm_meter=gnorm_meter
                    )
                )
    
                if args.task in ['density', 'hybrid','gmm']:
                    s += (
                        f' | Bits/dim {bpd_meter.val}({bpd_meter.avg}) | '
                        # f' | params {[p.clone() for p in gmm.parameters().grad]}) | '
                        f'Logpz {logpz_meter.avg} | '
                        f'-DeltaLogp {deltalogp_meter.avg} | '
                        f'EstMoment ({firmom_meter.avg},{secmom_meter.avg})\n\n'
                        )
                
                if hvd.rank() == 0:
                    logger.info(s)
            if global_itr % args.vis_freq == 0 and idx > 0:
                visualize(epoch, model,gmm, idx, x, global_itr)
        
            del x
            torch.cuda.empty_cache()
            gc.collect()
            step += 1
    return

def savegamma(gamma,global_itr,pred=0):
    ClsLbl = np.argmax(gamma, axis=-1)
    ClsLbl = ClsLbl.astype('float32')
    ColorTable = [[255,0,0],[0,255,0],[0,0,255],[255,255,0], [0,255,255], [255,0,255]]
    colors = np.array(ColorTable, dtype='float32')
    Msk = np.tile(np.expand_dims(ClsLbl, axis=-1),(1,1,1,3))
    for k in range(0, args.nclusters):
        #                                       1 x 256 x 256 x 1                           1 x 3 
        ClrTmpl = np.einsum('anmd,df->anmf', np.expand_dims(np.ones_like(ClsLbl), axis=3), np.reshape(colors[k,...],[1,3]))
        # ClrTmpl = 1 x 256 x 256 x 3
        Msk = np.where(np.equal(Msk,k), ClrTmpl, Msk)
    
    im_gamma = Msk[0].astype('uint8')
    im_gamma = Image.fromarray(im_gamma)
    if pred == 0:
        im_gamma.save(os.path.join(args.log_dir,'imgs',f'im_gamma_{global_itr}.png'))
    elif pred == 1:
        im_gamma.save(os.path.join(args.log_dir,'imgs',f'im_pi_{global_itr}.png'))
    elif pred == 2:
        im_gamma.save(os.path.join(args.log_dir,'imgs',f'im_recon_{global_itr}.png'))
    elif pred == 3:
        im_gamma.save(os.path.join(args.log_dir,'imgs',f'im_fake_{global_itr}.png'))
        
    im_gamma.close()
    return
    
def validate(epoch, model,gmm, ema=None):
    """
    - Deploys the color normalization on test image dataset for args.deploy_samples
    - Evaluates NMI / CV / SD
    """
    
        
    if hvd.rank() == 0: utils.makedirs(os.path.join(args.log_dir, 'imgs')), print("Starting Deployment")


    if ema is not None:
        ema.swap()

    update_lipschitz(model)

    model.eval()
    gmm.eval()

    mu_tmpl = 0
    std_tmpl = 0
    N = 0


    if hvd.rank() == 0: print(f"Deploying on {args.deploy_samples} templates...")
    for idx, (x, y) in enumerate(train_loader):
        
        if idx == args.deploy_samples: break
        print(f"Worker {hvd.rank()}: template {idx + 1} / {args.deploy_samples}")
        t1 = time.time()
        
        x = x.to(device)
       
        ### TEMPLATES ###
        D = x[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.img_size,args.img_size), x)
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.img_size,args.img_size), x)
        

        
        mu, std, gamma =  params
        
        mu      = mu.cpu().numpy()
        std     = std.cpu().numpy()
        gamma   = gamma.cpu().numpy() 
    
        
        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)

        N = N+1
        mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
        std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
        
        if idx % 10 == 0 and hvd.rank() == 0: print(f"Batch {idx} at { hvd.size()*args.batch_size / (time.time() - t1) } imgs / sec")
        if args.save_conv:
            # save images for transformation
            for ct, (img,path) in enumerate(zip(x,train_loader.dataset.cur_wsi_path[0])):
                im_tmpl = img.cpu().numpy()
                im_tmpl = np.swapaxes(im_tmpl,0,1)  
                im_tmpl = np.swapaxes(im_tmpl,1,-1)
                im_tmpl = imgtf.HSD2RGB_Numpy(im_tmpl)
                im_tmpl = (im_tmpl).astype('uint8')
                im_tmpl = Image.fromarray(im_tmpl)
                im_tmpl.save(os.path.join(args.log_dir,'imgs',f'im_tmpl-{path.split("/")[-1]}-eval.png'))
                im_tmpl.close()
        
            
    if hvd.rank() == 0: print("Allreduce mu_tmpl / std_tmpl ...")
    mu_tmpl   = hvd.allreduce(torch.tensor(mu_tmpl).contiguous())
    std_tmpl  = hvd.allreduce(torch.tensor(std_tmpl).contiguous())
    if hvd.rank() == 0: print("Broadcast mu_tmpl / std_tmpl ...")
    hvd.broadcast(mu_tmpl,0)
    hvd.broadcast(std_tmpl,0)
    hvd.join()
    
    if hvd.rank() == 0:
        print("Estimated Mu for template(s):")
        print(mu_tmpl)
          
        print("Estimated Sigma for template(s):")
        print(std_tmpl)
        
        del x
        torch.cuda.empty_cache()
        gc.collect()

         
    metrics = dict()
    for tc in range(1,args.nclusters+1):
        metrics[f'mean_{tc}'] = []
        metrics[f'median_{tc}']=[]
        metrics[f'perc_95_{tc}']=[]
        metrics[f'nmi_{tc}']=[]
        metrics[f'sd_{tc}']=[]
        metrics[f'cv_{tc}']=[]
    
    if hvd.rank() == 0: print(f"Deploying on {args.deploy_samples} test images...")
    for idx, (x_test, y_test) in enumerate(test_loader):
        if idx == args.deploy_samples: break
        print(f"Worker {hvd.rank()}: template {idx + 1} / {args.deploy_samples}")
        t1 = time.time()
        x_test = x_test.to(device)
        

        ### DEPLOY ###
        D = x_test[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
        
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.img_size,args.img_size), x_test)
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.img_size,args.img_size), x_test)

        
        mu, std, pi =  params

            
        mu  = mu.cpu().numpy()
        std = std.cpu().numpy()
        pi  = pi.cpu().numpy()
        
        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)
        
        X_hsd = np.swapaxes(x_test.cpu().numpy(),1,2)
        X_hsd = np.swapaxes(X_hsd,2,3)
        
        X_conv = imgtf.image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, args)

        ClsLbl = np.argmax(np.asarray(pi),axis=-1) + 1
        ClsLbl = ClsLbl.astype('int32')
        mean_rgb = np.mean(X_conv,axis=-1)

        for tc in range(1,args.nclusters+1):
            msk = torch.where(torch.tensor(ClsLbl) == tc , torch.tensor(1),torch.tensor(0))
            msk = [(i,msk.cpu().numpy()) for i, msk in enumerate(msk) if torch.max(msk).cpu().numpy()] # skip metric if no class labels are found
            if not len(list(msk)): continue
            # Take indices from valid msks and get mean_rgb at valid indices, then multiply with msk
            idces = [x[0] for x in msk]
            msk   = np.array([x[1] for x in msk])
            try:
                ma = mean_rgb[idces,...] * msk
                mean    = np.array([np.mean(ma[ma!=0]) for ma in ma])
                median  = np.array([np.median(ma[ma!=0]) for ma in ma])
                perc    = np.array([np.percentile(ma[ma!=0],95) for ma in ma])
                nmi = median / perc
            except:
                mean, median, perc, nmi = np.zeros((1)),np.zeros((1)),np.zeros((1)),np.zeros((1))
            
            metrics['mean_'     +str(tc)].extend(list(mean))
            metrics['median_'   +str(tc)].extend(list(median))
            metrics['perc_95_'  +str(tc)].extend(list(perc))
            metrics['nmi_'      +str(tc)].extend(list(nmi))
            
        
        if args.save_conv:
            for ct, (img, path) in enumerate(zip(x_test,test_loader.dataset.cur_wsi_path[0])):
                im_test = img.cpu().numpy()
                im_test = np.swapaxes(im_test,0,1)
                im_test = np.swapaxes(im_test,1,-1)
                im_test = imgtf.HSD2RGB_Numpy(im_test)
                im_test = (im_test).astype('uint8')
                im_test = Image.fromarray(im_test)
                im_test.save(os.path.join(args.log_dir,'imgs',f'im_test-{path.split("/")[-1]}-eval.png'))
                im_test.close()
            # for ct, img in enumerate(X_conv):
                im_conv = X_conv.reshape(args.img_size,args.img_size,3)
                im_conv = Image.fromarray(im_conv)
                im_conv.save(os.path.join(args.log_dir,'imgs',f'im_conv-{path.split("/")[-1]}-eval.png'))
                im_conv.close()
            
            # savegamma(pi,f"{idx}-eval",pred=1)
        
        
        if idx % 10 == 0 and hvd.rank() == 0: print(f"Batch {idx} at { hvd.size()*args.batch_size / (time.time() - t1) } imgs / sec")

    
    if args.gather_metrics:
        # average sd of nmi across tissue classes
        av_sd = []
        # average cv of nmi across tissue classes
        av_cv = []
        # total nmi across tissue classes
        tot_nmi = np.empty((0,0))
        
        
        for tc in range(1,args.nclusters+1):
            if len(metrics['mean_' + str(tc)]) == 0: continue
            print("Allgather...")
            nmi = hvd.allgather(torch.tensor(np.array(metrics['nmi_' + str(tc)])[...,None]))
            metrics[f'sd_' + str(tc)] = torch.std(nmi).cpu().numpy()
            metrics[f'cv_' + str(tc)] = torch.std(nmi).cpu().numpy() / torch.mean(nmi).cpu().numpy()
            if hvd.rank() == 0:
                print(f'sd_' + str(tc)+':', metrics[f'sd_{tc}'])
                print(f'cv_' + str(tc)+':', metrics[f'cv_{tc}'])
            av_sd.append(metrics[f'sd_' + str(tc)])
            av_cv.append(metrics[f'cv_' + str(tc)])
            tot_nmi = np.append(tot_nmi,nmi)
        
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots()
        ax1.set_title(f'Box Plot Eval {args.log_dir.split("/")[-1]}')
        ax1.boxplot(tot_nmi)
        
        
        if hvd.rank() == 0:
            plt.savefig(f'worker-{hvd.rank()}-{args.log_dir.split("/")[-1]}-boxplot-eval.png')
            print(f"Average sd = {np.array(av_sd).mean()}")
            print(f"Average cv = {np.array(av_cv).mean()}")
            import csv
            file = open(f'worker-{hvd.rank()}-{args.log_dir.split("/")[-1]}-metrics-eval.csv',"w")
            writer = csv.writer(file)
            for key, value in metrics.items():
                writer.writerow([key, value])
            
            
            file.close()
    

    # correct = 0
    # total = 0

    # start = time.time()
    # with torch.no_grad():
    #     for i, (x, y) in enumerate(tqdm(test_loader)):
    #         x = x.to(device)

    #         bpd, logits, _, _ = compute_loss(x, model)
    #         bpd_meter.update(bpd.item(), x.size(0))

    # val_time = time.time() - start

    # if ema is not None:
    #     ema.swap()
    # s = 'Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {bpd_meter.avg:.4f}'.format(epoch, val_time, bpd_meter=bpd_meter)
    # if args.task in ['classification', 'hybrid']:
    #     s += ' | CE {:.4f} | Acc {:.2f}'.format(ce_meter.avg, 100 * correct / total)
    # logger.info(s)
    # return bpd_meter.avg
    print("Finished Evaluation")
    return 1


def visualize(epoch, model, gmm, itr, real_imgs, global_itr):
    model.eval()
    gmm.eval()
    if hvd.rank() == 0: utils.makedirs(os.path.join(args.log_dir, 'imgs')), print("Starting Visualisation...")

    for x_test, y_test in test_loader:
        # x_test = x_test[0,...].unsqueeze(0)
        # y_test = y_test[0,...].unsqueeze(0)
        x_test = x_test.to(device)
        ### TEMPLATES ###
        D = real_imgs[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        x = real_imgs
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
            
            
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.img_size,args.img_size), x)
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.img_size,args.img_size), x)

    
        mu_tmpl, std_tmpl, gamma =  params
        mu_tmpl  = mu_tmpl.cpu().numpy()
        std_tmpl = std_tmpl.cpu().numpy()
        gamma    = gamma.cpu().numpy() 
    
        mu_tmpl  = mu_tmpl[...,np.newaxis]
        std_tmpl = std_tmpl[...,np.newaxis]
        
        mu_tmpl  = np.swapaxes(mu_tmpl,0,1) # (3,4,1) -> (4,3,1)
        mu_tmpl  = np.swapaxes(mu_tmpl,1,2) # (4,3,1) -> (4,1,3)
        std_tmpl = np.swapaxes(std_tmpl,0,1) # (3,4,1) -> (4,3,1)
        std_tmpl = np.swapaxes(std_tmpl,1,2) # (4,3,1) -> (4,1,3)
        
            
        ### DEPLOY ###
        D = x_test[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
                # recon = model(model(D.view(-1, *input_size[1:])), inverse=True).view(-1, *input_size[1:])
                # fake_imgs = model(fixed_z, inverse=True).view(-1, *input_size[1:])

            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.img_size,args.img_size), x_test)
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.img_size,args.img_size), x_test)

        mu, std, pi =  params
        mu  = mu.cpu().numpy()
        std = std.cpu().numpy()
        pi  = pi.cpu().numpy()
        # recon  = np.swapaxes(np.swapaxes(recon.cpu().numpy(),1,2),2,-1)
        # fake_imgs  = np.swapaxes(np.swapaxes(fake_imgs.cpu().numpy(),1,2),2,-1)

        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)
        
    
        X_hsd = np.swapaxes(x_test.cpu().numpy(),1,2)
        X_hsd = np.swapaxes(X_hsd,2,3)
        
        X_hsd        = X_hsd 
        X_conv       = imgtf.image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, args)
        # X_conv_recon = imgtf.image_dist_transform(X_hsd, mu, std, recon, mu_tmpl, std_tmpl, args)
        # save a random image from the batch
        """
        Here images are saved:
            1. Get image at im_no
            2. Transfer to cpu
            3. Watch torch ordering, bring back to RGB
            4. see lib.image_tranforms for HSD2RGB_Numpy
            5. Watch transform and *
            6. save with global_itr
        """
        im_no = random.randint(0,args.batch_size-1) 
        im_tmpl = real_imgs[im_no,...].cpu().numpy()
        im_tmpl = np.swapaxes(im_tmpl,0,1)
        im_tmpl = np.swapaxes(im_tmpl,1,-1)
        im_tmpl = im_tmpl
        im_tmpl = imgtf.HSD2RGB_Numpy(im_tmpl)
        im_tmpl = Image.fromarray(im_tmpl.astype('uint8'))
        im_tmpl.save(os.path.join(args.log_dir,'imgs',f'im_tmpl_{global_itr}.png'))
        
        im_test = x_test[im_no,...].cpu().numpy()
        im_test = np.swapaxes(im_test,0,1)
        im_test = np.swapaxes(im_test,1,-1)
        im_test = im_test
        im_test = imgtf.HSD2RGB_Numpy(im_test)
        im_test = Image.fromarray(im_test.astype('uint8'))
        im_test.save(os.path.join(args.log_dir,'imgs',f'im_test_{global_itr}.png'))
        
        ## This will be the image coming out of invert resnet ##
        # D = x_test[:,0,...].unsqueeze(1).clone()
        # im_D = D[0,0,...].cpu().numpy()
        # im_D = (im_D).astype('uint8')
        # im_D = Image.fromarray(im_D,'L')
        # im_D.save(os.path.join(args.log_dir,'imgs',f'im_beforegmm_{global_itr}.png'))
        if args.batch_size>1:
            im_conv = X_conv[im_no,...].reshape(args.img_size,args.img_size,3)
        else:
            im_conv = X_conv.reshape(args.img_size,args.img_size,3)
        im_conv = Image.fromarray(im_conv)
        im_conv.save(os.path.join(args.log_dir,'imgs',f'im_conv_{global_itr}.png'))
        
        # im_conv_recon = np.squeeze(X_conv_recon)[im_no,...].reshape(args.img_size,args.img_size,3)
        # im_conv_recon = Image.fromarray(im_conv_recon)
        # im_conv_recon.save(os.path.join(args.log_dir,'imgs',f'im_conv_recon_{global_itr}.png'))
        
        # gamma
        savegamma(gamma,global_itr,pred=0)
        
        # pi
        savegamma(pi,global_itr,pred=1)
        # reconstructed images
        # savegamma(recon,global_itr,pred=2)
        
        # reconstructed images
        # savegamma(fake_imgs,global_itr,pred=3)
        
        model.train()
        gmm.train()
        return


def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
            lipschitz_constants.append(m.scale)
    return lipschitz_constants


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def get_ords(model):
    ords = []
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            domain, codomain = m.compute_domain_codomain()
            if torch.is_tensor(domain):
                domain = domain.item()
            if torch.is_tensor(codomain):
                codomain = codomain.item()
            ords.append(domain)
            ords.append(codomain)
    return ords


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'


def main():
    global best_test_bpd

    last_checkpoints = []
    lipschitz_constants = []
    ords = []

    if args.evaluate:
        assert isinstance(args.resume,str),"WARNING: CANNOT START EVALUATION WITHOUT CHECKPOINT DIRECTORY (args.resume -> str)"
        validate(args.begin_epoch - 1, model,gmm)
        sys.exit(0)
        
    for epoch in range(args.begin_epoch, args.nepochs):
        
        if hvd.rank() == 0:
            logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        train(epoch, model, gmm)
        lipschitz_constants.append(get_lipschitz_constants(model))
        ords.append(get_ords(model))
        if hvd.rank() == 0:
            logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
            logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        # if args.ema_val:
        #     validate(epoch, model,gmm,ema)
        # else:
        #     validate(epoch, model,gmm)

        if args.scheduler and scheduler is not None:
            scheduler.step()

        
        if hvd.rank() == 0 and epoch % args.save_every == 0:
            print("Saving model...")
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                # 'test_bpd': test_bpd,
            }, os.path.join(args.log_dir, 'models'), epoch, last_checkpoints, num_checkpoints=5)
    
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                # 'test_bpd': test_bpd,
            }, os.path.join(args.log_dir, 'models', f'most_recent_{hvd.size()}_workers.pth'))
        
        # validate(args.begin_epoch - 1, model,gmm, ema)
            


if __name__ == '__main__':
    main()
