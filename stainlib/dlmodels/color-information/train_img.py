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
from skimage import io
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image
import random
import torch.utils.data.distributed
#import horovod.torch as hvd
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets
from torchsummary import summary

from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
from lib.GMM import GMM_model as gmm
import lib.image_transforms as imgtf
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts


"""
TODO:


"""
# Arguments
parser = argparse.ArgumentParser(description='Residual Flow Model Color Information', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--data', type=str, default='custom', choices=[
        'custom'
    ]
)
# mnist
parser.add_argument('--dataroot', type=str, default='data')
## GMM ##
parser.add_argument('--nclusters', type=int, default=4,help='The amount of tissue classes trained upon')

## CAMELYON ##
parser.add_argument('--dataset', type=str, default="17", help='Which dataset to use. "16" for CAMELYON16 or "17" for CAMELYON17')
parser.add_argument('--train_centers', nargs='+', default=[-1], type=int, help='Centers for training. Use -1 for all, otherwise 2 3 4 eg.')
parser.add_argument('--val_centers', nargs='+', default=[-1], type=int,help='Centers for validation. Use -1 for all, otherwise 2 3 4 eg.')
parser.add_argument('--train_path', type=str, help='Folder of where the training data is located', default=None)
parser.add_argument('--valid_path', type=str, help='Folder where the validation data is located', default=None)
parser.add_argument('--val_split', type=float, default=0.15)
parser.add_argument('--debug', action='store_true', help='If running in debug mode')
parser.add_argument('--fp16_allreduce', action='store_true', help='If all reduce in fp16')
##
parser.add_argument('--imagesize', type=int, default=32)
# 28
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

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
parser.add_argument('--idim', type=int, default=512)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)

parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
# 0
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid','gmm'], default='gmm')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--nworkers', type=int, default=8)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=1)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=5)
args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)



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
    
    
def get_image_lists(args):
    """ Get the image lists"""

    if args.dataset == "17":
        image_list, mask_list, val_image_list, val_mask_list, sample_weight_list = load_camelyon17(args)
    elif args.dataset == "16":
        image_list, mask_list, val_image_list, val_mask_list, sample_weight_list = load_camelyon_16(args)

    print('Found', len(image_list), 'training images')
    print('Found', len(mask_list), 'training masks')
    print('Found', len(val_image_list), 'validation images')
    print('Found', len(val_mask_list), 'validation masks')
    return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list


def open_img(path):
    return np.asarray(Image.open(path))[:, :, 0] / 255


def get_valid_idx(mask_list):
    """ Get the valid indices of masks by opening images in parallel """
    num_cores = multiprocessing.cpu_count()
    data = Parallel(n_jobs=num_cores)(delayed(open_img)(i) for i in mask_list)
    return data

def load_camelyon_16(args):
    """  Load the camelyon16 dataset """
    image_list = [x for x in sorted(glob(str(args.train_path) + '/*', recursive=True)) if 'mask' not in x]
    mask_list = [x for x in sorted(glob(str(args.train_path) + '/*', recursive=True)) if 'mask' in x]
    
    if args.debug:
        image_list, mask_list = shuffle(image_list[:5], mask_list[:5])
    else:
        image_list, mask_list = shuffle(image_list, mask_list)

    sample_weight_list = [1.0] * len(image_list)

    val_split = int(len(image_list) * (1-args.val_split))
    val_image_list = image_list[val_split:]
    val_mask_list = mask_list[val_split:]
    sample_weight_list = sample_weight_list[:val_split]
    image_list = image_list[:val_split]
    mask_list = mask_list[:val_split]

    # idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
    idx = get_valid_idx(val_mask_list)
    num_pixels = args.imagesize ** 2
    valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
    valid_idx = [i for i, x in enumerate(valid_idx) if x]

    val_image_list = [val_image_list[i] for i in valid_idx]
    val_mask_list = [val_mask_list[i] for i in valid_idx]

    val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)

    return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list


def load_camelyon17(args):
    """ Load the camelyon17 dataset """
    image_list = [x for c in args.train_centers for x in sorted(glob(str(args.train_path).replace('center_XX', f'center_{c}') + f'/patches_positive_{args.imagesize}/*', recursive=True)) if 'mask' not in x]
    
    mask_list = [x for c in args.train_centers for x in sorted(glob(str(args.train_path).replace('center_XX', f'center_{c}') + f'/patches_positive_{args.imagesize}/*', recursive=True)) if'mask' in x]
    if args.debug:
        image_list, mask_list = shuffle(image_list[:5], mask_list[:5])
    else:
        image_list, mask_list = image_list, mask_list
        # image_list, mask_list = shuffle(image_list, mask_list)
    
    sample_weight_list = [1.0] * len(image_list)

    # If validating on everything, 00 custom
    if args.val_centers == [1, 2, 3, 4]:
        val_split = int(len(image_list) * (1-args.val_split))
        val_image_list = image_list[val_split:]
        val_mask_list = mask_list[val_split:]
        sample_weight_list = sample_weight_list[:val_split]
        image_list = image_list[:val_split]
        mask_list = mask_list[:val_split]

        idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
        num_pixels = args.imagesize ** 2
        valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
        valid_idx = [i for i, x in enumerate(valid_idx) if x]

        val_image_list = [val_image_list[i] for i in valid_idx]
        val_mask_list = [val_mask_list[i] for i in valid_idx]

        val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)

    else:
        val_image_list = [x for c in args.val_centers for x in
                          sorted(glob(args.valid_path.replace('center_XX', f'center_{c}') + f'/patches_positive_{args.imagesize}/*', recursive=True)) if
                          'mask' not in x]
        val_mask_list = [x for c in args.val_centers for x in
                         sorted(glob(args.valid_path.replace('center_XX', f'center_{c}') + f'/patches_positive_{args.imagesize}/*', recursive=True)) if
                         'mask' in x]
        
        # if args.debug:
        #     val_image_list, val_mask_list = val_image_list[:5], val_mask_list[:5]
            
        # # idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
        # idx = get_valid_idx(val_mask_list)
        # num_pixels = args.imagesize ** 2
        # valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
        # valid_idx = [i for i, x in enumerate(valid_idx) if x]

        # val_image_list = [val_image_list[i] for i in valid_idx]
        # val_mask_list = [val_mask_list[i] for i in valid_idx]

        # val_split = int(len(image_list) * args.val_split)
        # val_image_list = val_image_list[:val_split]
        # val_mask_list = val_mask_list[:val_split]


        # val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)
    return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list



logger.info('Loading dataset {}'.format(args.data))



class make_dataset(torch.utils.data.Dataset):
    """Make Pytorch dataset."""
    def __init__(self, image_list, mask_list,train=True):
        """
        Args:
            image_list 
            mask_list 
        """
        self.image_list = image_list
        self.mask_list  = mask_list
        self.train = train
        if train:
            self.transform  = transforms.Compose([
                                # transforms.ToPILImage(),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                reduce_bits,
                                lambda x: add_noise(x, nvals=2**args.nbits),
                            ])
        else:
            self.transform  = transforms.Compose([
                                transforms.ToTensor(),
                                reduce_bits,
                                lambda x: add_noise(x, nvals=2**args.nbits),
                            ])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = io.imread(image_list[idx],as_gray=False, pilmode="RGB")
        mask  = io.imread(mask_list[idx],as_gray=True)
        
        # im = image.astype('uint8')
        # im = Image.fromarray(im)
        # im.save('test1.png')
        image = imgtf.RGB2HSD(image/255.0).astype('float32')
        image = self.transform(image)
        
        
        # im = image.permute(1,2,0)
        # im = im.cpu().detach().numpy()
        # im = im * 255
        # im = im.astype('uint8')
        # im = Image.fromarray(im)
        # im.save('test2.png')
        mask  = mask 
        sample = (image ,mask)
        return sample





# Dataset and hyperparameters
if args.data == 'celebahq':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 256:
        logger.info('Changing image size to 256.')
        args.imagesize = 256
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                reduce_bits,
                lambda x: add_noise(x, nvals=2**args.nbits),
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=False, transform=transforms.Compose([
                reduce_bits,
                lambda x: add_noise(x, nvals=2**args.nbits),
            ])
        ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )
elif args.data == 'custom':
    im_dim = args.nclusters
    n_classes = args.nclusters
    init_layer = layers.LogitTransform(0.05)
    
    image_list, mask_list, val_image_list, val_mask_list, sample_weight_list = get_image_lists(args)
    train_dataset = make_dataset(image_list, mask_list,train=True)
    test_dataset  = make_dataset(val_image_list, val_mask_list,train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batchsize, shuffle=False)

if args.task in ['classification', 'hybrid','gmm']:
    try:
        n_classes
    except NameError:
        raise ValueError('Cannot perform classification with {}'.format(args.data))
else:
    n_classes = 1

logger.info('Dataset loaded.')
logger.info('Creating model.')

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)
dataset_size = len(train_loader.dataset)

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

ema = utils.ExponentialMovingAverage(model)

def parallelize(model):
    return torch.nn.DataParallel(model)

model = parallelize(model)
gmm   = parallelize(gmm)
model.to(device)
gmm.to(device)

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



best_test_bpd = math.inf
if (args.resume is not None):
    logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    checkpt = torch.load(args.resume)
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    if not isinstance(model,torch.nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    # state.update(sd)
    model.load_state_dict(state, strict=True)
    ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state

logger.info(optimizer)

fixed_z = standard_normal_sample([min(32, args.batchsize),
                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)



def compute_loss(x, model,gmm, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)


    if args.data == 'celebahq' or 'custom':
        nvals = 2**args.nbits
    else:
        nvals = 256

    x, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    
    if args.task == 'gmm' :
        D = x[:,0,...].unsqueeze(0)
        D = rescale(D) # rescaling to [0,1]
        D = D.repeat(1, args.nclusters, 1, 1)
        z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
        

        z, delta_logp = z_logp
        
        # log p(z)
        # logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (args.imagesize * args.imagesize * (im_dim + args.padding)) - logpu
        bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

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


def train(epoch, model,gmm):

    model.train()
    gmm.train()

    
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        
        x = x.to(device)
        
        # for i in range(args.gmmsteps):
        global_itr = epoch * len(train_loader) + i
        update_lr(optimizer, global_itr)
        print(f'Step {global_itr}')
        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        beta = beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.

        bpd, logits, logpz, neg_delta_logp, params = compute_loss(x, model,gmm, beta=beta)

        firmom, secmom = estimator_moments(model)

        bpd_meter.update(bpd.item())
        logpz_meter.update(logpz.item())
        deltalogp_meter.update(neg_delta_logp.item())
        firmom_meter.update(firmom)
        secmom_meter.update(secmom)



        # compute gradient and do SGD step
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
            print("Optimizer.step() done")
            optimizer.zero_grad()
            
            update_lipschitz(model)
            ema.apply()

            gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
                )
            )

            if args.task in ['density', 'hybrid','gmm']:
                s += (
                    f' | Bits/dim {bpd_meter.val}({bpd_meter.avg}) | '
                    # f' | params {[p.clone() for p in gmm.parameters().grad]}) | '
                    f'Logpz {logpz_meter.avg} | '
                    f'-DeltaLogp {deltalogp_meter.avg} | '
                    f'EstMoment ({firmom_meter.avg},{secmom_meter.avg})'
                    )
            

            logger.info(s)
        if i % args.vis_freq == 0 and i > 0:
            visualize(epoch, model,gmm, i, x, global_itr)
    
        del x
        torch.cuda.empty_cache()
        gc.collect()
        if i == len(train_loader) - 2: break
    return


def validate(epoch, model,gmm, ema=None):
    """
    - Deploys the color normalization on test image dataset
    - Evaluates NMI / CV / SD
    # Evaluates the cross entropy between p_data and p_model.
    """
    print("Starting Validation")
    model = parallelize(model)
    gmm   = parallelize(gmm)
    
    model.to(device)
    gmm.to(device)

    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    update_lipschitz(model)

    model.eval()
    gmm.eval()

    mu_tmpl = 0
    std_tmpl = 0
    N = 0
    

    print(f"Deploying on {len(train_loader)} batches of {args.batchsize} templates...")
    idx = 0
    for x, y in tqdm(train_loader):
        x = x.to(device)
        ### TEMPLATES ###
        D = x[:,0,...].unsqueeze(1)
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))

        mu, std, gamma =  params
        mu  = mu.cpu().numpy()
        std = std.cpu().numpy()
        gamma    = gamma.cpu().numpy() 
    
        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)
        
          
        N = N+1
        mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
        std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
        
        if idx == len(train_loader) - 1: break
        idx+=1
                
        
        
      
    print("Estimated Mu for template(s):")
    print(mu_tmpl)
      
    print("Estimated Sigma for template(s):")
    print(std_tmpl)
    
          
    metrics = dict()
    for tc in range(1,args.nclusters+1):
        metrics[f'mean_{tc}'] = []
        metrics[f'median_{tc}']=[]
        metrics[f'perc_95_{tc}']=[]
        metrics[f'nmi_{tc}']=[]
        metrics[f'sd_{tc}']=[]
        metrics[f'cv_{tc}']=[]
    
    print(f"Predicting on {len(test_loader)} batches of {args.val_batchsize} templates...")
    idx=0
    for x_test, y_test in tqdm(test_loader):
        x_test = x_test.to(device)
        ### DEPLOY ###
        D = x_test[:,0,...].unsqueeze(1)
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
        
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))

        
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
        
        ClsLbl = np.argmax(np.asarray(pi),axis=-1)
        ClsLbl = ClsLbl.astype('int32')
        mean_rgb = np.mean(X_conv,axis=-1)
        pdb.set_trace()
        for tc in range(1,args.nclusters+1):
            msk = ClsLbl==tc
            if not msk.any(): continue # skip metric if no class labels are found
            ma = mean_rgb[msk]
            mean = np.mean(ma)
            median = np.median(ma)
            perc = np.percentile(ma, 95)
            nmi = median / perc
            metrics[f'mean_{tc}'].append(mean)
            metrics[f'median_{tc}'].append(median)
            metrics[f'perc_95_{tc}'].append(perc)
            metrics[f'nmi_{tc}'].append(nmi)
        
        if idx == len(test_loader) - 1: break
        idx+=1
        
  
    av_sd = []
    av_cv = []
    for tc in range(1,args.nclusters+1):
        if len(metrics[f'mean_{tc}']) == 0: continue
        metrics[f'sd_{tc}'] = np.array(metrics[f'nmi_{tc}']).std()
        metrics[f'cv_{tc}'] = np.array(metrics[f'nmi_{tc}']).std() / np.array(metrics[f'nmi_{tc}']).mean()
        print(f'sd_{tc}:', metrics[f'sd_{tc}'])
        print(f'cv_{tc}:', metrics[f'cv_{tc}'])
        av_sd.append(metrics[f'sd_{tc}'])
        av_cv.append(metrics[f'cv_{tc}'])
    
    print(f"Average sd = {np.array(av_sd).mean()}")
    print(f"Average cv = {np.array(av_cv).mean()}")
    import csv
    file = open(f"metrics-{args.train_centers[0]}-{args.val_centers[0]}.csv","w")
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
    
    return


def visualize(epoch, model, gmm, itr, real_imgs, global_itr):
    print("Starting Visualisation")
    model.eval()
    gmm.eval()
    utils.makedirs(os.path.join(args.save, 'imgs'))

    for x_test, y_test in test_loader:
        # x_test = x_test[0,...].unsqueeze(0)
        # y_test = y_test[0,...].unsqueeze(0)
        x_test = x_test.to(device)
        ### TEMPLATES ###
        D = real_imgs[:,0,...].unsqueeze(1)
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
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))

        mu_tmpl, std_tmpl, gamma =  params
        mu_tmpl  = mu_tmpl.cpu().numpy()
        std_tmpl = std_tmpl.cpu().numpy()
        gamma    = gamma.cpu().numpy() 
    
        mu_tmpl  = mu_tmpl[...,np.newaxis]
        std_tmpl = std_tmpl[...,np.newaxis]
        
        mu_tmpl = np.swapaxes(mu_tmpl,0,1) # (3,4,1) -> (4,3,1)
        mu_tmpl = np.swapaxes(mu_tmpl,1,2) # (4,3,1) -> (4,1,3)
        std_tmpl = np.swapaxes(std_tmpl,0,1) # (3,4,1) -> (4,3,1)
        std_tmpl = np.swapaxes(std_tmpl,1,2) # (4,3,1) -> (4,1,3)
        
            
        ### DEPLOY ###
        D = x_test[:,0,...].unsqueeze(1)
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
        
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))

    
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

        # save a random image from the batch
        im_no = random.randint(0,args.batchsize-1) 
        im_tmpl = real_imgs[im_no,...].cpu().numpy()
        im_tmpl = np.swapaxes(im_tmpl,0,1)
        im_tmpl = np.swapaxes(im_tmpl,1,-1)
        im_tmpl = imgtf.HSD2RGB_Numpy(im_tmpl)
        im_tmpl = (im_tmpl*255).astype('uint8')
        im_tmpl = Image.fromarray(im_tmpl)
        im_tmpl.save(os.path.join(args.save,'imgs',f'im_tmpl_{global_itr}.png'))
        
        im_test = x_test[im_no,...].cpu().numpy()
        im_test = np.swapaxes(im_test,0,1)
        im_test = np.swapaxes(im_test,1,-1)
        im_test = imgtf.HSD2RGB_Numpy(im_test)
        im_test = (im_test*255).astype('uint8')
        im_test = Image.fromarray(im_test)
        im_test.save(os.path.join(args.save,'imgs',f'im_test_{global_itr}.png'))
        
        im_D = D[0,0,...].cpu().numpy()
        im_D = (im_D*255).astype('uint8')
        im_D = Image.fromarray(im_D,'L')
        im_D.save(os.path.join(args.save,'imgs',f'im_D_{global_itr}.png'))
        
        im_conv = X_conv[im_no,...].reshape(args.imagesize,args.imagesize,3)
        im_conv = Image.fromarray(im_conv)
        im_conv.save(os.path.join(args.save,'imgs',f'im_conv_{global_itr}.png'))
        
        # gamma
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
        im_gamma.save(os.path.join(args.save,'imgs',f'im_gamma_{global_itr}.png'))
        
        # pi
        ClsLbl = np.argmax(pi, axis=-1)
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
        im_gamma.save(os.path.join(args.save,'imgs',f'im_pi_{global_itr}.png'))
        
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

    if args.resume:
        validate(args.begin_epoch - 1, model,gmm, ema)
        sys.exit(0)
        
    for epoch in range(args.begin_epoch, args.nepochs):

        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        
        train(epoch, model, gmm)
        lipschitz_constants.append(get_lipschitz_constants(model))
        ords.append(get_ords(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
        logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        # if args.ema_val:
        #     test_bpd = validate(epoch, model, ema)
        # else:
        #     test_bpd = validate(epoch, model)

        if args.scheduler and scheduler is not None:
            scheduler.step()

        # if test_bpd < best_test_bpd:
        #     best_test_bpd = test_bpd
        
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'ema': ema,
            # 'test_bpd': test_bpd,
        }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'ema': ema,
            # 'test_bpd': test_bpd,
        }, os.path.join(args.save, 'models', 'most_recent.pth'))
        


if __name__ == '__main__':
    main()
