# WP3 - EXAMODE COLOR INFORMATION with Residual Flows for Invertible Generative Modeling 

## Based on paper
```
@inproceedings{chen2019residualflows,
  title={Residual Flows for Invertible Generative Modeling},
  author={Chen, Ricky T. Q. and Behrmann, Jens and Duvenaud, David and Jacobsen, J{\"{o}}rn{-}Henrik},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2019}
}
```

- Flow-based generative models parameterize probability distributions through an
invertible transformation and can be trained by maximum likelihood. Invertible
residual networks provide a flexible family of transformations where only Lipschitz
conditions rather than strict architectural constraints are needed for enforcing
invertibility.
- This application concerns the invertible mapping of the color information in histopathology Whole Slide Image patches


- Color Information measures can be evaluated using: 
    - Normalized Median Intensity (NMI) measure
    - Standard deviation of NMI
    - Coefficient of variation of NMI
    
    ref: <a href="https://pubmed.ncbi.nlm.nih.gov/26353368/">Stain Specific Standardization of Whole-Slide Histopathological Images</a>


<img  width="250" height="250" src=_images/im_tmpl_30.png> ==> <img  width="250" height="250" src=_images/im_gamma_30.png>


<img  width="250" height="250" src=_images/im_test_30.png> ==> <img  width="250" height="250" src=_images/im_pi_30.png>


<img  width="250" height="250" src=_images/im_test_30.png> ==> <img  width="250" height="250" src=_images/im_conv_30.png>
> The tissue class membership, followed by conversion


## Analysis of Normalized Median Intensity (NMI -  scores)

|        | RadboudUMC (template) - TCGA (target)     |  RadboudUMC (template) - AOEC (target) |
|:------:|:-------------------------:|:-------------------------:|
| DCGMM  |![](_images/Radboudumc-TCGA-boxplot-eval.png)  |  ![](_images/Radboudumc-AOEC-boxplot-eval.png)|
|iResFlow|![](_images/worker-0-Rad_TCGA-boxplot-eval.png)  |  ![](_images/worker-0-Rad_AOEC-boxplot-eval.png)|

> Circles are outliers.

### RadboudUMC (template) - TCGA (target)
| **Model**|**NMI - Standard Deviation**|**NMI - Coefficient of Variation**|
|:--------:|:------------:|:-----------------:|
|   DCGMM  |   0.0686 +- 0.0065     |  0.0776 +- 0.0110 |
| iResFlow |   0.0381 +- 0.0094     |  0.0425 +- 0.0148 |

### RadboudUMC (template) - AOEC (target)
| **Model**|**NMI - Standard Deviation**|**NMI - Coefficient of Variation**|
|:--------:|:------------:|:-----------------:|
|   DCGMM  |   0.0547 +- 0.0222    | 0.0670 +- 0.0249 |
| iResFlow |   0.0497 +- 0.0126    | 0.0563 +- 0.0170  |

> NMI_SD / NMI_CV metrics based on 5 runs of 100 256 x 256 patches of RadboudUMC (template), AOEC (target), TCGA (target) data


## Semantic Segmentation Comparison with DCGMM              
| **Model**|**Parameters**|**Validation mIoU**|
|:--------:|:------------:|:-----------------:|
|   DCGMM  |   517233     |  0.7928 +- 0.0413 |
| iResFlow |   500388     |  0.8477 +- 0.0237 |


> Comparison done on CAMLEYON17 with RadboudUMC data (template), medical center 1 patches of 256 x 256, 4 independent runs of 50 epochs


<img  width="250" height="250" src=_images/dcgmm_center_1_467.png> <==> <img  width="250" height="250" src=_images/tumor_center_1_256_467.png>


<img  width="250" height="250" src=_images/dcgmm_center_1_468.png> <==> <img  width="250" height="250" src=_images/tumor_center_1_256_468.png>

> Left: DCGMM | Right: iResFlow


# Setup
- First install a virtual environment with OpenSlide and PyVips from https://github.com/sara-nl/SURF-deeplab.
- This will install the libraries needed for processing of Whole Slide Images.


Load Modules:
```
module purge
module load 2020
module load NCCL/2.7.8-gcccuda-2020a
module load OpenMPI/4.0.3-GCC-9.3.0
```
Set Environment Variables:
```
VENV_NAME=openslide-pyvips
source ~/virtualenvs/openslide-pyvips/bin/activate

# Setting ENV variables

export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
export HOROVOD_WITH_PYTORCH=1 
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=1
export PATH=$HOME/virtualenvs/$VENV_NAME/bin:$PATH
export LD_LIBRARY_PATH=$HOME/virtualenvs/$VENV_NAME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/virtualenvs/$VENV_NAME/lib:$LD_LIBRARY_PATH
export CPATH=$HOME/virtualenvs/$VENV_NAME/include:$CPATH

```


Install requirements:
```
pip install -r requirements.txt
```

Find hosts of job (on SLURM) and set to `hosts` variable with `slots_per_host` slots available per worker:
```
#!/bin/bash
slots_per_host=4
hosts=""
for host in $(scontrol show hostnames);
do
	hosts="$hosts$host:$slots_per_host,"
done
hosts="${hosts%?}"
echo $hosts
```

Set the appropriate options:

```
mpirun --host $hosts -map-by ppr:4:node -np 1 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py --help

>>>>>
Residual Flow Model Color Information

optional arguments:
  -h, --help            show this help message and exit
  --nclusters NCLUSTERS
                        The amount of tissue classes trained upon (default: 4)
  --slide_path SLIDE_PATH
                        Folder of where the training data whole slide images
                        are located (default: None)
  --label_path LABEL_PATH
                        Folder of where the training data whole slide images
                        masks are located (default: None)
  --valid_slide_path VALID_SLIDE_PATH
                        Folder of where the validation data whole slide images
                        are located (default: None)
  --valid_label_path VALID_LABEL_PATH
                        Folder of where the validation data whole slide images
                        masks are located (default: None)
  --test_path TEST_PATH
                        Folder of where the testing data whole slide images
                        are located (default: None)
  --slide_format SLIDE_FORMAT
                        In which format the whole slide images are saved.
                        (default: tif)
  --label_format LABEL_FORMAT
                        In which format the masks are saved. (default: xml)
  --bb_downsample BB_DOWNSAMPLE
                        OpenSlide(<slide_path>).level_dimensions[bb_downsample
                        ] is used for the contour construction as downsampling
                        level of whole slide image (default: 7)
  --log_dir LOG_DIR     Path of savepath of downsampled image with processed
                        rectangles on it. (default: .)
  --verbose {info,debug}
                        Verbosity of the data sampler (default: info)
  --batch_tumor_ratio BATCH_TUMOR_RATIO
                        round(batch_tumor_ratio*batch_size) images will have
                        tumor in the mini batch, set to 0 if no tumor
                        (default: 0)
  --val_split VAL_SPLIT
  --debug               If running in debug mode (default: False)
  --steps_per_epoch STEPS_PER_EPOCH
                        The hard - coded amount of iterations in one epoch.
                        (default: 1000)
  --print-freq PRINT_FREQ
                        Print progress every so iterations (default: 1)
  --vis-freq VIS_FREQ   Visualize progress every so iterations (default: 5)
  --save_every SAVE_EVERY
                        Save model every so epochs (default: 1)
  --fp16_allreduce      If all reduce in fp16 (default: False)
  --img_size IMG_SIZE   The Field of View (patch size) to use (default: 32)
  --evaluate            If running evaluation (default: False)
  --resume RESUME       File of checkpoint (.pth file) if resuming from
                        checkpoint (default: None)
  --save_conv           Save converted images. (default: False)
  --deploy_samples DEPLOY_SAMPLES
                        How many samples of images should be used as templates
                        in evaluation. (default: 2)
  --begin-epoch BEGIN_EPOCH
  --gather_metrics      If gathering NMI metrics (default: False)
  --nbits NBITS
  --block {resblock,coupling}
  --coeff COEFF
  --vnorms VNORMS
  --n-lipschitz-iters N_LIPSCHITZ_ITERS
  --sn-tol SN_TOL
  --learn-p {True,False}
                        Learn Lipschitz norms, see paper (default: False)
  --n-power-series N_POWER_SERIES
                        Amount of power series evaluated, see paper (default:
                        None)
  --factor-out {True,False}
                        Factorize dimensions, see paper (default: False)
  --n-dist {geometric,poisson}
  --n-samples N_SAMPLES
  --n-exact-terms N_EXACT_TERMS
                        Exact terms computed in series estimation, see paper
                        (default: 2)
  --var-reduc-lr VAR_REDUC_LR
  --neumann-grad {True,False}
                        Neumann gradients, see paper (default: True)
  --mem-eff {True,False}
                        Memory efficient backprop, see paper (default: True)
  --act {softplus,elu,swish,lcube,identity,relu}
  --idim IDIM
  --nblocks NBLOCKS
  --squeeze-first {True,False}
  --actnorm {True,False}
  --fc-actnorm {True,False}
  --batchnorm {True,False}
  --dropout DROPOUT
  --fc {True,False}
  --kernels KERNELS
  --add-noise {True,False}
  --quadratic {True,False}
  --fc-end {True,False}
  --fc-idim FC_IDIM
  --preact {True,False}
  --padding PADDING
  --first-resblock {True,False}
  --cdim CDIM
  --optimizer {adam,adamax,rmsprop,sgd}
  --scheduler {True,False}
  --nepochs NEPOCHS     Number of epochs for training (default: 1000)
  --batch_size BATCH_SIZE
                        Minibatch size (default: 64)
  --lr LR               Learning rate (default: 0.001)
  --wd WD               Weight decay (default: 0)
  --warmup-iters WARMUP_ITERS
  --annealing-iters ANNEALING_ITERS
  --seed SEED
  --ema-val {True,False}
                        Use exponential moving averages of parameters at
                        validation. (default: False)
  --update-freq UPDATE_FREQ
  --task {density,classification,hybrid,gmm}
  --scale-dim {True,False}
  --rcrop-pad-mode {constant,reflect}
  --padding-dist {uniform,gaussian}
```
### Training
```
mpirun --host $hosts -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_*/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --bb_downsample 7 \
 --log_dir experiments/test/ \
 --img_size 512 \
 --batch_tumor_ratio 1 \
 --batch_size 1 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --verbose debug \
 --save experiments/test \
 --nblocks 16 \
 --steps_per_epoch 100 \
 --save_every 2 \
 --vis-freq 100 \
 --nepochs 500 

 
```

- This will train the invertible resnet for 500 epochs with 1 worker nodes with 4 GPU 's, and save visualisations and checkpoints in `experiments/test`.
- It will train on _all_ medical centers of CAMELYON17 (`..../CAMELYON17/training/center_*/`)
    

### Evaluation
```
mpirun --host $hosts -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_0/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --bb_downsample 7 \
 --log_dir experiments/test/ \
 --img_size 512 \
 --batch_tumor_ratio 1 \
 --batch_size 1 \
 --actnorm True \
 --nbits 9 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --verbose debug \
 --nblocks 16 \
 --steps_per_epoch 25 \
 --save_every 1 \
 --vis-freq 10 \
 --nepochs 2 \
 --evaluate \
 --resume experiments/test/models/most_recent_1_workers.pth \
 --save_conv \
 --deploy_samples 100 
```

- This will evaluate the checkpoint with 1 worker nodes with 4 GPU 's in experiments/test/models for 100 samples, and save visualisations in `experiments/test`.


### TODO
- [x] Implement multi node framework (Horovod)

