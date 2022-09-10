#!/bin/bash

#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p gpu_titanrtx
np=$(($SLURM_NNODES * 4))

module purge
module load 2020
module load OpenMPI/3.1.4-GCC-8.3.0
module load NCCL/2.5.6-CUDA-10.1.243
module list


VENV_NAME=dspeed
#source ~/virtualenvs/openslide-torch/bin/activate
source ~/virtualenvs/dspeed/bin/activate


# Setting ENV variables

# Export MPICC
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



#pip3 install --no-cache-dir --upgrade --force-reinstall horovod
#pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip install scikit-learn 
#pip install Pillow 
#pip install tqdm 
#pip install six
#pip install opencv-python
#pip install openslide-python
#pip install torchsummary
#pip install scikit-image
#pip install torchsummary
#pip install horovod[pytorch]


 


hosts=`sh ~/hosts.sh`
 

#/nfs/managed_datasets/CAMELYON17/training/center_1/ 
mpirun --host $hosts -map-by ppr:4:node -np 1 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_0/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --valid_slide_path '/home/rubenh/color-information/templates' \
 --valid_label_path '/home/rubenh/color-information/templates' \
 --bb_downsample 7 \
 --log_dir experiments/center_0/ \
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
 --resume experiments/center_0/models/most_recent_1_workers.pth \
 --save_conv \
 --deploy_samples 100 
 
 
 
 
 
 
 
 
 
 mpirun --host $hosts -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_1/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --valid_slide_path '/home/rubenh/color-information/templates' \
 --valid_label_path '/home/rubenh/color-information/templates' \
 --test_path '/home/rubenh/color-information/test' \
 --bb_downsample 7 \
 --log_dir experiments/center_1/ \
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
 --nepochs 2 
 
 mpirun --host $hosts -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_2/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --valid_slide_path '/home/rubenh/color-information/templates' \
 --valid_label_path '/home/rubenh/color-information/templates' \
 --test_path '/home/rubenh/color-information/test' \
 --bb_downsample 7 \
 --log_dir experiments/center_2/ \
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
 --nepochs 2 
 
 mpirun --host $hosts -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_3/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --valid_slide_path '/home/rubenh/color-information/templates' \
 --valid_label_path '/home/rubenh/color-information/templates' \
 --test_path '/home/rubenh/color-information/test' \
 --bb_downsample 7 \
 --log_dir experiments/center_3/ \
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
 --nepochs 2 

mpirun --host $hosts -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --slide_format tif \
 --slide_path '/nfs/managed_datasets/CAMELYON17/training/center_4/' \
 --label_path '/nfs/managed_datasets/CAMELYON17/training' \
 --valid_slide_path '/home/rubenh/color-information/templates' \
 --valid_label_path '/home/rubenh/color-information/templates' \
 --test_path '/home/rubenh/color-information/test' \
 --bb_downsample 7 \
 --log_dir experiments/center_4/ \
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
 --nepochs 2 
 
exit


mpirun -map-by ppr:4:node -np 8 -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --data custom \
 --fp16_allreduce \
 --train_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
 --valid_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc  \
 --imagesize 256 \
 --batchsize 4 \
 --val-batchsize 4 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --save experiments/Rad_AOEC \
 --nblocks 21 \
 --nclusters 4 \
 --vis-freq 10 \
 --nepochs 5 \
 --resume /home/rubenh/examode/color-information/checkpoints/Radboudumc_8_workers.pth \
 --save_conv True

exit

"""
TRAINING

 mpirun -map-by ppr:4:node -np $np -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --data custom \
 --train_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
 --valid_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
 --imagesize 256 \
 --batchsize 4 \
 --val-batchsize 4 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --factor-out True \
 --save experiments/Radboudumc \
 --nblocks 21 \
 --nclusters 3 \
 --vis-freq 4 \
 --nepochs 10 \
 --lr 1e-3 \
 --idim 128

"""
 
"""

EVALUATION

  mpirun -map-by ppr:4:node -np $np -x LD_LIBRARY_PATH -x PATH python -u train_img_horo.py \
 --data custom \
 --fp16_allreduce \
 --train_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
 --valid_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/AOEC \
 --imagesize 256 \
 --batchsize 1 \
 --val-batchsize 1 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --save experiments/Radboudumc \
 --nblocks 16 \
 --vis-freq 10 \
 --nepochs 5 \
 --resume /home/rubenh/examode/color-information/experiments/Radboudumc/models/most_recent_4_workers.pth \
 --save_conv True
 
"""
