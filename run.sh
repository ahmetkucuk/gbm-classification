
export CUDA_VISIBLE_DEVICES="0"
nohup python train_lenet.py /home/ahmet/workspace/data/tissue_patches/ /home/ahmet/workspace/tensorboard/tissue/ 0.01 256 32 100 8 > training_output.txt 2>&1 &