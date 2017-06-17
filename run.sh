
export CUDA_VISIBLE_DEVICES="1"
nohup python train_model.py /home/ahmet/workspace/data/tissue_patches/ /home/ahmet/workspace/tensorboard/tissue_alexnet/ 0.01 256 32 1000 8 > training_output.txt 2>&1 &