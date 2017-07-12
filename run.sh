
export CUDA_VISIBLE_DEVICES="1"
nohup python train_model.py /home/ahmet/workspace/data/tcga/annotated_patches/ /home/ahmet/workspace/tensorboard/tissue_alexnet_annotated_3/ 0.0001 256 32 100000 2 > output_sgd2.txt 2>&1 &
