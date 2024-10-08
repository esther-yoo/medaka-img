#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ey267@cam.ac.uk
#SBATCH -e /nfs/research/birney/users/esther/medaka-img/err/%x-%j.err
#SBATCH -o /nfs/research/birney/users/esther/medaka-img/out/%x-%j.out

# Get the names of the train and test split files
date="2024-10-03"
train_file="/nfs/research/birney/users/esther/medaka-img/src_files/train_set_${date}.csv"
test_file="/nfs/research/birney/users/esther/medaka-img/src_files/test_set_${date}.csv"
train_dir="/nfs/research/birney/users/esther/medaka-img/src_files/train_${date}/raw_images/"
test_dir="/nfs/research/birney/users/esther/medaka-img/src_files/test_${date}/raw_images/"

mkdir -p $train_dir
mkdir -p $test_dir

# Read in the image file names from the train set split, and move the images to the train folder
awk -F ',' '(NR>1) {print $4}' $train_file | while read img_name; do
  if [ -f "/nfs/research/birney/users/esther/medaka-img/all_images/$img_name" ]; then
    scp "/nfs/research/birney/users/esther/medaka-img/all_images/$img_name" $train_dir
  fi
done

# Read in the image file names from the test set split, and move the images to the test folder
awk -F ',' '(NR>1) {print $4}' $test_file | while read img_name; do
  if [ -f "/nfs/research/birney/users/esther/medaka-img/all_images/$img_name" ]; then
    scp "/nfs/research/birney/users/esther/medaka-img/all_images/$img_name" $test_dir
  fi
done