#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --time=2:30:00
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ey267@cam.ac.uk
#SBATCH -e /nfs/research/birney/users/esther/medaka-img/err/%x-%j.err
#SBATCH -o /nfs/research/birney/users/esther/medaka-img/out/%x-%j.out

module purge
module load nextflow/23.04.1
module load singularity-3.8.7-gcc-11.2.0-jtpp6xx

export TOWER_ACCESS_TOKEN=eyJ0aWQiOiA4NjgyfS4yM2EzMjYyNWZmZTczNTkzZTdhOTc3NTU1ZjE3NWI5ODlkMDA3NDA3
export NXF_VER=23.04.1
export NXF_OPTS="-Xms500M -Xmx2G"

nextflow run birneylab/flexlmm -profile medaka,stitch,ebi_codon_slurm,singularity -params-file /nfs/research/birney/users/esther/medaka-img/scripts/flexlmm_params/convnet-ae-pytorch-medaka/flexlmm_params_restful-sweep-4-epoch980-PC9.yaml -r main -with-tower