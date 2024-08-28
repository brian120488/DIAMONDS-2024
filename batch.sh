#!/bin/bash
#SBATCH -p preempt  #if you don't have ccgpu access, use "preempt"
#SBATCH -n 16   # 8 cpu cores
#SBATCH --mem=64g       #64GB of RAM
#SBATCH --time=2-0      #run 2 days, up to 7 days "7-00:00:00"
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -N 1
#SBATCH --gres=gpu:1    # number of GPUs. please follow instructions in Pax User Guide when submit jobs to different partition and selecting different GPU architectures.

module load alphafold/2.1.1
module list
nvidia-smi

module help alphafold/2.1.1 # this command will print out all input options for "runaf2" command

#Please use your own path/value for the following variables
#Make sure to specify the outputpath to a path that you have write permission
outputpath=/cluster/tufts/tasissalab/byang05/alpha-fold/
fastapath=/cluster/tufts/tasissalab/byang05/alpha-fold/2PYB.fasta
maxtemplatedate=2020-06-10

source activate alphafold2.1.1

#running alphafold 2.1.1

runaf2 -o $outputpath -f $fastapath -t $maxtemplatedate