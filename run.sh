#!/bin/bash
#SBATCH --job-name=Red_Test
#SBATCH --account=project_2006609
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:v100:1

module --force purge
module load pytorch/2.2

#python3 -m venv --system-site-packages venv
source venv/bin/activate
#pip3 install -r req.txt
pip3 install llama-index-llms-llama-cpp  llama-cpp-python

python3 main_vanilla.py

