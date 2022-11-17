#!/bin/bash
#SBATCH --gres=gpu:a40
#SBATCH --mem=28gb
#SBATCH --cpus-per-task=1
#SBATCH --time=0-24:00:00
#SBATCH --job-name=gentext
#SBATCH --account=argon
#SBATCH --partition=ckpt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=artidoro@uw.edu

# print args
echo $@

# conda activate facteval
source activate miniconda3/envs/transformers


if [ "random" = "$1" ]
then
    python generate.py \
        --batch_size=6 \
        --num_beams=3 \
        --pretrained_model_name_or_path=outputs/$2 \
        --path_to_prompt_data=data/nela-covid-2020-$3.json \
        --path_to_generated_data=data/generated/$2-$3-random.json 

elif [ "topk" = "$1" ]
then
    python generate.py \
        --batch_size=6 \
        --num_beams=3 \
        --top_k=40 \
        --pretrained_model_name_or_path=outputs/$2 \
        --path_to_prompt_data=data/nela-covid-2020-$3.json \
        --path_to_generated_data=data/generated/$2-$3-topk40.json 

elif [ "topp" = "$1" ]
then
    python generate.py \
        --batch_size=6 \
        --num_beams=3 \
        --top_p=0.96 \
        --pretrained_model_name_or_path=outputs/$2 \
        --path_to_prompt_data=data/nela-covid-2020-$3.json \
        --path_to_generated_data=data/generated/$2-$3-topp96.json 
else
    echo "Not a valid option"
fi
