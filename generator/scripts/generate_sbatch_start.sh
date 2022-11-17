for split in train valid test
do
    for mode in "random" "topk" "topp"
    do
        sbatch outputs/generate_sbatch.sh \
            $mode gpt2-xl-covid $split
    done
done