#!/bin/bash
#number:card的倍数,与portioning相关
card=8
file_path=$1
number=$2
for ((i=0;i<$card;i++))
do
    sum=$(( i + number ))
    echo $sum
    CUDA_VISIBLE_DEVICES=$i python preprocess_llmclean_query.py $file_path $sum &
done

echo "所有进程已启动。"
