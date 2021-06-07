#!/bin/bash

# for i in `seq 0 $1`
for i in `seq 0 20`
do
#     python experiment_1D_Task01.py &
#     03-run-BO-for-artificial-model.py --itag=$i --n_start=$1 &
    python 03-run-BO-for-artificial-model.py --itag=$i --n_start=20 &
    sleep 1
done
