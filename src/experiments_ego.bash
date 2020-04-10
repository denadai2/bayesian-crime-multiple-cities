#!/usr/bin/env bash
cities=("bogota1m" "boston1m" "LA1m")
models=("--full")
for c in "${cities[@]}"
do
    for m in "${models[@]}"
    do
        command="nice python3 pystan_fit4.py --city ${c} ${m} -R 1 &> '../data/generated_files/logs/$c-ego-${m}.log' &"

        echo $command
        eval $command
	done
done


cities=("chicago1m")
models=("--sd --uf")
for c in "${cities[@]}"
do
    for m in "${models[@]}"
    do
        command="nice python3 pystan_fit4.py --city ${c} ${m} -R 1 &> '../data/generated_files/logs/$c-ego-${m}.log' &"

        echo $command
        eval $command
	done
done