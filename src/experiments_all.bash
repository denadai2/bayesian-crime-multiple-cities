#!/usr/bin/env bash
cities=("bogota" "boston" "LA" "chicago")
models=("--sd" "--uf" "--m" "--full" "--sd --uf" "--sd --m" "--uf --m" "--core-only")
modelstype=("BSF" "REESF-a" "ESF")
for c in "${cities[@]}"
do
    for tp in "${modelstype[@]}"
    do
        {
        for m in "${models[@]}"
        do
            command="nice python3 pystan_fit4.py --city ${c} -M ${tp} ${m} -R 1 &> '../data/generated_files/logs/$c-ego-${m}-${tp}.log' &"

            echo $command
            eval $command
        done
        }
        wait
	done
done