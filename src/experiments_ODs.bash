#!/usr/bin/env bash
cities=("boston" "LA" "chicago" "bogota")
models=("--sd" "--uf" "--m" "--full" "--sd --uf" "--sd --m" "--uf --m" "--core-only")
modelstype=("REESF-a" "ESF" "BSF")
for c in "${cities[@]}"
do
    for tp in "${modelstype[@]}"
    do
        for m in "${models[@]}"
        do
            {
            command="nice python3 pystan_fit4.py --city ${c} -M 'BSF' --od-m ${m} -I 20000 -T 15000 -R 1 &> '../data/generated_files/logs/$c-ego-${m}-BSF--od-m.log' &"

            echo $command
            eval $command

            command="nice python3 pystan_fit4.py --city ${c} -M 'BSF' --od-d ${m} -I 20000 -T 15000 -R 1 &> '../data/generated_files/logs/$c-ego-${m}-BSF--od-d.log' &"

            echo $command
            eval $command
            }
            wait
        done
	done
done