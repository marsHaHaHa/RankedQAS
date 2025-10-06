#!/bin/bash

tasks=("Heisenberg_8" "TFIM_8" "TFCluster_8")
seeds=(0 1 2 3 4)

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "Running task $task with seed $seed"
    python QAS.py --seed $seed \
      --task_name "$task" \
      --top_k_save 20 \
      --noise True
  done
done
