for seed in 0 1 2 3 4;do
  python sampling.py --seed $seed --task 'Heisenberg_8' --sample_num 111000 --device_name 'grid_16q' --num_layer 10
done

for seed in 0 1 2 3 4;do
  python sampling.py --seed $seed --task 'TFIM_8' --sample_num 111000 --device_name 'grid_16q' --num_layer 10
done

for seed in 0 1 2 3 4;do
  python sampling.py --seed $seed --task 'TFCluster_8' --sample_num 111000 --device_name 'grid_16q' --num_layer 10
done



