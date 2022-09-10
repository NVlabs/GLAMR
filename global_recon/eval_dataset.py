import os, sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'kama'))
import os.path as osp
import pickle
import argparse
import numpy as np
import torch
from global_recon.utils.evaluator import Evaluator


test_sequences = {
    '3dpw': ['downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00', 'downtown_car_00', 'downtown_crossStreets_00', 'downtown_downstairs_00', 
             'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_stairs_00',
             'downtown_upstairs_00', 'downtown_walkBridge_01', 'downtown_walkUphill_00', 'downtown_walking_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00',
             'downtown_windowShopping_00', 'flat_guitar_01', 'flat_packBags_00', 'office_phoneCall_00', 'outdoors_fencing_01']
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='3dpw')
parser.add_argument('--results_dir', default='out/3dpw')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seeds', default="1")
args = parser.parse_args()

results_dir = args.results_dir
sequences = test_sequences[args.dataset]
seeds = [int(x) for x in args.seeds.split(',')]
multi_seeds = len(seeds) > 1

if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')
torch.torch.set_grad_enabled(False)

evaluator = Evaluator(results_dir, args.dataset, device=device, log_file=f'{results_dir}/log_eval.txt', compute_sample=multi_seeds)
seed_evaluator = Evaluator(results_dir, args.dataset, device=device, log_file=f'{results_dir}/log_eval_seed.txt', compute_sample=multi_seeds)

for sind, seq_name in enumerate(sequences[:2]):
    metrics_dict_arr = []
    evaluator.log.info(f'{sind}/{len(sequences)} evaluating global reconstruction for {seq_name}')

    for seed in seeds:
        data_file = f'{results_dir}/{seq_name}/grecon/{seq_name}_seed{seed}.pkl'
        data = pickle.load(open(data_file, 'rb'))

        metrics_dict = seed_evaluator.compute_sequence_metrics(data, seq_name, accumulate=False)
        metrics_dict_arr.append(metrics_dict)

    metrics_dict_allseeds = evaluator.metrics_from_multiple_seeds(metrics_dict_arr)
    evaluator.update_accumulated_metrics(metrics_dict_allseeds, seq_name)
    evaluator.print_metrics(metrics_dict_allseeds, prefix=f'{sind}/{len(sequences)} --- All seeds {seq_name} --- ', print_accum=False)

evaluator.print_metrics(prefix=f'Total ------- ', print_accum=True)