import os 
import glob
import numpy as np 
import random
import copy
import time
import argparse
from pathlib import Path
import torch.nn.functional as F

import networkx as nx

# from ..utils.aiger_utils import aiger_utils
# from ..utils.circuit_utils import circuit_utils
import shutil
from utils.utils import run_command
from collections import defaultdict
from multiprocessing import Pool, cpu_count
gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}

import sys
sys.setrecursionlimit(1000000)

def get_parse_args():
    parser = argparse.ArgumentParser()

    # Range
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=200000, type=int)
    
    parser.add_argument('--lib_path', default=str(Path(__file__).with_name('sky130_wo_adder.lib')), type=str)
    # Input
    parser.add_argument('--aig_dir', default='../../../../data/aig/raw', type=str)
    
    # Output
    parser.add_argument('--pm_dir', default='../../../../data/pm_netlist/raw', type=str)
    
    args = parser.parse_args()
    
    return args


def mapping(data):
    i,aig = data
    aig_file = os.path.join(args.aig_dir, aig + '.aig')
    pm_file = os.path.join(args.pm_dir, aig + '.v')
    cmd = f'/data/shared/abc_map -c "read {args.lib_path}; read_aiger {aig_file}; strash; map; write_verilog {pm_file}"'
    run_command(cmd)

    aig_file = os.path.join(args.aig_dir, aig + '_syn.aig')
    pm_file = os.path.join(args.pm_dir, aig + '_syn.v')
    cmd = f'/data/shared/abc_map -c "read {args.lib_path}; read_aiger {aig_file}; strash; map; write_verilog {pm_file}"'
    run_command(cmd)
    if i%100 == 0:
        print(f'process {i} circuits')

if __name__ == '__main__':     
    args = get_parse_args()
    
    aig_files = glob.glob('{}/*.aig'.format(args.aig_dir))
    aig_namelist = []
    for aig_file in aig_files:
        aig_name = os.path.basename(aig_file).replace('.aig', '')
        if aig_name.split('_')[-1] == 'rd' or aig_name.split('_')[-1] == 'syn':
            continue
        aig_namelist.append(aig_name)
    
    no_circuits = len(aig_namelist)
    tot_time = 0

    # for i,aig in enumerate(aig_namelist):
    #     aig_file = os.path.join(args.aig_dir, aig + '.aig')
    #     pm_file = os.path.join(args.pm_dir, aig + '.v')
    #     cmd = f'abc -c "read {args.lib_path}; read_aiger {aig_file}; strash; map; write_verilog {pm_file}"'
    #     run_command(cmd)
    #     if i%100 == 0:
    #         print(f'process {i}/{no_circuits} circuits')
    data = [(i, aig) for i,aig in enumerate(aig_namelist)]

    num_workers = 16

    with Pool(num_workers) as pool:
        pool.map(mapping,  data)

    print('finish all')
