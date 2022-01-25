import argparse
import json

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--num-clusters', type=int, nargs='+', required=True)
    parser.add_argument('--uniform-clusters', type=bool, required=False, default=True)
    parser.add_argument('--shuffle-workers', type=bool, required=False, default=False)
    parser.add_argument('--batch-size', type=int, required=False, default=0)
    parser.add_argument('--test-batch-size', type=int, required=False, default=0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--nesterov', type=bool, required=False, default=False)
    parser.add_argument('--eta', type=int, required=False, default=10)
    parser.add_argument('--decay' type=float, required=False, default=0.1)
    parser.add_argument('--no-cuda', type=bool, required=False, default=False)
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--log-interval', type=int, required=False, default=1)
    parser.add_argument('--save-model', type=bool, required=False, default=True)
    parser.add_argument('--stratify', type=bool, required=False, default=True)
    parser.add_argument('--uniform-data', type=bool, required=False, default=True)
    parser.add_argument('--shuffle-data', type=bool, required=False, default=True)
    parser.add_argument('--non-iid', type=int, required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--radius', type=str, required=True)
    parser.add_argument('--use-same-graphs', type=bool, required=False, default=True)
    parser.add_argument('--graph', type=str, required=False, default='multi')
    parser.add_argument('--d2d', type=float, required=False, default=1.0)
    parser.add_argument('--factor', type=int, required=True)
    parser.add_argument('--var-theta', type=bool, required=True)
    parser.add_argument('--true-eps', type=bool, required=True)
    parser.add_argument('--alpha', type=float, required=True)
