import argparse
from distributor import get_connected_graph
import json
import os
import pickle as pkl
from utils import get_average_degree, get_rho, in_range


ap = argparse.ArgumentParser()
ap.add_argument("--num-nodes", required=True, type=int)
ap.add_argument("--degree", required=True, type=int)
ap.add_argument("--factor", required=True, type=int)
ap.add_argument("--tolerance", required=False, type=float, default=0.2)
ap.add_argument("--topology", required=False, type=str, default='rgg')
ap.add_argument("--retries", required=False, type=int, default=10)

args = vars(ap.parse_args())
num_nodes = args['num_nodes']
degree = args['degree']
factor = args['factor']
tolerance = args['tolerance']
topology = args['topology']
retries = args['retries']

print("+"*80)
print(json.dumps(args, indent=4))
print("+"*80)

save = 'r'
while save == 'r':

    graph = get_connected_graph(num_nodes, 0.5, topology)
    avg_deg = get_average_degree(graph)

    for radius in range(1, 10):
        radius = radius/10
        counter = 0
        while not in_range(avg_deg, degree+tolerance, degree-tolerance):
            graph = get_connected_graph(num_nodes, radius, topology)
            if not graph:
                break
            avg_deg = get_average_degree(graph)
            counter += 1
            if counter > retries:
                break

    rho = get_rho(graph, num_nodes, factor)
    file_ = '../graphs/topology_{}_degree_{:.1f}_rho_{:.4f}.pkl'.format(
        topology, avg_deg, rho)
    print("avg_deg: {}\nradius: {}\nrho:{}".format(
        avg_deg, radius, rho))
    print("+"*80)
    save = input("Save {}? (y/n/r)".format(file_))
    if save == 'y':
        overwrite = 'y'
        if os.path.isfile(file_):
            overwrite = input("Overwrite {}? (y/n)")
        if overwrite == 'y':
            print("Saving: ", file_)
            pkl.dump(graph, open(file_, 'wb'))
        else:
            print("Skipping: ", file_)
