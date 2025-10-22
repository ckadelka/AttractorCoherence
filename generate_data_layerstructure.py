# Import preamble.py
import argparse
import time
import boolforge
import pickle
import os
repo_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Repository directory: {repo_dir}")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run network simulations with specified parameters.")
    parser.add_argument('--num_simulations', type=int, default=10000,help='Number of networks')
    parser.add_argument('--indegree_distribution', type=str, default='constant', help='Indegree distribution type')
    parser.add_argument('--total_nodes', type=int, default=12, help='Total number of nodes')
    parser.add_argument('--avg_degree', type=int, default=3, help='Average degree')
    
    args = parser.parse_args()

    # Main simulation parameters
    num_networks = args.num_simulations
    indegree_distribution = args.indegree_distribution
    num_nodes = args.total_nodes
    avg_degree = args.avg_degree
    
    
    N = num_nodes
    
    for w in range(1,2**(avg_degree-1),2):
        all_basin_sizes = []
        all_basin_coherences = []
        all_attractor_coherences = []
        
        
        all_attractors = []
        layerstructure_NCF = boolforge.get_layer_structure_of_an_NCF_given_its_Hamming_weight(avg_degree,w)
        print(f"Layer structure for weight {w}: {layerstructure_NCF}")
        
        
        for i in range(num_networks):
            start_time = time.time()
            if i % 100 == 0:
                print(f"Simulation {i+1}/{num_networks} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            
            BN = boolforge.random_network(N=12,n=avg_degree,layer_structure = layerstructure_NCF, STRONGLY_CONNECTED = True)
            
            
            robmetrics = BN.get_attractors_and_robustness_measures_synchronous_exact()

            
            all_basin_coherences.append(robmetrics["BasinCoherence"])
            all_basin_sizes.append(robmetrics["BasinSizes"])
            all_attractor_coherences.append(robmetrics["AttractorCoherence"])

            all_attractors.append(robmetrics["Attractors"])
        data_dir = os.path.join(repo_dir, "data")

        # Save to a file
        file_name = f'layeredcanBN_{num_nodes}_{avg_degree}_{layerstructure_NCF}_{num_networks}.pkl'
        file_path = os.path.join(data_dir, file_name)
        data = {'meta_data':f"This file contains simulation for {num_networks} networks with {num_nodes} nodes, {layerstructure_NCF} layer structure and {avg_degree} average degree  with indegree distribution {indegree_distribution}.",
                  'all_basin_coherences': all_basin_coherences, 'all_basin_sizes': all_basin_sizes,'all_attractors': all_attractors,
                'all_attractor_coherences': all_attractor_coherences}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        
   