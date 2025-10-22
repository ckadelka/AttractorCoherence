# Import preamble.py
import argparse
import time
import boolforge
import numpy as np
import pickle
import os
repo_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Repository directory: {repo_dir}")
if __name__ == "__main__":
    from collections import defaultdict
    import itertools

    parser = argparse.ArgumentParser(description="Run network simulations with specified parameters.")
    parser.add_argument('--num_simulations', type=int, default=10000,help='Number of networks')
    parser.add_argument('--indegree_distribution', type=str, default='constant', help='Indegree distribution type')
    parser.add_argument('--total_nodes', type=int, default=12, help='Total number of nodes')
    parser.add_argument('--avg_degree', type=int, default=5, help='Average degree')
    
    

    args = parser.parse_args()

    # Main simulation parameters
    num_networks = args.num_simulations
    indegree_distribution = args.indegree_distribution
    num_nodes = args.total_nodes
    avg_degree = args.avg_degree
    can_depths = [i for i in range(avg_degree-1)]+[avg_degree]
    


    
    

    
    
    N = num_nodes
    
    for cd in can_depths:
        all_basin_sizes = []
        all_basin_coherences = []
        all_attractor_coherences = []
        
        
        all_attractors = []
        
        
        
        for i in range(num_networks):
            start_time = time.time()
            if i % 100 == 0:
                print(f"Simulation {i+1}/{num_networks} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            
            BN = boolforge.random_network(N=12,n=avg_degree,depth = cd, EXACT_DEPTH = True, STRONGLY_CONNECTED = True)
            
            
            robmetrics = BN.get_attractors_and_robustness_measures_synchronous_exact()
            
            # return dict(zip(["Attractors", "ExactNumberOfAttractors", 
            #              "BasinSizes","AttractorDict",
            #              "Coherence", "Fragility",
            #              "BasinCoherence", "BasinFragility",
            #              "AttractorCoherence", "AttractorFragility"],
            #         (attractors, n_attractors, 
            #          basin_sizes, attractor_dict,
            #          coherence,fragility,
            #          basin_coherences, basin_fragilities,
            #          attractor_coherences, attractor_fragilities)))
            
            
            all_basin_coherences.append(robmetrics["BasinCoherence"])
            all_basin_sizes.append(robmetrics["BasinSizes"])
            all_attractor_coherences.append(robmetrics["AttractorCoherence"])

            all_attractors.append(robmetrics["Attractors"])
        data_dir = os.path.join(repo_dir, "data")

        # Save to a file
        file_name = f'canBNexactdepth_{num_nodes}_{avg_degree}_{cd}_{num_networks}.pkl'
        file_path = os.path.join(data_dir, file_name)
        data = {'meta_data':f"This file contains simulation for {num_networks} networks with {num_nodes} nodes, {cd} canalizing depth and {avg_degree} average degree  with indegree distribution {indegree_distribution}.",
                  'all_basin_coherences': all_basin_coherences, 'all_basin_sizes': all_basin_sizes,'all_attractors': all_attractors,
                'all_attractor_coherences': all_attractor_coherences}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    # print(all_effective_degrees)
    # print(all_basin_coherence)
    # print(all_frozen_core_coherence)
    # print(all_stratified_basin_sizes)
    # print(all_attractor_markov_matrix)
    # print(all_basin_sizes)


    # print(all_effective_degrees)    

    # data_dir = os.path.join(repo_dir, os.getenv('DATA_DIR'))

    # # Save to a file
    # file_name = f'effdeg_frozen_canBNexactdepth_{num_nodes}_{avg_degree}_{can_depth}_{num_networks}.pkl'
    # file_path = os.path.join(data_dir, file_name)
    # data = {'meta_data':f"This file contains simulation for {num_networks} networks with {num_nodes} nodes, {can_depth} canalizing depth and {avg_degree} average degree  with indegree distribution {indegree_distribution}.",
    #          "all_avg_effective_degree": all_effective_degrees, 'all_basin_coherence': all_basin_coherence, 'all_basin_sizes': all_basin_sizes,'all_attractors': all_attractors,
    #         'all_attractor_markov_matrix': all_attractor_markov_matrix,'all_frozen_core_coherence': all_frozen_core_coherence,'all_height_basin_coherence': all_height_basin_coherence}
    # with open(file_path, 'wb') as f:
    #     pickle.dump(data, f)
        
   