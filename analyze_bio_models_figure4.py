#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:00:58 2025

@author: ckadelka

Generates Figure 3
"""



import numpy as np
import random
import boolforge

import utils
import scipy.stats as stats

#biological BN parameters
max_degree = 16
max_N = 20
repository = 'expert-curated (ckadelka)'

#simulation parameters
EXACT = True #should only use True for bio models. To properly use False, we need to fix source nodes in the models
number_different_IC = 100
max_n_fixed_source_networks = 16

#extract networks and remove non-essential regulations
bns,urls_loaded,urls_not_loaded = boolforge.get_bio_models_from_repository(repository)
for index,bn in enumerate(bns):
    bn.simplify_functions()


n_networks = len(bns)
good_indices = []
for i,bn in enumerate(bns):
    if bn.N<=max_N and max(bn.indegrees)<=max_degree and min(bn.indegrees)>0:
        good_indices.append(i)
good_indices = np.array(good_indices)

n_source_nodes = np.array([len(bns[i].get_source_nodes(False)) for i in good_indices])

#Generate up to {max_n_fixed_source_networks} fixed-source networks per biological network
bns_to_analyze = []
for ii,index in enumerate(good_indices):
    if n_source_nodes[ii] == 0:
        bns_to_analyze.append([bns[index]])
    else:
        bns_to_analyze.append([])
        if 2**n_source_nodes[ii] <= max_n_fixed_source_networks:
            decimal_choices_for_constants_binary_strings = list(range(2**n_source_nodes[ii]))
        else:
            decimal_choices_for_constants_binary_strings = np.array(random.sample(range(2**n_source_nodes[ii]),max_n_fixed_source_networks))
        for decimal_choice_for_constants in decimal_choices_for_constants_binary_strings:
            values_source_nodes = boolforge.dec2bin(decimal_choice_for_constants,n_source_nodes[ii])
            bn_fixed_source_nodes = bns[index].get_network_with_fixed_source_nodes(values_source_nodes)
            bns_to_analyze[-1].append(bn_fixed_source_nodes)

#Analyze all fixed-source biological networks
bio_infos = []
for ii,index in enumerate(good_indices):
    bio_infos.append([])
    for jj,bn in enumerate(bns_to_analyze[ii]):
        print(index,jj,bn.N)
        if EXACT:
            bio_infos[-1].append(bn.get_attractors_and_robustness_measures_synchronous_exact())
        else:
            bio_infos[-1].append(bn.get_attractors_and_robustness_measures_synchronous(number_different_IC=number_different_IC))

#Store results
basin_sizes = []
n_attractors = []
attractor_lengths = []
basin_coherences = []
basin_fragility = []
attractor_coherences = []
attractor_fragility = []   
model_ids = []
model_repeats = []
number_frozen_nodes = []
for ii,index in enumerate(good_indices):
    for jj in range(min(max_n_fixed_source_networks,2**n_source_nodes[ii])):
        attractors = bio_infos[ii][jj]['Attractors']
        number_attractors = len(attractors)
        attractor_lengths.append(list(map(len,attractors)))
        n_attractors.append([number_attractors] * number_attractors)
        basin_sizes.append(bio_infos[ii][jj]['BasinSizes' if EXACT else 'BasinSizesApproximation'])
        basin_coherences.append(bio_infos[ii][jj]['BasinCoherence' if EXACT else 'BasinCoherenceApproximation'])
        basin_fragility.append(bio_infos[ii][jj]['BasinFragility' if EXACT else 'BasinFragilityApproximation'])
        attractor_coherences.append(bio_infos[ii][jj]['AttractorCoherence'])
        attractor_fragility.append(bio_infos[ii][jj]['AttractorFragility'])
        model_ids.append([index] * number_attractors)
        model_repeats.append([jj] * number_attractors)

res = []
for list_of_lists in [attractor_lengths,n_attractors,basin_sizes,basin_coherences,
                      attractor_coherences,basin_fragility,attractor_fragility,
                      model_ids,model_repeats]:
    res.append(boolforge.flatten(list_of_lists))
res=np.array(res)

#turn fragility into 1-fragility
res[5:7] = 1-res[5:7]

names_res = ['length of attractor','number of attractors','basin size','basin coherence','attractor coherence','1 - basin fragility','1 - attractor fragility','model ID','model repeat']

#suffix_plot = ', '.join(['%s = %s' % (name,str(val)) for val,name in zip([max_N,max_degree,int(EXACT)],['maxN','maxdegree','EXACT'])])
suffix_plot = '' #', '.join(['%s = %s' % (name,str(val)) for val,name in zip([max_N,max_degree,int(EXACT)],['maxN','maxdegree','EXACT'])])
suffix_save = 'bio_'+'_'.join(['%s%s' % (name,str(val)) for val,name in zip([max_N,max_degree,int(EXACT)],['maxN','maxdegree','EXACT'])])

# spearman_mat,pearson_mat = utils.compute_correlation_matrices(res[:-2])
# utils.plot_correlation_matrix(spearman_mat,'spearman',names_res[:-2],suffix_plot,suffix_save)
# utils.plot_correlation_matrix(pearson_mat,'pearson',names_res[:-2],suffix_plot,suffix_save)
# utils.plot_attractor_and_basin_coherence_vs_basin_size(res,names_res,suffix_plot,suffix_save)
# utils.plot_attractor_and_basin_coherence_vs_basin_size_nice(res,names_res,suffix_plot,suffix_save)
# utils.plot_attractor_and_basin_fragility_vs_basin_size(res,names_res,suffix_plot,suffix_save)


# n_points = 1000
# auc_attractor_coherencess = utils.get_auc(res[2],res[4],n_points)
# auc_basin_coherencess = utils.get_auc(res[2],res[3],n_points)

utils.plot_basin_vs_attractor_coherence(res,names_res,suffix_plot,suffix_save)

print(stats.shapiro(res[3]-res[4]))
if stats.shapiro(res[3]-res[4])[1] > 0.05:
    print(stats.ttest_rel(res[3],res[4]))
else:
    print(stats.wilcoxon(res[3],res[4]))