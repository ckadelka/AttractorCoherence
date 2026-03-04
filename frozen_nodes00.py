#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:15:18 2026

@author: ckadelka
"""

import boolforge
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

nsim = 10000

N = 12
avg_degree = 5

hamming_weights = np.arange(1,2**(avg_degree-1),2)
abs_biases = 2 * np.abs((2**(avg_degree-1)-hamming_weights)/2**avg_degree)
std_biases = np.abs(hamming_weights/2**avg_degree) * (1-np.abs(hamming_weights/2**avg_degree))
all_attractors = []
all_basin_sizes = []
all_n_attractors = np.zeros((2**(avg_degree-2),nsim))
all_dist_attractors = np.zeros((2**(avg_degree-2),nsim))
all_n_frozen_nodes = np.zeros((2**(avg_degree-2),nsim))
for i,w in enumerate(hamming_weights):    
    
    all_attractors.append( [] )
    all_basin_sizes.append( [] )
    layerstructure_NCF = boolforge.utils.hamming_weight_to_ncf_layer_structure(avg_degree,w)
    for j in range(nsim):
        BN = boolforge.random_network(N=N,
                                      n=avg_degree,
                                      layer_structure = layerstructure_NCF, 
                                      strongly_connected = True)
        attr_info = BN.get_attractors_synchronous_exact()
        n_attractors = attr_info['NumberOfAttractors']
        attractors = attr_info["Attractors"]
        attractors_bin = [np.array([boolforge.dec2bin(state,N) for state in attractor]) 
                          for attractor in attractors]
        mean_state_per_attractor = np.array([np.mean(matrix,0) for matrix in attractors_bin])
        M = mean_state_per_attractor
        all_zero = np.all(M == 0, axis=0)
        all_one  = np.all(M == 1, axis=0)
        frozen_nodes = all_zero | all_one
        all_n_frozen_nodes[i,j] = frozen_nodes.sum()
        
        if n_attractors > 1:
            row_sums = M.sum(axis=1)
            dot_products = M @ M.T
            pairwise_dist = ( #note that the diagonale DOES NOT equal the within attractor distance
                row_sums[:, None]
                + row_sums[None, :]
                - 2 * dot_products
            )
            all_dist_attractors[i,j] = pairwise_dist[np.triu_indices(n_attractors, 1)].mean()
        else:
            all_dist_attractors[i,j] = np.nan

        all_attractors[-1].append(attractors)
        all_basin_sizes[-1].append(attr_info["BasinSizes"])
        all_n_attractors[i,j] = n_attractors

## plotting
PROPORTION = True
SHOW_MEAN = True
ABS_BIAS = False
if PROPORTION:
    modifier = N
    infix = 'proportion'
    infix2 = 'normalized '
else:
    modifier = 1
    infix = 'number'
    infix2 = ''
if ABS_BIAS:
    infix3 = 'absolute'
    x = abs_biases
else:
    infix3 = 'standardized'
    x = std_biases

#basic plot: frozen nodes
f,ax = plt.subplots()
ax.plot(x,all_n_frozen_nodes.mean(1)/modifier,'x--')
ax.set_xlabel(f'{infix3} bias')
ax.set_ylabel(f'average {infix} of frozen nodes')
ax.set_xlim([0,1 if ABS_BIAS else 0.25])
ax.spines[['right', 'top']].set_visible(False)
plt.savefig(f'analysis_{N}_{nsim}_frozen_nodes_total.pdf',bbox_inches='tight')

#basic dist between attractors nodes
f,ax = plt.subplots()
ax.plot(x,np.nanmean(all_dist_attractors,1)/modifier,'x--')
ax.set_xlabel(f'{infix3} bias')
ax.set_ylabel(f'average {infix2}distance between attractors\n(in networks with multiple attractors)')
ax.set_xlim([0,1 if ABS_BIAS else 0.25])
ax.spines[['right', 'top']].set_visible(False)
plt.savefig(f'analysis_{N}_{nsim}_avg_dist_attr_total.pdf',bbox_inches='tight')

#stratified by number of attractors
cmap = matplotlib.cm.tab10
n_attractors = [1,2,3,4]
which = [{} for _ in range(2**(avg_degree-2))]
for i in range(2**(avg_degree-2)):
    for n in n_attractors:
        which[i][n] = np.where(all_n_attractors[i]==n)[0]
f,ax = plt.subplots(figsize=(5,2.5))
for j,n in enumerate(n_attractors):
    y = [all_n_frozen_nodes[i,which[i][n]].mean()/modifier for i in range(2**(avg_degree-2))]
    ax.plot(x,y,'x--',label=str(n),color=cmap((n-1)))
if SHOW_MEAN:
    ax.plot(x,all_n_frozen_nodes.mean(1)/modifier,'ko:',label='mean')
ax.set_xlabel(f'{infix3} bias')
ax.set_ylabel(f'average {infix}\nof frozen nodes')
ax.set_xlim([0,1 if ABS_BIAS else 0.255])
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=False,loc='center',bbox_to_anchor=[0.5,1.08],ncol=5,title='number of attractors')
plt.savefig(f'analysis_{N}_{nsim}_frozen_nodes_stratified_bias_{infix3}.pdf',bbox_inches='tight')

n_attractors = [2,3,4]
f,ax = plt.subplots(figsize=(5,2.5))
for j,n in enumerate(n_attractors):
    y = [all_dist_attractors[i,which[i][n]].mean()/modifier for i in range(2**(avg_degree-2))]
    ax.plot(x,y,'x--',label=str(n),color=cmap((n-1)))
if SHOW_MEAN:
    ax.plot(x,np.nanmean(all_dist_attractors,1)/modifier,'ko:',label='mean')
ax.set_xlabel(f'{infix3} bias')
ax.set_ylabel(f'average {infix2}distance\nbetween attractors')
ax.set_xlim([0,1 if ABS_BIAS else 0.255])
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=False,loc='center',bbox_to_anchor=[0.5,1.08],ncol=4,title='number of attractors')
plt.savefig(f'analysis_{N}_{nsim}_avg_dist_attr_stratified_bias_{infix3}.pdf',bbox_inches='tight')


