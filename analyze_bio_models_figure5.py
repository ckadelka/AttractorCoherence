#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:00:58 2025

@author: ckadelka

Generates Figure 3
"""



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

import boolforge
import utils


#biological BN parameters
max_degree = 16
max_N = 64
repository = 'expert-curated (ckadelka)'

#simulation parameters
EXACT = False #should only use True for bio models. To properly use False, we need to fix source nodes in the models
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

suffix_plot = ', '.join(['%s = %s' % (name,str(val)) for val,name in zip([max_N,max_degree,int(EXACT)],['maxN','maxdegree','EXACT'])])
suffix_save = 'bio_'+'_'.join(['%s%s' % (name,str(val)) for val,name in zip([max_N,max_degree,int(EXACT)],['maxN','maxdegree','EXACT'])])

spearman_mat,pearson_mat = utils.compute_correlation_matrices(res[:-2])
utils.plot_correlation_matrix(spearman_mat,'spearman',names_res[:-2],suffix_plot,suffix_save)
utils.plot_correlation_matrix(pearson_mat,'pearson',names_res[:-2],suffix_plot,suffix_save)
utils.plot_attractor_and_basin_coherence_vs_basin_size(res,names_res,suffix_plot,suffix_save)
utils.plot_attractor_and_basin_coherence_vs_basin_size_nice(res,names_res,suffix_plot,suffix_save)
utils.plot_attractor_and_basin_fragility_vs_basin_size(res,names_res,suffix_plot,suffix_save)



n_points = 1000
auc_attractor_coherencess = utils.get_auc(res[2],res[4],n_points)
auc_basin_coherencess = utils.get_auc(res[2],res[3],n_points)

mean_function = utils.rolling_mean
cmap = matplotlib.cm.tab10
window_size = 50#min(max(3,int(res.shape[1]/50)),30)
SEPARATE = True
threshold_basin_size = 0.05

if SEPARATE == False:
    threshold_basin_size = 2

f,ax = plt.subplots(4,2,figsize=(4,6),height_ratios = [1,4,4,4],width_ratios=[6,1],sharex='col')
indices = np.argsort(res[2])
for ii,i in enumerate([3,4]):
    v = ax[ii+1,1].violinplot(res[i],vert=True,side='high',showextrema=False,positions=[0])
    for pc in v['bodies']:
        pc.set_facecolor(cmap(ii))
        pc.set_edgecolor(cmap(ii))
    ax[ii+1,0].scatter(res[2],res[i],alpha=0.05,color=cmap(ii),edgecolors=None)
    ax[ii+1,0].plot([0,1],[0,1],'k--')
    ax[ii+1,0].set_ylabel(names_res[i])
    ax[ii+1,1].set_axis_off()
    ax[ii+1,0].spines[['right', 'top']].set_visible(False)
for jj in range(2):
    ax[0,jj].set_axis_off()

for ii,(i,label) in enumerate(zip([3,4],['BC','AC'])):
    ax[ii+1,0].plot(res[2,indices],mean_function(res[i,indices],window_size),color='k',label=names_res[i])
    ax[ii+1,0].text(0.83,0.2,f'AUC:{label}=',ha='right',va='center',color='k')
    ax[ii+1,0].text(1.0,0.2,str(round(utils.get_auc(res[2],res[i],n_points),3)),ha='right',va='center',color='k')
     
v = ax[0,0].violinplot(res[2],vert=False,side='high',showextrema=False,positions=[0])
for pc in v['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')


for iii in range(int(SEPARATE)+1):
    if iii==0:
        which = res[2] < threshold_basin_size
    else:
        which = res[2] >= threshold_basin_size
    
    ax[3,0].scatter(res[2][which],(res[3] - res[4])[which],alpha=0.05,color=cmap(2+2*int(iii)))
    ax[3,0].plot([0,1],[0,0],'k--')
    v = ax[3,1].violinplot(res[3][which] - res[4][which],vert=True,side='high',showextrema=False,positions=[0])
    for pc in v['bodies']:
        pc.set_facecolor(cmap(2+2*int(iii)))
        pc.set_edgecolor(cmap(2+2*int(iii)))
ax[3,0].set_ylabel(r'basin $-$ attractor'+'\ncoherence')
ax[3,0].plot(res[2][indices],mean_function((res[3] - res[4])[indices],window_size),color='k')
ax[3,0].text(0.83,-0.6,r'$\Delta$AUC=',ha='right',va='center',color='k')
ax[3,0].text(1,-0.6,str(round(utils.get_auc(res[2],res[3]-res[4],n_points),3)),ha='right',va='center',color='k')
 
ax[3,0].spines[['right', 'top']].set_visible(False)
ax[3,1].set_axis_off()
    
ax[3,0].set_xlabel(names_res[2])
plt.subplots_adjust(wspace=0.01, hspace=0.1)
plt.savefig('figs/figure5_%s.pdf' % suffix_save,bbox_inches='tight')


    
