#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:00:58 2025

@author: ckadelka

Generates Figure 3
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import boolforge

#biological BN parameters
max_degree = 16
max_N = 20
repository = 'expert-curated (ckadelka)'

#simulation parameters
EXACT = True #should only use True for bio models. To properly use False, we need to fix source nodes in the models
number_different_IC = 100

#number of null models per biological network: test with 2, run with 100
n_null_models = 100


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

import boolforge
bio_infos = []
for index in good_indices:
    print(index,bns[index].N)
    if EXACT:
        bio_infos.append(bns[index].get_attractors_and_robustness_measures_synchronous_exact())
    else:
        bio_infos.append(bns[index].get_attractors_and_robustness_measures_synchronous(number_different_IC=number_different_IC))

random_infos = []
rbns = []
for index in good_indices:
    random_infos.append([])
    rbns.append([])
    for _ in range(n_null_models):
        rbn = boolforge.random_null_model(bns[index],PRESERVE_BIAS=False,PRESERVE_CANALIZING_DEPTH=False)
        rbns[-1].append(rbn)
        if EXACT:
            random_infos[-1].append(rbn.get_attractors_and_robustness_measures_synchronous_exact())
        else:
            random_infos[-1].append(rbn.get_attractors_and_robustness_measures_synchronous(number_different_IC=number_different_IC))
        print(index,_)
        

attractors_bio = [info['Attractors'] for info in bio_infos]
indices_source_nodes_bio = [bns[index].get_source_nodes(False) for index in good_indices]
sizes = np.array([bns[index].N for index in good_indices])
n_source_nodes = np.array([len(bns[index].get_source_nodes(False)) for index in good_indices])

coherences_bio,entropies_bio = [], []
coherences_random, entropies_random = [], []
for i in range(len(bio_infos)):
    if n_source_nodes[i]==0:
        coherences_bio.append(bio_infos[i]['Coherence'])
        entropies_bio.append(boolforge.get_entropy_of_basin_size_distribution(bio_infos[i]['BasinSizes']))
        coherences_random.append([random_infos[i][j]['Coherence'] for j in range(n_null_models)])
        entropies_random.append([boolforge.get_entropy_of_basin_size_distribution(random_infos[i][j]['BasinSizes']) for j in range(n_null_models)])
    else:
        coherences_bio.append(bio_infos[i]['Coherence']*sizes[i]/(sizes[i]-n_source_nodes[i]))
        attractors = bio_infos[i]['Attractors']
        basin_sizes = bio_infos[i]['BasinSizes']
        indices_source_nodes = indices_source_nodes_bio[i]
        source_node_associations = np.array([boolforge.bin2dec(np.array(boolforge.dec2bin(attractor[0],sizes[i]))[indices_source_nodes]) for attractor in attractors])
        entropies_bio.append(np.mean([boolforge.get_entropy_of_basin_size_distribution(2**n_source_nodes[i] * basin_sizes[source_node_associations==k]) for k in range(2**n_source_nodes[i])]))
        
        coherences_random.append([random_infos[i][j]['Coherence']*sizes[i]/(sizes[i]-n_source_nodes[i]) for j in range(n_null_models)])
        entropies_random.append([])
        for j in range(n_null_models):
            attractors = random_infos[i][j]['Attractors']
            basin_sizes = random_infos[i][j]['BasinSizes']
            source_node_associations = np.array([boolforge.bin2dec(np.array(boolforge.dec2bin(attractor[0],sizes[i]))[indices_source_nodes]) for attractor in attractors])
            entropies_random[-1].append(np.mean([boolforge.get_entropy_of_basin_size_distribution(2**n_source_nodes[i] * basin_sizes[source_node_associations==k]) for k in range(2**n_source_nodes[i])]))
                        

coherences_bio = np.array(coherences_bio)
entropies_bio = np.array(entropies_bio)
coherences_random = np.array(coherences_random)
entropies_random = np.array(entropies_random)

suffix_save = 'bio_'+'_'.join(['%s%s' % (name,str(val)) for val,name in zip([max_N,max_degree,int(EXACT)],['maxN','maxdegree','EXACT'])])

f,ax = plt.subplots(figsize=(3,3))
im = ax.plot(coherences_bio,coherences_random.mean(1),marker='x',linestyle='None')
# for i in range(len(coherences_bio)):
#     ax.plot([coherences_bio[i],coherences_bio[i]],[np.percentile(coherences_random[i],5),np.percentile(coherences_random[i],95)],'r:')
[x1,x2] = ax.get_xlim()#f.colorbar(im)
[y1,y2] = ax.get_ylim()#f.colorbar(im)
ax.plot([0,1],[0,1],'k--')
ax.set_xlim([min(x1,y1),1.02])
ax.set_ylim([min(x1,y1),1.02])
#f.colorbar(im)
ax.set_xlabel('coherence biological network')
ax.set_ylabel('mean coherence null model')
ticks = [0.5,0.6,0.7,0.8,0.9,1]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.savefig('figure3_%s_nnull%i.pdf' % (suffix_save,n_null_models), bbox_inches = "tight")


print(stats.ttest_rel(coherences_bio, coherences_random.mean(1), alternative='greater'))



# order = np.argsort(coherences_bio)
# coherences_bio_sorted = coherences_bio[order]
# coherences_random_sorted = coherences_random[order]

# # Create figure
# fig, ax = plt.subplots(figsize=(10, 5))

# # Violin plots for random null models
# parts = ax.violinplot(coherences_random_sorted.T, showmeans=True, showmedians=False, widths=0.8)

# # Style violins
# for pc in parts['bodies']:
#     pc.set_facecolor('#d3d3d3')   # light gray
#     pc.set_edgecolor('black')
#     pc.set_alpha(0.7)
# parts['cbars'].set_color('black')
# parts['cmeans'].set_color('black')

# # Overlay biological coherence values
# ax.plot(np.arange(1, len(coherences_bio_sorted)+1),
#         coherences_bio_sorted, 
#         'o', color='crimson', label='Biological network')

# # Labels and aesthetics
# ax.set_xlabel('Biological network (sorted by coherence)')
# ax.set_ylabel('Coherence')
# ax.set_title('Coherence in Biological vs. Random Networks')
# ax.legend(loc='upper left', frameon=False)

# # Optional: adjust tick labels for clarity
# ax.set_xticks([1, 10, 20, 30, 40])
# ax.set_xlim(0.5, len(coherences_bio_sorted)+0.5)

# plt.tight_layout()
# plt.show()



# for min_size in range(8,13):
#     #min_size = 10
#     max_size = min_size
#     which = np.bitwise_and(sizes<=max_size,sizes>=min_size)
#     # f,ax = plt.subplots()
#     # ax.plot(boolforge.flatten(entropies_random[which]),boolforge.flatten(coherences_random[which]),'bo')
#     # ax.plot(entropies_bio[which],coherences_bio[which],'rx')

    
#     from matplotlib import cm
#     f,ax = plt.subplots()
#     colors = [cm.tab20(i*1./sum(which)) for i in range(sum(which))]
#     colors = [cm.tab10(i) for i in range(sum(which))]
#     for ii,i in enumerate(np.where(which)[0]):
#         ax.plot(entropies_random[i],coherences_random[i],'o',color=colors[ii],alpha=0.3)
#         ax.scatter(entropies_bio[i],coherences_bio[i],marker='x',color=colors[ii],s=200)


    
