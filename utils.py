#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 23:03:45 2025

@author: ckadelka
"""

from collections import deque

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import boolforge

def paired_test(x,y,alternative='two-sided'):
    diff = x - y
    
    # Step 1: Normality
    p_norm = stats.shapiro(diff).pvalue
    
    if p_norm > 0.05:
        test = stats.ttest_rel
    else:
        # Step 2: Symmetry
        centered = diff - np.median(diff)
        p_sym = stats.ks_2samp(centered, -centered).pvalue
        if p_sym > 0.05:
            test = stats.wilcoxon
        else:
            test = sign_test  # fallback
    return test(x,y,alternative=alternative)
    
def sign_test(x, y, alternative='two-sided'):
    """
    Performs a paired sign test.
    
    Parameters
    ----------
    x, y : array-like
        Paired observations.
    alternative : {'two-sided', 'greater', 'less'}
        Defines the alternative hypothesis.

    Returns
    -------
    p_value : float
        The p-value for the sign test.
    """
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y

    # Remove zero differences (no information)
    nonzero = diff != 0
    n = np.sum(nonzero)
    n_pos = np.sum(diff[nonzero] > 0)

    # Perform exact binomial test
    result = stats.binomtest(n_pos, n, p=0.5, alternative=alternative)
    return result.pvalue

def running_mean(data, window_size):
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    
    running_means = []
    window = deque(maxlen=window_size)
    running_sum = 0
    for num in data:
        window.append(num)
        running_sum = sum(window)
        running_means.append(running_sum / len(window))
    
    return running_means

def rolling_mean(data, window_size,min_periods=1):
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    
    return pd.DataFrame(data).rolling(window=window_size,min_periods = min_periods,closed='both').mean()

def running_max(data):
    running_maxs = []
    maximum = data[0]
    for num in data:
        if num>maximum:
            maximum = num
        running_maxs.append(maximum)
    return running_maxs

def running_min(data):
    running_mins = []
    minimum = data[-1]
    for num in data[::-1]:
        if num<minimum:
            minimum = num
        running_mins.append(minimum)
    return running_mins[::-1]

def get_auc(x,y,n_points):
    assert len(x)==len(y),'x and y need to be of same length'
    indices = np.argsort(x)
    x_sorted = np.sort(x)
    y_sorted = np.array(y)[indices]
    x_equidistant = np.linspace(x_sorted[0],x_sorted[-1],n_points)
    y_equidistant = []
    for x_value in x_equidistant:
        distances_to_x_value = np.abs(x_sorted - x_value)
        y_equidistant.append( np.mean(y_sorted[distances_to_x_value == min(distances_to_x_value)]) )
    return np.mean(y_equidistant)


def correct_for_fixed_sources(basin_sizes, basin_coherence, attractor_coherence, number_variables, number_constants):
    #note: cannot correct fragility like this
    if number_constants==0:
        return np.array(basin_sizes), np.array(basin_coherence), np.array(attractor_coherence)
    else:
        correction_factor1 = 2**number_constants
        correction_factor2 = (number_variables+number_constants)/number_variables
        
        basin_sizes_fixed_sources = np.array(basin_sizes) * correction_factor1
        
        basin_coherence_fixed_sources = np.array(basin_coherence) * correction_factor2
        attractor_coherence_fixed_sources = np.array(attractor_coherence) * correction_factor2
        
        return basin_sizes_fixed_sources,basin_coherence_fixed_sources,attractor_coherence_fixed_sources

def run_random_BN_analysis(N,n,k,STRONGLY_CONNECTED,indegree_distribution,EXACT,n_networks,number_different_IC=1000):
    suffix_plot = ', '.join(['%s = %s' % (name,str(val)) for val,name in zip([N,n,k,int(STRONGLY_CONNECTED),indegree_distribution],['N','n','k','SC','distr'])])
    suffix_save = '_'.join(['%s%s' % (name,str(val)) for val,name in zip([N,n,k,int(STRONGLY_CONNECTED),indegree_distribution,int(EXACT),n_networks,number_different_IC],['N','n','k','SC','distr','EXACT','nnetworks','numberdifferentIC'])])

    #simulation
    basin_sizes = []
    n_attractors = []
    attractor_lengths = []
    basin_coherences = []
    basin_fragility = []
    attractor_coherences = []
    attractor_fragility = []
    for i in range(n_networks):
        bn = boolforge.random_network(N=N,n=n,k=k,STRONGLY_CONNECTED=STRONGLY_CONNECTED,indegree_distribution=indegree_distribution)
        if EXACT:
            info = bn.get_attractors_and_robustness_measures_synchronous_exact()
            number_attractors = info['ExactNumberOfAttractors']
            basins = info['BasinSizes']
            basin_coh = info['BasinCoherence']
            basin_frag = info['BasinFragility']
        else:
            info = bn.get_attractors_and_robustness_measures_synchronous_exact()
            number_attractors = info['LowerBoundOfNumberOfAttractors']
            basins = info['BasinSizesApproximation']
            basin_coh = info['BasinCoherenceApproximation']
            basin_frag = info['BasinFragilityApproximation']            
        attractor_lengths.append(list(map(len,info['Attractors'])))
        n_attractors.append([number_attractors] * number_attractors)
        basin_sizes.append(basins)
        basin_coherences.append(basin_coh)
        basin_fragility.append(basin_frag)
        attractor_coherences.append(info['AttractorCoherence'])
        attractor_fragility.append(info['AttractorFragility'])
    
    res = []
    for list_of_lists in [attractor_lengths,n_attractors,basin_sizes,basin_coherences,attractor_coherences,basin_fragility,attractor_fragility]:
        res.append(boolforge.flatten(list_of_lists))
    res=np.array(res)
    
    #turn fragility into 1-fragility
    res[-2:] = 1-res[-2:]
    
    names_res = ['length of attractor','number of attractors','basin size','basin coherence','attractor coherence','1-basin fragility','1-attractor fragility']
    assert res.shape[0] == len(names_res), "number of metric labels does not match the number of metrics"
    
    return res,names_res,suffix_plot,suffix_save


def compute_correlation_matrices(res):
    n_metrics = res.shape[0]
    spearman_mat = np.ones((n_metrics,n_metrics))
    for i in range(n_metrics):
        for j in range(n_metrics):
            spearman_mat[i,j] = stats.spearmanr(res[i],res[j])[0]
            spearman_mat[j,i] = spearman_mat[i,j]
    pearson_mat = np.ones((n_metrics,n_metrics))
    for i in range(n_metrics):
        for j in range(n_metrics):
            pearson_mat[i,j] = stats.pearsonr(res[i],res[j])[0]
            pearson_mat[j,i] = pearson_mat[i,j]
    return spearman_mat,pearson_mat

def plot_correlation_matrix(correlation_matrix,name,names_res,suffix_plot,suffix_save):
    n_metrics = len(names_res)
    f,ax = plt.subplots(figsize=(3.5,3.5))
    im = ax.imshow(correlation_matrix,origin='upper',vmin=-1,vmax=1,cmap=matplotlib.cm.RdBu)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(names_res)
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(names_res,rotation=90)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    divider = make_axes_locatable(ax)
    caxax = divider.append_axes("right", size='8%', pad=0.1)    
    f.colorbar(im,cax=caxax,label=name.capitalize()+' correlation coefficient')

    plt.savefig('figs/'+name+'_random_BN_%s.pdf' % suffix_save,bbox_inches='tight')

def plot_attractor_and_basin_coherence_vs_basin_size(res,names_res,suffix_plot,suffix_save):
    cmap = matplotlib.cm.tab10
    window_size = min(max(3,int(res.shape[1]/50)),30)

    f,ax = plt.subplots(figsize=(4,3))
    indices = np.argsort(res[2])
    for ii,i in enumerate([3,4]):
        ax.scatter(res[2],res[i],alpha=0.05,color=cmap(ii),edgecolors=None)
        ax.violinplot(res[i],vert=True,side='low',showextrema=False,positions=[0])
    for ii,i in enumerate([3,4]):
        ax.plot(res[2,indices],running_mean(res[i,indices],window_size),color=cmap(ii),label=names_res[i])
    ax.plot([0,1],[0,1],'k--')
    ax.violinplot(res[2],vert=False,side='low',showextrema=False,positions=[0])
    ax.set_xlabel(names_res[2])
    ax.set_ylabel('coherence')
    ax.set_title(suffix_plot)
    ax.legend(loc='center',bbox_to_anchor=[0.5,1.15],frameon=False,ncol=2)
    plt.savefig('figs/coherence_random_BN_%s.pdf' % suffix_save,bbox_inches='tight')



def plot_attractor_and_basin_coherence_vs_basin_size_v2(res,names_res,suffix_plot,suffix_save,window_size=None):
    cmap = matplotlib.cm.tab10
    if window_size is None:
        window_size = min(max(3,int(res.shape[1]/50)),30)

    f,ax = plt.subplots(3,2,figsize=(4,4),height_ratios = [1,4,4],width_ratios=[6,1],sharex='col')
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
    
    for ii,i in enumerate([3,4]):
        ax[ii+1,0].plot(res[2,indices],running_mean(res[i,indices],window_size),color='k',label=names_res[i])
    v = ax[0,0].violinplot(res[2],vert=False,side='high',showextrema=False,positions=[0])
    for pc in v['bodies']:
        pc.set_facecolor(cmap(2))
        pc.set_edgecolor(cmap(2))
    ax[2,0].set_xlabel(names_res[2])
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    


def plot_attractor_and_basin_fragility_vs_basin_size(res,names_res,suffix_plot,suffix_save):
    cmap = matplotlib.cm.tab10
    window_size = min(max(3,int(res.shape[1]/50)),30)

    f,ax = plt.subplots(figsize=(4,3))
    indices = np.argsort(res[2])
    for ii,i in enumerate([5,6]):
        ax.scatter(res[2],res[i],alpha=0.05,color=cmap(ii))
        ax.violinplot(res[i],vert=True,side='low',showextrema=False,positions=[0])
    for ii,i in enumerate([5,6]):
        ax.plot(res[2,indices],running_mean(res[i,indices],window_size),color=cmap(ii),label=names_res[i])
    #ax.plot([0,1],[0,1],'k--')
    ax.violinplot(res[2],vert=False,side='low',showextrema=False,positions=[0])
    ax.set_xlabel(names_res[2])
    ax.set_ylabel(r'1 $-$ fragility')
    ax.set_title(suffix_plot)
    ax.legend(loc='center',bbox_to_anchor=[0.5,1.15],frameon=False,ncol=2)
    plt.savefig('figs/fragility_random_BN_%s.pdf' % suffix_save,bbox_inches='tight')

def plot_basin_vs_attractor_coherence(res,names_res,suffix_plot,suffix_save):
    index_x,index_y,index_color = 3,4,2

    f,ax = plt.subplots()
    cm = matplotlib.cm.viridis
    im = ax.scatter(res[index_x],res[index_y],c=res[index_color],cmap=cm)
    ax.set_xlabel(names_res[index_x])
    ax.set_ylabel(names_res[index_y])
    f.colorbar(im,label=names_res[index_color])
    ax.spines[['right', 'top']].set_visible(False)
    #ax.text(0,1,r'$\rho$ =' + str(np.round(stats.pearsonr(res[index_x],res[index_y])[0],2)),va='center',ha='left')
    ax.plot([0,1],[0,1],'k--')
    ax.set_title(suffix_plot)
    plt.savefig('figs/basin_vs_attractor_coherence_%s.pdf' % suffix_save,bbox_inches='tight')

