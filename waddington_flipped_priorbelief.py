#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 23:50:15 2025

@author: ckadelka
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from matplotlib.collections import LineCollection
import warnings


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


# -------------------- Domain -------------------------------------------------
nx, nt = 501, 501
x = np.linspace(0.0, 1.0, nx)
t = np.linspace(0.0, 1.0, nt)
X, T = np.meshgrid(x, t)

# -------------------- Base Double-Well --------------------------------------
# Two stable valleys near 0.25 and 0.75 for all t>0
DW = (X - 0.25)**2 * (X - 0.75)**2

# Relief amplitude over time: shallower at ends, steepest mid-time
sigma_A = 0.2
A = 2.0 + 18.0 * np.exp(-((T - 0.5)/sigma_A)**2)   # keep >0 at ends to preserve two valleys

# -------------------- Ridges -------------------------------------------------
# Central ridge at x=0.5:
#  - baseline that persists to t=1,
#  - strong mid-time bump (highest near t=0.5),
#  - onset factor so it starts shortly after t=0.
sigma_r_x = 0.17           # spatial width of central ridge
sigma_r_t = 20           # width of the mid-time bump
R_base   = 1.2             # baseline height (persists to t=1)
R_peak   = 2.5          # extra mid-time height
tau_on   = 0.3            # onset time scale: smaller -> forms sooner after t=0

onset = 1.0 - np.exp(-T / tau_on)                      # ~0 at t=0, ->1 quickly
central_amp = onset * (R_base + R_peak * np.exp(-((T - 0.5)/sigma_r_t)**2))
CentralRidge = central_amp * np.exp(-((X - 0.5)/sigma_r_x)**2)

# Side ridges at x=0 and x=1 that exist for all t (barriers at the edges)
sigma_edge = 0.2
Edge_amp = 3.5                                              # constant over time
LeftRidge  = Edge_amp * np.exp(-(X / sigma_edge)**2)
RightRidge = Edge_amp * np.exp(-(((1.0 - X) / sigma_edge)**2))

# -------------------- Time tilt ---------------------------------------------
s = 2.5   # gentle downhill in t for visualization
V = A * DW + CentralRidge + LeftRidge + RightRidge - s * T

# -------------------- dV/dx (for trajectories) ------------------------------
def A_of_t(t):
    return 2.0 + 18.0 * np.exp(-((t - 1)/sigma_A)**2)

def central_amp_of_t(t):
    return (1.0 - np.exp(-t / tau_on)) * (R_base + R_peak * np.exp(-((t - 0.5)/sigma_r_t)**2))

def dVdx(x, t):
    # d/dx of quartic
    dDWdx = 2.0 * (x - 0.25) * (x - 0.75) * (2.0 * x - 1.0)

    # central ridge derivative
    Rx = np.exp(-((x - 0.5)/sigma_r_x)**2)
    dCentraldx = central_amp_of_t(t) * (-2.0) * (x - 0.5) / (sigma_r_x**2) * Rx

    # side ridges derivatives
    dLeftdx  = Edge_amp * (-2.0) * (x) / (sigma_edge**2) * np.exp(-(x / sigma_edge)**2)
    dRightdx = Edge_amp * (+2.0) * (1.0 - x) / (sigma_edge**2) * np.exp(-(((1.0 - x) / sigma_edge)**2))

    return A_of_t(t) * dDWdx + dCentraldx + dLeftdx + dRightdx
    # (time tilt -s*T has no x-derivative)

# -------------------- Integrate paths: dx/dt = -dV/dx -----------------------
def integrate_path(x0, dt=5e-4, t0=0):
    steps = int(1.0 / dt)
    xs = np.empty(steps + 1)
    ts = np.linspace(t0, 1.0, steps + 1)
    xs[0] = x0
    for k in range(steps):
        xs[k+1] = np.clip(xs[k] - dVdx(xs[k], ts[k]) * dt, 0.0, 1.0)
    return ts, xs

# Slightly asymmetric starts so they cleanly diverge and converge to 0.25 / 0.75
t_L, x_L = integrate_path(0.5 - 0.005, dt=1/501,t0=0.022)
t_R, x_R = integrate_path(0.5 + 0.005, dt=1/501,t0=0.022)

# -------------------- Plot (no rendering call) ------------------------------
fig, ax = plt.subplots(figsize=(4,3), dpi=140)
cf = ax.contourf(X, T, V[::-1,:], levels=60, cmap='terrain')
ax.contour(X, T, V[::-1,:], levels=12, colors='k', linewidths=0.35, alpha=0.5)

ax.plot(x_L, 1-t_L, lw=2.2, color='w',label='→ left fate (x→0.25)')
ax.plot(x_R, 1-t_R, lw=2.2, color='w',label='→ right fate (x→0.75)')

ax.scatter([0.5], [1-0.022], s=60, edgecolor='k', facecolor='white', zorder=5)
ax.scatter([x_L[-1]], [1-0.978], s=60, edgecolor='k', facecolor='white', zorder=5)
ax.scatter([x_R[-1]], [1-0.978], s=60, edgecolor='k', facecolor='white', zorder=5)

ax.text(0.5,1-0.08,'stem cell',va='center',ha='center')
ax.text(0.5,1-0.98,'differentiated\nmature cells',va='bottom',ha='center')

#ax.annotate("Pluripotent\n(t=0, x=0.5)", xy=(0.5, 0.0), xytext=(0.62, 0.06),
#            arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9)

#ax.axhline(0.5, color='k', lw=1.0, ls='--', alpha=0.6)
#ax.text(0.02, 0.52, "High central ridge (t≈0.5)", fontsize=9, alpha=0.9)

ax.arrow(-0.03,1,0,-1,clip_on=False,length_includes_head=True,head_width=0.03,head_length=0.03,color='k')
ax.arrow(0.02,-0.03,0.98,0,clip_on=False,length_includes_head=True,head_width=0.03,head_length=0.03,color='k')
ax.arrow(0.98,-0.03,-0.98,0,clip_on=False,length_includes_head=True,head_width=0.03,head_length=0.03,color='k')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Phenotypic space",labelpad=8);
ax.set_ylabel("Developmental time",labelpad=8)
#ax.set_title("Waddington Landscape: Persistent Central Ridge and Edge Barriers")

cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_ticks([cbar.vmin,cbar.vmax])
cbar.set_ticklabels(['high','low'])
cbar.set_label("Fitness",labelpad=-15)
#ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('waddington_flipped.pdf',bbox_inches='tight')



newV=   np.repeat(V[:,250], 501).reshape(501, 501)
resilience = np.zeros((501,501))
stability_gradient = np.zeros(501)
for i in range(501):
    j = 250
    while True:
        if j/501 > x_L[i]:
            j-=1
        else:
            break
    stability_gradient[i] = V[i,250] - V[i,j]


fig, ax = plt.subplots(figsize=(4.5,3), dpi=140)
cf = ax.contourf(X, T, V[::-1,:], levels=60, cmap='terrain')
ax.contour(X, T, V[::-1,:], levels=12, colors='k', linewidths=0.35, alpha=0.5)

#ax.plot(x_L, 1-t_L, lw=2.2, color='w',label='→ left fate (x→0.25)')
#ax.plot(x_R, 1-t_R, lw=2.2, color='w',label='→ right fate (x→0.75)')

cmap  = truncate_colormap(cm.Reds,maxval=0.4)
lines = colored_line(x_L, 1-t_L, stability_gradient, ax, linewidth=6, cmap=cmap)
lines = colored_line(x_R, 1-t_R, stability_gradient, ax, linewidth=6, cmap=cmap)


ax.scatter([0.5], [1-0.022], s=60, edgecolor='k', facecolor='white', zorder=5)
ax.scatter([x_L[250]], [0.5], s=60, edgecolor='k', facecolor='white', zorder=5)
ax.scatter([x_R[250]], [0.5], s=60, edgecolor='k', facecolor='white', zorder=5)
ax.scatter([x_L[-1]], [1-0.978], s=60, edgecolor='k', facecolor='white', zorder=5)
ax.scatter([x_R[-1]], [1-0.978], s=60, edgecolor='k', facecolor='white', zorder=5)

ax.text(0.5,1-0.05,'undifferentiated\nstem cells',va='top',ha='center')
ax.text(0.5,0.5,'stable\nprogenitor cells',va='center',ha='center')
ax.text(0.5,1-0.98,'very stable\nmature cells',va='bottom',ha='center')

#ax.annotate("Pluripotent\n(t=0, x=0.5)", xy=(0.5, 0.0), xytext=(0.62, 0.06),
#            arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9)

#ax.axhline(0.5, color='k', lw=1.0, ls='--', alpha=0.6)
#ax.text(0.02, 0.52, "High central ridge (t≈0.5)", fontsize=9, alpha=0.9)

ax.arrow(-0.03,1,0,-1,clip_on=False,length_includes_head=True,head_width=0.03,head_length=0.03,color='k')
ax.arrow(0.02,-0.03,0.98,0,clip_on=False,length_includes_head=True,head_width=0.03,head_length=0.03,color='k')
ax.arrow(0.98,-0.03,-0.98,0,clip_on=False,length_includes_head=True,head_width=0.03,head_length=0.03,color='k')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Phenotypic space",labelpad=8);
ax.set_ylabel("Developmental time",labelpad=8)
#ax.set_title("Waddington Landscape: Persistent Central Ridge and Edge Barriers")

cbar2 = fig.colorbar(lines,ax=ax,pad=0.05)  # add a color legend
cbar2.set_ticks([cbar2.vmin,cbar2.vmax])
cbar2.set_ticklabels(['low','high'])
cbar2.set_label("Stability to perturbations",labelpad=-15)

cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_ticks([cbar.vmin,cbar.vmax])
cbar.set_ticklabels(['high','low'])
cbar.set_label("Fitness",labelpad=-15)
#ax.legend(loc='lower right', frameon=True)



plt.tight_layout()
plt.savefig('waddington_flipped_priorbelief.pdf',bbox_inches='tight')