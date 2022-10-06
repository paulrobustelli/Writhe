import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm 
import math
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
import itertools
import mdtraj as md

def plt_avg_matrx(matrx, title="Writhe, Average", count=0): 
    plt.imshow(matrx,cmap='jet')
    plt.xlabel("segment 1", size = 20)
    plt.xticks()
    plt.ylabel("segment 2", size = 20)
    plt.title(title, size = 20)
    plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=20)

    
def plt_project_time(PC1, PC2, time=50, title= "writhe PCA"):    
    time= np.linspace(0,time,trj.n_frames)
    plt.figure()
    plt.scatter(PC1, PC2, marker='x', c=time)
    plt.xlabel('PC1',size=20)
    plt.ylabel('PC2',size=20) 
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.title(title, size=20)
    cbar = plt.colorbar()
    cbar.set_label('Time [$\mu$s]',size=20)
    
def plt_project_rg(PC1, PC2, title= "writhe PCA"): 
    # Projecting radius of gyration on principal components 
    rg=md.compute_rg(trj, masses=None)
    plt.figure()
    plt.scatter(PC1, PC2, marker='x', c=rg)
    plt.xlabel('PC1', size=20)
    plt.xticks(size=16)
    plt.ylabel('PC2', size=20) 
    plt.yticks(size=16)
    plt.title(title, size=20)
    cbar = plt.colorbar()
    cbar.set_label('Rg (nm)', size=20)


def free_energy(a, b, T, y0, ymax, x0, xmax):
    free_energy, xedges, yedges = np.histogram2d(a, b, 30, [[y0, ymax], [x0, xmax]], normed=True, weights=None)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.001987*T)*free_energy
    return free_energy, xedges, yedges

def plt_free_energy(PC1, PC2, col_map, title="Free Energy, Writhe", xmin=-3, xmax=3, ymin=-3, ymax=3):
    """
    plots the free energy 
    given the limits for the x-axis and y axis, this could be changed 
    """
    
    # set limits and overide with arguments 
    xmin = np.amin(PC1) if np.amin(PC1) < xmin else xmin
    xmax = np.amax(PC1) if np.amax(PC1) > xmax else xmax 
    ymin = np.amin(PC2) if np.amin(PC2) < ymin else ymin
    ymax = np.amax(PC2) if np.amax(PC2) > ymax else ymax 
    
    dG,xedges,yedges=free_energy(PC2, PC1, 300, ymin, ymax, xmin, xmax)
    fig = plt.figure(figsize=(8, 6))
    
    # set color 
    im = plt.imshow(dG, interpolation='gaussian', extent=[
                    yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap=col_map, aspect='auto')
    cbar_ticks = [i for i in range(1,9)]
    cb = plt.colorbar(ticks=cbar_ticks, format=('% .1f'), aspect=10)  # grab the Colorbar instance
    cb.set_label("kcal/mol", size=20) 

    plt.ylabel("PC2", size=20, labelpad=15)
    plt.xlabel("PC1", size=20, labelpad=15)
    plt.title(title, size = 20, loc="left")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axes(cb.ax)
    plt.clim(vmin=0.01, vmax=8.0)
    plt.yticks(size='16')
    plt.tight_layout()