"""
Computes writhe for a given trajectory and 
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm 
import math
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
import itertools
import mdtraj as md
from sklearn.cluster import KMeans
import pandas as pd 

# load in trajectory and pdb, can make this into an argument in the future 
# use the command line 
# argument order is trajectory, pdb, outdir, w_cluster, c_cluster 
print("\nrunning this file:", sys.argv[0])
trajectory = "./" +sys.argv[1]
pdb = "./" + sys.argv[2]
outdir = "./" + sys.argv[3] + "/"
w_clusters =int(sys.argv[4])
c_clusters = int(sys.argv[5])

# trajectory = "./Ab40-a03ws.dcd"
# pdb = "./Ab40.1.pdb"
# outdir= './outdir_a03ws/'
sys = os.path.splitext(os.path.basename(pdb))[0]

# can possibly automate this to search for pdb, and add that root, but would need to stop adding dots in pdb/dcd files
if not os.path.exists(outdir):
    os.makedirs(outdir)

trj = md.load(trajectory, top=pdb)

def writhe_matrix(s1,s2):
    v13 = s2[:,0] - s1[:,0]
    v23 = s2[:,0] - s1[:,1]
    v24 = s2[:,1] - s1[:,1]
    v14 = s2[:,1] - s1[:,0]
    v = [v13,v23,v24,v14,v13,v23]
    e = []
    l13 = np.linalg.norm(v13,axis=1)
    l23 = np.linalg.norm(v23,axis=1)
    l24 = np.linalg.norm(v24,axis=1)
    l14 = np.linalg.norm(v14,axis=1)
    ls = [l13,l23,l24,l14]   
    for l in ls:
        if np.sum(l) == 0.0:
            return 0
    e13 = v13/l13[:,None]
    e23 = v23/l23[:,None]
    e24 = v24/l24[:,None]
    e14 = v14/l14[:,None]
    e = [e13,e23,e24,e14,e13,e23] 
        #compute the angles
    s = 0
    for i in range(1,len(e)-1):
        a = np.asarray(e[i-1])
        b = np.asarray(e[i])
        c = np.asarray(e[i+1])
        #a =e[i-1]
        #b =e[i]
        #c = e[i+1]
        v1=a-b
        v3=c-b
        #a=np.dot(v1,v3)
        a=(v1*v3).sum(axis=1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v3,axis=1))
        #a=np.dot(v1,v3)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v3,axis=1))
        theta=np.arccos(np.clip(a, -1.0, 1.0))  
        s = s +theta
    w = np.sign(s)*2*np.pi - s
    return w

def wr_tot(CAlabel, CApos):
    """
    computes the top half of the total writhe matrix with values of dimension n_frames
    input: CAlabel, CApos
    returns: dictionary of dictionary, where each key indicates segment pair
    """
    dim = int((len(CAlabel) - 1)*(len(CAlabel) - 2)/2)
    model_input = np.empty((dim, trj.n_frames))
    wr_total = {}
    
    count = 0
    for j in range(0,len(CAlabel)-1):
        wr_total[j]={}
        v1=CApos[:,j]; v2=CApos[:,j+1];
        s1=np.empty((len(CApos),2,3))
        s1[:,0]=v1
        s1[:,1]=v2
        
        for k in range(j+1,len(CAlabel)-1):
            v3=CApos[:,k]; v4=CApos[:,k+1];
            s2=np.empty((len(CApos),2,3))
            s2[:,0]=v3
            s2[:,1]=v4
            wr_total[j][k]= writhe_matrix(s1,s2) 
            
            if k > j+1: 
                model_input[count] = wr_total[j][k]
                count += 1 

    model_input = np.transpose(model_input)
    return wr_total, model_input 

def writhe_by_res(wr_total, CApos):
    """
    computes the writhe by residue, total contribution for each frame
    ouput: dictionary, writhe by residue 
    """
    
    writhe_by_res={}
    wtot=np.zeros(len(CApos[:,0]))
    
    for i in wr_total:
        w=np.zeros(len(CApos[:,0]))
        for j in wr_total[i]:
            w+=wr_total[i][j] 
        writhe_by_res[i]=w
        wtot+=w
    return writhe_by_res

def writhe_avg(wr_total, CAlabel): 
    """
    computes the writhe average 
    output: numpy array of averages 
    """
    writhe_matrx= np.zeros((len(CAlabel)-1,len(CAlabel)-1))
    for j in wr_total:
        for k in wr_total[j]:
            writhe_matrx[j,k]= np.average(wr_total[j][k])
            writhe_matrx[k,j]= writhe_matrx[j,k]
    return writhe_matrx

def gen_CA(CAatoms):
    """
    Generate an array of alpha carbons  
    input: CAatoms 
    output: numpy array of CA labels 
    """
    
    CAlabel=[]    
    for i in range(0,len(CAatoms)):
        CAlabel.append(trj.topology.atom(CAatoms[i]).residue.resSeq)
    CAlabel =np.array(CAlabel).astype(int)

    return CAlabel

def W_ij(trj):
    """
    compute writhe 
    returns: total writhe, writhe by residue, average writhe matrix, and the writhe input 
    """
    CAatoms= trj.topology.select('name CA')
    CApos= trj.xyz[:,CAatoms]
    CAlabel= gen_CA(CAatoms)
    wr_total, model_input= wr_tot(CAlabel, CApos)
    wr_by_res= writhe_by_res(wr_total, CApos)
    writhe_matrx= writhe_avg(wr_total, CAlabel)
    return CAlabel, wr_total, wr_by_res, writhe_matrx, model_input, CAatoms

def make_contact_distance_map(res_num):
    """
    All elements to create a contact map 
    returns: (np.array) model input for PCA, contact average, and total contacts 
    """
    contacts = np.zeros(shape=(res_num,res_num,trj.n_frames))
    contact_avg = np.zeros(shape=(res_num,res_num))
    
    dim = int((res_num)*(res_num - 1)/2)
    model_input = np.zeros(shape=(dim, trj.n_frames))
    
    count = 0 
    for i in range(0, res_num):
        for j in range(i+1, res_num):
            contacts[i][j] = md.compute_contacts(trj,[[i,j]])[0].reshape(trj.n_frames,)
            contacts[j][i] = contacts[i][j]
             
            contact_avg[i][j] = np.average(contacts[i][j])
            contact_avg[j][i] = contact_avg[i][j] 
            
            model_input[count] = contacts[i][j]
            count += 1
    
    # transpose
    model_input = np.transpose(model_input)
    return model_input, contacts, contact_avg

def make_contact_map(res_num=20):
    """
    All elements to create a contact map 
    returns: (np.array) model input for PCA, contact average, and total contacts 
    """
    contacts = np.zeros(shape=(res_num,res_num,trj.n_frames))
    contact_avg = np.zeros(shape=(res_num,res_num))
    
    #dim = int((res_num)*(res_num - 1)/2)
    model_in = np.zeros(shape=((res_num*res_num), trj.n_frames))
    
    count = 0 
    for i in range(0, res_num):
        for j in range(0, res_num):
            if i == j: 
                contacts[i][j] = 0
            else: 
                contacts[i][j] = np.where(md.compute_contacts(trj,[[i,j]])[0] < 0.5, 1, 0).reshape(trj.n_frames)
                contact_avg[i][j] = np.average(contacts[i][j])

            model_in[count] = contacts[i][j]
            count += 1
    
    # transpose
    model_in = np.transpose(model_in)
    return model_in, contacts, contact_avg


def plot_avg_matrx(matrx, title="Writhe, Average", count=0): 
    plt.figure(figsize=(10, 8))
    plt.imshow(matrx,cmap='jet')

    if title[0] == "W":
        plt.xlabel("segment 1", size = 20)
        plt.ylabel("segment 2", size = 20)
    else:
        plt.xlabel("Residue", size = 20)
        plt.ylabel("Residue", size = 20)
    plt.title(title, size = 20)
    plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(outdir+sys+title[0]+str(count)+".png", dpi=200)
    plt.clf()
    return

def reconstruct_matrx(pca, n_components=2, num_res=20): 
    reconstructed_matrices = np.empty((num_res-1,num_res-1,n_components))
    rev = np.flip(np.arange(1, num_res-2))
    
    for n in range(n_components): 
        # grab eigenvalue 
        eigh = pca.components_[n,:]
        matrx = np.zeros((num_res-1,num_res-1))
        
        # reconstruct the matrix 
        count = 0 
        for k in range(2, num_res-1): # starts at 2 because of hack
            val = eigh[:rev[count]]
            eigh = eigh[rev[count]:]
            matrx[count][k:] = val
            #matrx[rev[count]+1][:rev[count]] = np.flip(val)
            count += 1 
        
        
        # do other side of matrix 
        for i in range(num_res-1): 
            for j in range(i+1, num_res-1):
                matrx[j][i] = matrx[i][j]
        
        reconstructed_matrices[:,:,n] = matrx
        
    return reconstructed_matrices

def pca(model_input):
    """
    principal component analysis on model input, deconstructs the matrices 
    """
    model = PCA(n_components=2)
    model.fit(model_input)
    
    # eigenvector decomposition
    # only need to reconstruct the matrix for writhe, since top half was taken
    if model_input.shape[1] < (trj.n_residues-1)*(trj.n_residues-1): 
        reconstructed_matrices = reconstruct_matrx(model, 2, trj.n_residues)
        matrx_e1 = reconstructed_matrices[:,:,0]
        matrx_e2 = reconstructed_matrices[:,:,1]
        plot_avg_matrx(matrx_e1, title="Writhe, E1 Weights", count="e1")
        plot_avg_matrx(matrx_e2, title="Writhe, E2 Weights", count="e2")
    
    # reshape components to roll back for contacts 
    else: 
        matrx_e1 = np.reshape(model.components_[0,:], (trj.n_residues, trj.n_residues))
        matrx_e2 = np.reshape(model.components_[1,:], (trj.n_residues, trj.n_residues))
        plot_avg_matrx(matrx_e1, title="Contacts, E1 Weights", count="e1")
        plot_avg_matrx(matrx_e2, title="Contacts, E2 Weights", count="e2")

    # transform data, dot product of covariance matrix and weights from SVD 
    reduced = model.fit_transform(model_input) 
    PC1=reduced[:, 0]
    PC2=reduced[:,1]
    return PC1,PC2
    
def plot_project_time(PC1, PC2, time=50, title= "writhe PCA"):  
    time= np.linspace(0,time,trj.n_frames)
    plt.figure(figsize=(12, 10))
    plt.scatter(PC1, PC2, marker='x', c=time)
    plt.xlabel('PC1',size=20)
    plt.ylabel('PC2',size=20) 
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.title(title, size=20)
    cbar = plt.colorbar()
    cbar.set_label('Time [$\mu$s]',size=20)
    plt.savefig(outdir+sys+"_project_time_"+title[0]+".png", dpi=200) # could change this later? 
    plt.clf()
    return
    
def plot_project_rg(PC1, PC2, title= "writhe PCA"): 
    # Projecting radius of gyration on principal components 
    plt.figure(figsize=(12, 10))
    rg=md.compute_rg(trj, masses=None)
    plt.scatter(PC1, PC2, marker='x', c=rg)
    plt.xlabel('PC1', size=20)
    plt.xticks(size=16)
    plt.ylabel('PC2', size=20) 
    plt.yticks(size=16)
    plt.title(title, size=20)
    cbar = plt.colorbar()
    cbar.set_label('Rg (nm)', size=20)
    plt.savefig(outdir+sys+"_project_rg_"+title[0]+".png", dpi=200)
    plt.clf() 
    return 

def free_energy(a, b, T, y0, ymax, x0, xmax):
    free_energy, xedges, yedges = np.histogram2d(a, b, 30, [[y0, ymax], [x0, xmax]], normed=True, weights=None)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.001987*T)*free_energy
    return free_energy, xedges, yedges

def plot_free_energy(PC1, PC2, col_map, centers=None, scat=False, title="Free Energy, Writhe", xmin=-3, xmax=3, ymin=-3, ymax=3):
    """
    plots the free energy 
    given the limits for the x-axis and y axis, this could be changed 
    """
    
    # set limits and overide with arguments 
    xmin = (np.amin(PC1) - 1) if (np.amin(PC1) - 1) < xmin else xmin
    xmax = (np.amax(PC1) + 1) if (np.amax(PC1) + 1) > xmax else xmax 
    ymin = (np.amin(PC2) - 1) if (np.amin(PC2) - 1) < ymin else ymin
    ymax = (np.amax(PC2) + 1) if (np.amax(PC2) + 1) > ymax else ymax 
    
    dG,xedges,yedges=free_energy(PC2, PC1, 300, ymin, ymax, xmin, xmax)
    fig = plt.figure(figsize=(10, 8))
    
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
    
    if scat: 
        # insert centers 
        scat = np.split(centers, 2, axis=1)
        plt.scatter(scat[0], scat[1], c="white", marker = "o", s=100)
    
    plt.axes(cb.ax)
    plt.clim(vmin=0.01, vmax=8.0)
    plt.yticks(size='16')
    plt.tight_layout()
    plt.savefig(outdir+sys+"free_energy_"+title[0]+".png", dpi=200)
    plt.close(fig)
    return

def locate_basin(PC1,PC2, box): 
    xmin,xmax,ymin,ymax = box 
    basin = np.where((xmin < WPC1) & (PC1< xmax))
    basin = np.where((PC2 > ymin)& (PC2 < ymax))
    basin_ind = np.intersect1d(f_pc1_basin1, f_pc2_basin1)
    return basin_ind

def writhe_avg_from_ind(wr_total, CAlabel, np_ind): 
    """
    computes the writhe average from indices 
    output: numpy array of averages 
    """
    writhe_matrx= np.zeros((len(CAlabel)-1,len(CAlabel)-1))
    np_empty = np.empty(np_ind.size,)
    
    for j in wr_total:
        for k in wr_total[j]:
            for n,i in enumerate(np_ind): 
                if type(wr_total[j][k]) == int: 
                    np_empty[n] = 0
                else:  
                    np_empty[n] = wr_total[j][k][i] # a faster way is to make a mask and remove zero values
            writhe_matrx[j,k]= np.average(np_empty)
            writhe_matrx[k,j]= writhe_matrx[j,k]
    return writhe_matrx

def contact_map_avg(con, res_num, np_ind):
    """
    computes the contant distance average from indices 
    input: contact distances dictionary, number of residues, a numpy array with the indices
    output: contact_avg 
    """
    contact_avg = np.zeros(shape=(res_num,res_num))
    np_empty = np.empty(np_ind.size,)
    for i in range(res_num):
        for j,nparr in enumerate(con[i]):
            for num, k in enumerate(np_ind): 
                np_empty[num] = nparr[k] # contacts[i][j][k]
            contact_avg[i,j] = np.average(np_empty)
    return contact_avg

def get_centroid(frames): 
    step = frames.n_frames/30 # number of steps for 30 structures 
    stride = np.arange(0, frames.n_frames, int(frames.n_frames/1000)) 
    traj_basin = frames.slice(stride)
    atom_indices = [a.index for a in traj_basin.topology.atoms if a.element.symbol != 'H']
    distances = np.empty((traj_basin.n_frames, traj_basin.n_frames))
    
    for i in range(traj_basin.n_frames):
        distances[i] = md.rmsd(traj_basin, traj_basin, i, atom_indices=atom_indices)

    beta = 1
    index = np.exp(-beta*distances / distances.std()).sum(axis=1).argmax()

    return int(index), int(step)

def kmeans_cluster(PC1, PC2, clusters=3, title = "Writhe, Kmeans Clustering"): 
    concat = np.column_stack((PC1, PC2))
    X = concat
    kmeans = KMeans(n_clusters=clusters).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # make a plot and save it 
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    fig = plt.figure(figsize=(10,8))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask] #  & core_samples_mask for dbscan
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.ylabel("PC2", size=20, labelpad=15)
    plt.xlabel("PC1", size=20, labelpad=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, size = 20, loc="center")
    plt.savefig(outdir+sys+"_kmeans_clustering_"+title[0]+".png",dpi=200)
    plt.close(fig)

    # save the trajectories for each basin and calculate the centroid 
    for i in unique_labels: 
        basin = np.where(labels == i)[0]
        basin_trj = trj.slice(basin)
        
        if title[0] == "W":
            print("this is writhe basin", i, " ", basin_trj.n_frames)
            basin_trj.save_dcd(outdir + sys + "kmeans_writhe_basin_" + str(i) + "_trj.dcd")
            # save info for vmd 
            file_vmd_info = outdir+sys+"kmeans_writhe_centroid_step.txt"

        else: 
            basin_trj.save_dcd(outdir + sys + "kmeans_contacts_basin_"+ str(i) + "_trj.dcd")
            print("this is contacts basin", i, " ", basin_trj.n_frames)
            # save info for vmd 
            file_vmd_info = outdir+sys+"kmeans_contacts_centroid_step.txt"
        
        # save centroid 
        index, step = get_centroid(basin_trj)
        centroid_step = np.asarray((index, step), dtype=np.int32)
        
        with open(file_vmd_info, "ab") as f:
            f.write(b"\n")
            np.savetxt(f, centroid_step)
        f.close()

    return centers

def main():  
    CAlabel, wr_total, wr_by_res, writhe_matrx, w_model_input, CAatoms = W_ij(trj)
    trj.superpose(trj, 0)
    c_model_input, contacts, contact_avg = make_contact_map(trj.n_residues) 
    
    # create a new cmap to make background white 
    cdict = cm.get_cmap("nipy_spectral")._segmentdata
    cdict["red"][-1]=(1,1,1)
    cdict["green"][-1]=(1,1,1)
    cdict["blue"][-1]=(1,1,1)
    n_cmap = colors.LinearSegmentedColormap("new_cmap", cdict)
    
    # PCA 
    WPC1, WPC2 = pca(w_model_input)
    plot_project_time(WPC1, WPC2, time=30, title= 'Writhe PCA: Ab40, time') # title matters because 1st letter is in output png
    plot_project_rg(WPC1, WPC2, title= "Writhe PCA: Ab40, Rg")
    wcenter = kmeans_cluster(WPC1, WPC2, clusters=w_clusters, title = "Writhe, Kmeans Clustering")
    plot_free_energy(WPC1, WPC2, n_cmap, centers=wcenter, scat=True, title= "Writhe PCA: Ab40, Free Energy,", xmin=-2, xmax=3, ymin=-2, ymax=3)
    
    CPC1, CPC2 = pca(c_model_input)
    plot_project_time(CPC1, CPC2, time=30, title= 'Contacts PCA: Ab40, time')
    plot_project_rg(CPC1, CPC2, title= "Contacts PCA: Ab40, Rg")
    ccenter = kmeans_cluster(CPC1, CPC2, clusters= c_clusters, title = "Contacts, Kmeans Clustering")
    plot_free_energy(CPC1, CPC2, n_cmap, centers=ccenter, scat=True, title="Contacts PCA: Ab40, Free Energy", xmin=-2, xmax=3, ymin=-2, ymax=3)
    
    # write the cluster centers to file 
    with open(outdir+'/centers.txt', 'w') as f:
        f.write("Writhe Centers \n")
        for num, i in enumerate(wcenter): 
            f.write(str(num) + " " + str(i) + "\n")

        f.write("Contact Centers \n")
        for num, i in enumerate(ccenter): 
            f.write(str(num) + " " + str(i) + "\n")
    f.close()
    
    # save avg writhe plot 
    
    plot_avg_matrx(writhe_matrx, title="Writhe, Average", count=0)
    plot_avg_matrx(contact_avg, title="Contacts, Average", count=0)
    
    # Use the following functions in a notebook, to toggle between different clusters manually
    #writhe_avg_from_ind(wr_total, CAlabel, np_ind)
    #locate_basin(PC1,PC2, box)
    
    
    return

    
    
if __name__ == "__main__":
    main()