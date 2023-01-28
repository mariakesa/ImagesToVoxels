import numpy as np
from utils import argObj, load_fmri
from sklearn.decomposition import PCA


data_dir='/home/maria/Algonauts2023'
parent_submission_dir='/home/maria/Algonauts2023_submission'

all_verts_lh=np.load('/home/maria/Algonauts2023/subj01/roi_masks/mapping_floc-bodies.npy',allow_pickle=True)

print(all_verts_lh.shape)
print(np.unique(all_verts_lh))

args=argObj(data_dir,parent_submission_dir, 1)

lh_fmri,rh_fmri=load_fmri(args)
print(lh_fmri.shape)
pca=PCA(n_components=1000)
#pcs=pca.fit_transform(lh_fmri)