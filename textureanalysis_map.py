import numpy as np
import tifffile as tiff
import os
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage import exposure
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from skimage.feature import graycomatrix, graycoprops
from skimage import data


# --- parameters ---
IMAGE_PATH  = r"G:/FluorescentCollagen/20260302_ows2_col/appliedMASKmean_flu_8bit_testpatch.tif"
out_dir = r"G:/FluorescentCollagen/20260302_ows2_col/texturemap"
stackname = r"flu"
kernelsize = 5

# ------------------

def sliding_window(image, kernelsize, stride=1, angles=0):
    kh = kernelsize
    kw = kernelsize
    ih, iw = image.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    outputdis = np.zeros((oh, ow))
    outputcor = np.zeros((oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            #skipping calculating background
            if image[i,j] == 0:
                continue
            else:
                #output[i, j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw])
                glcm = graycomatrix(
                image[i*stride:i*stride+kh, j*stride:j*stride+kw], distances=[1], angles=angles, levels=256, symmetric=True, normed=True
                )
                #outputdis[i,j,a] =(graycoprops(glcm, 'dissimilarity')[0, 0])
                #outputcor[i,j,a] =(graycoprops(glcm, 'correlation')[0, 0])

                dis = graycoprops(glcm, 'dissimilarity')  # shape (4,) — one value per angle
                cor = graycoprops(glcm, 'correlation')    # shape (4,)

                outputdis[i,j] = np.mean(dis)  # equivalent to your per-angle mean
                outputcor[i,j] = np.mean(cor) 
        
    return outputdis,outputcor

def process_z(z, imgpad, kernelsize):
    dis, cor = sliding_window(imgpad[:,:,z], kernelsize, stride=1,
                              angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    print(f"still thinking: z is {z}")
    return dis,cor


img = tiff.imread(IMAGE_PATH).astype(np.uint8)
img = np.transpose(img, (1,2,0))
print(img.shape)
#padding with 0 and keeping 0 in mask bc glcm cant handle NaNs. has some bias towards background now, but i figured better than mean
#skipping calcuating values at 0 (for speed and so no texture calculated for background)
imgpad = np.pad(img, [(kernelsize//2, kernelsize//2), (kernelsize//2, kernelsize//2), (0,0)], mode='constant', constant_values=0)
print(imgpad.shape)

disimg = np.zeros(img.shape)
corimg = np.zeros(img.shape)

results = Parallel(n_jobs=-2)(  # uses all CPU cores
    delayed(process_z)(z, imgpad, kernelsize)
    for z in range(img.shape[2])
)

for z, (dismean, cormean) in enumerate(results):
    disimg[:,:,z] = dismean
    corimg[:,:,z] = cormean
    #for z in range(img.shape[2]):
    #dissimilarity_angles, correlation_angles = sliding_window(imgpad[:,:,z], kernelsize,stride=1, angles =[0, np.pi/4, np.pi/2, 3* np.pi/4])

    #dismean = np.mean(dissimilarity_angles,axis=2)
    #cormean = np.mean(correlation_angles,axis=2)

    #disimg[:,:,z] = dismean
    #corimg[:,:,z] = cormean
    

out_path_dis = os.path.join(
                out_dir,
                f"{stackname}_dissimilaritymean.tif"
            )
out_path_cor = os.path.join(
                out_dir,
                f"{stackname}_correlationmean.tif"
            )

disimg = np.transpose(disimg, (2, 0, 1)).astype(np.float32)
corimg = np.transpose(corimg, (2, 0, 1)).astype(np.float32)
print(disimg.shape)

tiff.imwrite(out_path_dis, disimg.astype(np.float32))
tiff.imwrite(out_path_cor, corimg.astype(np.float32))




