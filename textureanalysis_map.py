import numpy as np
import tifffile as tiff
import os
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import glob
from skimage.feature import graycomatrix, graycoprops


# --- parameters ---
loc  = r"G:/FluorescentCollagen/20260302_ows2_col/appliedMASKimages"
out_dir = r"G:/FluorescentCollagen/20260302_ows2_col/texturemap"
stacknames = ["flu", "bkwshg","fwdshg"]
kernelsize = 5
filelist = []

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
                contrast = graycoprops(glcm, 'contrast')
                homogeneity = graycoprops(glcm, 'homogeneity')
                asm = graycoprops(glcm,'ASM')
                glcmmean = graycoprops(glcm,'mean')
                glcmvar = graycoprops(glcm,'variance')
                entropy = graycoprops(glcm, 'entropy')

                outputdis[i,j] = np.mean(dis)  # equivalent to your per-angle mean
                outputcor[i,j] = np.mean(cor) 
                outcontrast[i,j] = np.mean(contrast)
                outhom[i,j] = np.mean(homogeneity)
                outasm[i,j] = np.mean(asm)
                outglcmmean[i,j] = np.mean(glcmmean)
                outglcmvar[i,j] = np.mean(glcmvar)
                outent[i,j] = np.mean(entropy)
        
    return outputdis,outputcor, outcontrast, outhom, outasm, outglcmmean, outglcmvar, outent

def process_z(z, imgpad, kernelsize):
    dis, cor = sliding_window(imgpad[:,:,z], kernelsize, stride=1,
                              angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    print(f"still thinking: z is {z}")
    return dis,cor


for files in glob.glob(loc +'*appliedMaskmean'):
    filelist.append(files)
print(filelist)

for f in filelist:
    img = tiff.imread((f"{loc}{f}")).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    print(img.shape)
    #padding with 0 and keeping 0 in mask bc glcm cant handle NaNs. has some bias towards background now, but i figured better than mean
    #skipping calcuating values at 0 (for speed and so no texture calculated for background)
    imgpad = np.pad(img, [(kernelsize//2, kernelsize//2), (kernelsize//2, kernelsize//2), (0,0)], mode='constant', constant_values=0)
    print(imgpad.shape)

    disimg = np.zeros(img.shape)
    corimg = np.zeros(img.shape)
    outcontrast = np.zeros(img.shape)
    outhom = np.zeros(img.shape)
    outasm = np.zeros(img.shape)
    outglcmmean = np.zeros(img.shape)
    outglcmvar = np.zeros(img.shape)
    outent = np.zeros(img.shape)

    results = Parallel(n_jobs=-2)(  # uses all CPU cores except 1
        delayed(process_z)(z, imgpad, kernelsize)
        for z in range(img.shape[2])
    )

    for z, (dismean, cormean, contrastmean, homogenitymean, asmmean, glcmmean, glcmvarmean, entropymean) in enumerate(results):
        disimg[:,:,z] = dismean
        corimg[:,:,z] = cormean
        outcontrast[:,:,z] = contrastmean
        outhom[:,:,z] = homogenitymean
        outasm[:,:,z] = asmmean
        outglcmmean[:,:,z] = glcmmean
        outglcmvar[:,:,z] = glcmvarmean
        outent[:,:,z] = entropymean
        #for z in range(img.shape[2]):
        #dissimilarity_angles, correlation_angles = sliding_window(imgpad[:,:,z], kernelsize,stride=1, angles =[0, np.pi/4, np.pi/2, 3* np.pi/4])

        #dismean = np.mean(dissimilarity_angles,axis=2)
        #cormean = np.mean(correlation_angles,axis=2)

        #disimg[:,:,z] = dismean
        #corimg[:,:,z] = cormean
        

    out_path_dis = os.path.join(
                    out_dir,
                    f"{f[:-4]}_dissimilaritymean.tif"
                )
    out_path_cor = os.path.join(out_dir, f"{f[:-4]}_correlationmean.tif")
    out_path_contrast = os.path.join(out_dir, f"{f[:-4]}_contrastmean.tif")
    out_path_hom = os.path.join(out_dir, f"{f[:-4]}_homogeneitymean.tif")
    out_path_asm = os.path.join(out_dir, f"{f[:-4]}_asmmean.tif")
    out_path_mean = os.path.join(out_dir, f"{f[:-4]}_glcmmeanmean.tif")
    out_path_var = os.path.join(out_dir, f"{f[:-4]}_glcmvariancemean.tif")
    out_path_ent = os.path.join(out_dir, f"{f[:-4]}_entropymean.tif")

    disimg = np.transpose(disimg, (2, 0, 1)).astype(np.float32)
    corimg = np.transpose(corimg, (2, 0, 1)).astype(np.float32)
    outcontrast = np.transpose(outcontrast, (2, 0, 1)).astype(np.float32)
    outhom = np.transpose(outhom, (2, 0, 1)).astype(np.float32)
    outasm = np.transpose(outasm, (2, 0, 1)).astype(np.float32)
    outglcmmean = np.transpose(outglcmmean, (2, 0, 1)).astype(np.float32)
    outglcmvar = np.transpose(outglcmvar, (2, 0, 1)).astype(np.float32)
    outent = np.transpose(outent, (2, 0, 1)).astype(np.float32)
    print(disimg.shape)

    tiff.imwrite(out_path_dis, disimg.astype(np.float32))
    tiff.imwrite(out_path_contrast, outcontrast.astype(np.float32))
    tiff.imwrite(out_path_hom, outhom.astype(np.float32))
    tiff.imwrite(out_path_asm, outasm.astype(np.float32))
    tiff.imwrite(out_path_mean, outglcmmean.astype(np.float32))
    tiff.imwrite(out_path_var, outglcmvar.astype(np.float32))
    tiff.imwrite(out_path_cor, outent.astype(np.float32))
    tiff.imwrite(out_path_cor, corimg.astype(np.float32))




