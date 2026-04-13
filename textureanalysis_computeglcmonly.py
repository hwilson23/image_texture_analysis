import numpy as np
import tifffile as tiff
import os
from skimage.feature import graycomatrix
from joblib import Parallel, delayed
import glob
from skimage.feature import graycomatrix


# --- parameters ---
loc = r"C:/Users/loci.user/code/helenanalysis/"
out_dir = r"C:/Users/loci.user/code/helenanalysis/texturemap"

kernelsize = 5
filelist = []

# ------------------


def sliding_window(image, kernelsize, stride=1, angles=0):
    kh = kernelsize
    kw = kernelsize
    ih, iw = image.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    outputglcmmatrix = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # skipping calculating background
            if image[i, j] == 0:
                continue
            else:
                # output[i, j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw])
                glcm = graycomatrix(
                    image[i * stride : i * stride + kh, j * stride : j * stride + kw],
                    distances=[1],
                    angles=angles,
                    levels=256,
                    symmetric=True,
                    normed=True,
                )
                outputglcmmatrix = glcm


    return (outputglcmmatrix)


def process_z(z, imgpad, kernelsize):
    glcmmatrix = sliding_window(
        imgpad[:, :, z],
        kernelsize,
        stride=1,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    )
    print(f"still thinking: z is {z}")
    return glcmmatrix

print('started')
for files in glob.glob(loc + "*appliedMaskmean*"):
    filelist.append(files)
    print('looking for files')
print(filelist)


for f in filelist:
    print(f)
    img = tiff.imread((f"{f}")).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    # padding with 0 and keeping 0 in mask bc glcm cant handle NaNs. has some bias towards background now, but i figured better than mean
    # skipping calcuating values at 0 (for speed and so no texture calculated for background)
    imgpad = np.pad(
        img,
        [
            (kernelsize // 2, kernelsize // 2),
            (kernelsize // 2, kernelsize // 2),
            (0, 0),
        ],
        mode="constant",
        constant_values=0,
    )
    print(imgpad.shape)

    glcmmatriximg = np.zeros(img.shape)

    results = Parallel(n_jobs=-2)(  # uses all CPU cores except 1
        delayed(process_z)(z, imgpad, kernelsize) for z in range(img.shape[2])
    )

    for z, (glcmmatrixmean) in enumerate(results):
        glcmmatriximg[:, :, z] = glcmmatrixmean


    out_path_glcmmatrix = os.path.join(out_dir, f"{f[:-4]}_glcmmatrix.tif")

    glcmmatriximg = np.transpose(glcmmatriximg, (2, 0, 1)).astype(np.float32)
    print(glcmmatriximg.shape)

    tiff.imwrite(out_path_glcmmatrix, glcmmatriximg.astype(np.float32))
