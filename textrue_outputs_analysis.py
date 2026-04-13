import numpy as np
import tifffile as tiff
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


# --- parameters ---
loc = r"G:/FluorescentCollagen/20260302_ows2_col/texturemap/"
#out_dir = r"C:/Users/loci.user/code/helenanalysis/texturemap"
taglist = ['contrastmean', 'asmmean', 'glcmmeanmean', 'glcmvariancemean', 'correlationmean']


bkwshglist = []
fwdshglist = []
flulist = []


for files in glob.glob(loc + "*bkwshg*"):
    bkwshglist.append(files)

for files in glob.glob(loc +"*fwdshg*"):
    fwdshglist.append(files)

for files in glob.glob(loc +"*flu*"):
    flulist.append(files)

data = []
allfiles = bkwshglist + fwdshglist + flulist
#print(allfiles)

for f in allfiles:
    type = os.path.basename(f).split("_")[1]
    tag = f.split("8bit_")[-1].split(".")[0]
    
    if tag in taglist:
        print(f"type: {type}, tag: {tag}")
        img = tiff.imread((f"{f}")).astype(np.uint8)
        
        imgmean = np.mean(img)
        imgstd = np.std(img)

        data.append({"geltype": type, "prop":tag, "imgmean": imgmean, "imgstd": imgstd})

df = pd.DataFrame(data)

print(df)


#making charts
x = np.arange(3)
dfx = pd.DataFrame()

for tagit in taglist:
    dfx = df.loc[df["prop"] == tagit]
    title = tagit
    plt.figure(figsize=(10, 3))#, dpi=80)
    print(x)
    print(dfx.imgmean)
    plt.errorbar(x, dfx.imgmean, yerr=dfx.imgstd, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=5)
    plt.title(title)
    plt.show()
    #plt.savefig(f"E:\\MNtissueproject_CLEANED20250716\\figures\\figure2\\histograms\\{title}.pdf", format='pdf', bbox_inches='tight')
    #break


