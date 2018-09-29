# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 02:55:45 2018

@author: AdamT
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage import color as cl
from skimage.util import view_as_blocks
from skimage.transform import resize
from sklearn import mixture
from numba import jit

#=================================================
@jit
def getHists(img,bins=50):
    """Generate histogram containing <bins> amount of bins. 
    
    i-th value represents density of pixels with values contained in i-th bin
    """
    hists = np.array([])
    for i in range(3):#Images are loaded as three-dimensional matrices with three channels
        hists = np.append(hists,np.histogram(img[:,:,i], bins, density = True)[0])
    return hists

def genHistArrays(df,csname,bins=50):
    """Create pixels' color histograms for every row in df
    
    df - input dataframe
    csname - color space (RGB, HSV, CIE or YdBdR)
    bins - specifies how many bins should be in histogram for each color channel
    """
    #initiate matrix which will contain values of histograms
    allpixV = np.zeros((df.shape[0],bins*3))
    #attain histograms
    hists = df['SKImage'].apply(lambda x: getHists(x,bins))
    
    #Generate column names for result dataframe
    fullnames = []
    for chs in ['CH1', 'CH2', 'CH3']:
        fullnames.extend([chs+'-'+str(j) for j in range(bins)])
    fullnames = [csname+'-'+str(j) for j in fullnames]
    
    #extract histograms
    for rowi, histArr in enumerate(hists):
        allpixV[rowi,:] = np.array(histArr).flatten()
        
    return allpixV,fullnames
#=================================================
def blocksForImg(img,samplsiz,resc):
    imgr = resize(img,(resc,resc,3))
    view = []
    view.append(view_as_blocks(imgr[:,:,0], (samplsiz,samplsiz)))
    view.append(view_as_blocks(imgr[:,:,1], (samplsiz,samplsiz)))
    view.append(view_as_blocks(imgr[:,:,2], (samplsiz,samplsiz)))
    flv = [v.reshape(v.shape[0],v.shape[1],-1) for v in view]
    meanb = np.array([np.mean(f,axis=2) for f in flv])
    stdb = np.array([np.std(f,axis=2) for f in flv])
    meanb = meanb.flatten()
    stdb = stdb.flatten()
    meanbsort = np.sort(meanb)
    stdbsort = np.sort(stdb)
    return np.append(meanb,[stdb,meanbsort,stdbsort])

def blocksVals(df,ssiz,resc,cspacename='RGB'):
    amountBlocks = np.power(resc/ssiz,2)
    amountBlocks = int(amountBlocks)
    labelnames = []
    for ch in ['CH1','CH2','CH3']:   
        labelnames = np.append(labelnames,[cspacename+'-Mean-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])
        labelnames = np.append(labelnames,[cspacename+'-Std-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])
        labelnames = np.append(labelnames,[cspacename+'-Mean_Sorted-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])
        labelnames = np.append(labelnames,[cspacename+'-Std_Sorted-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])    
    blockArr = np.zeros((df.shape[0],np.size(labelnames)))
    blockDesc = df['SKImage'].apply(lambda x: blocksForImg(x,ssiz,resc))
    for i,bl in enumerate(blockDesc):
        blockArr[i,:] = bl
    return blockArr,labelnames
#=================================================
    
@jit
def runGMM(Xseg,n_kernels):
    gmm = mixture.GaussianMixture(n_components=n_kernels, covariance_type='full')
    gmm.fit(Xseg)
    labels = gmm.predict(Xseg)
    return labels

@jit
def prepareImg2GaussM(img):
    imgforSeg = np.zeros((img.shape[0]*img.shape[1],5))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgforSeg[(img.shape[0]*j)+i,:] = np.append(img[i][j],[j,i])
    Xseg = np.array(imgforSeg)
    return Xseg

@jit
def getClustersInfo(img,Xseg,labels,n_kernels):
    labsDat = pd.DataFrame(data = Xseg)
    labsDat['label'] = labels
    labsDat[3] = labsDat[3]/img.shape[0]
    labsDat[4] = labsDat[4]/img.shape[1]
    siz = np.zeros(n_kernels)
    mean = np.zeros((n_kernels,5))
    for i in range(5):
        siz[i] = labsDat.loc[labsDat['label']==i,0].shape[0]/labsDat.shape[0]
        mean[i,:] = labsDat.loc[labsDat['label']==i].mean(axis=0)[:-1]
    centrsDF = pd.DataFrame(data = mean)
    centrsDF['size'] = siz
    centrsDF = centrsDF.sort_values(by=0).append(centrsDF.sort_values(by=1)).append(
    centrsDF.sort_values(by=2)).append(centrsDF.sort_values(by=3)).append(
    centrsDF.sort_values(by=4)).append(centrsDF.sort_values(by='size'))
    centrDataArr = centrsDF.values.flatten()
    return centrDataArr

def prepareLabels(n_kernels,cspace):
    centrDataLabels = []
    for i in range(n_kernels):
        for s in ['sortCH1','sortCH2','sortCH3','sortX','sortY','sortSize']:
            centrDataLabels = np.append(
                centrDataLabels, ['Cluster: ' + str(i)+' cspace: ' + cspace + '; ' + s + '; ' + var for var in
                             ['CH1', 'CH2', 'CH3', 'X', 'Y', 'SIZE']])
    return centrDataLabels

@jit
def processImageGM(img, n_kernels, scale_factor = 1):
    if scale_factor != 1:
        img = resize(img,(int(img.shape[0]/scale_factor),int(img.shape[1]/scale_factor),3))
    data2seg = prepareImg2GaussM(img)
    labels = runGMM(data2seg,n_kernels)
    return getClustersInfo(img,data2seg,labels,n_kernels)

def getSegmentInfo(df,cspace,n_kernels, scale_factor = 1):
    colNames = prepareLabels(n_kernels,cspace)
    tqdm.pandas(ncols=50)
    segArr = np.zeros((df.shape[0],np.size(colNames)))
    allImgsSegInfo = df.SKImage.progress_apply(lambda x: processImageGM(x,n_kernels,scale_factor))
    for i,seg in enumerate(allImgsSegInfo):
        segArr[i,:] = seg
    return segArr,colNames

@jit
def addRawPixValsImg(img,new_width=32,new_height=32):
    img = resize(img,(new_width,new_height,3))
    return img.flatten()

def addRawPixValsImgDf(df,cspace,new_width=32,new_height=32):
    rawPixLabels = ['RawPix_' + cspace + str(i) for i in range(new_width*new_height*3)]
    rawPxDF = df['SKImage'].apply(lambda x: addRawPixValsImg(x,new_width,new_height))
    rawPicsVects = np.zeros((df.shape[0],np.size(rawPixLabels)))
    for i,flatIms in enumerate(rawPxDF):
        rawPicsVects[i,:] = flatIms
    return rawPicsVects,rawPixLabels
#=================================================
    
def describeImgs(df,cspace,bins=50,blocksResiz=160,blocksblSiz=20,n_clusters=5,clustResiz=3,
                 new_width=32,new_height=32):
    histV,histCols = genHistArrays(df,cspace,bins)
    histDF = pd.DataFrame(data = histV, columns = histCols)
    blocksV, blocksCols = blocksVals(df,blocksblSiz,blocksResiz, cspace)
    blocksDF = pd.DataFrame(data = blocksV, columns = blocksCols)
    segV, segCols = getSegmentInfo(df,cspace,n_clusters, clustResiz)
    segDF = pd.DataFrame(data = segV, columns = segCols)
    rawPicsV, rawPixCols = addRawPixValsImgDf(df,cspace,new_width,new_height)
    rawPicsDF = pd.DataFrame(data = rawPicsV, columns = rawPixCols)
    return pd.concat([histDF,blocksDF,segDF,rawPicsDF], axis=1)

def extractVarsForML(df,bins=50,blocksResiz=160,blocksblSiz=20,n_clusters=5,clustResiz=3,new_width=32,new_height=32):
    resDF = describeImgs(df,'RGB',bins,blocksResiz,blocksblSiz,n_clusters,clustResiz,
                         new_width,new_height)
    cspdict = {'HSV': cl.rgb2hsv, 'CIE': cl.rgb2rgbcie, 'YDBDR': cl.rgb2ydbdr}
    for csp, colFunc in cspdict.items():
        df['SKImage'] = df['SKImage'].apply(colFunc)
        resDF = pd.concat([resDF, describeImgs(df,csp,bins,blocksResiz,blocksblSiz,n_clusters,clustResiz,
                                               new_width,new_height)], axis=1)
    return resDF