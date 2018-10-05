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
    """Split image into blocks of pixels and calculate mean and standard variation of each block
    """
    #scale image into square shape
    imgr = resize(img,(resc,resc,3))

    #create blocks
    view = []
    view.append(view_as_blocks(imgr[:,:,0], (samplsiz,samplsiz)))
    view.append(view_as_blocks(imgr[:,:,1], (samplsiz,samplsiz)))
    view.append(view_as_blocks(imgr[:,:,2], (samplsiz,samplsiz)))

    #calculate mean values and standard deviation for each block
    flv = [v.reshape(v.shape[0],v.shape[1],-1) for v in view]
    meanb = np.array([np.mean(f,axis=2) for f in flv])
    stdb = np.array([np.std(f,axis=2) for f in flv])

    #return vector cancatenated from means, standard deviations, sorted means and sorted deviations
    meanb = meanb.flatten()
    stdb = stdb.flatten()
    meanbsort = np.sort(meanb)
    stdbsort = np.sort(stdb)
    return np.append(meanb,[stdb,meanbsort,stdbsort])

def blocksVals(df,ssiz,resc,cspacename='RGB'):
    """Divide each image in dataframe into set of blocks,
    calculate their mean pixel values and standard deviations and concatenate those values in following manner: [unsorted means, unsorted stds, sorted means, sorted stds]
    """

    #calculate amount of generated blocks
    amountBlocks = np.power(resc/ssiz,2)
    amountBlocks = int(amountBlocks)

    #generate labels for blocks
    labelnames = []
    for ch in ['CH1','CH2','CH3']:   
        labelnames = np.append(labelnames,[cspacename+'-Mean-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])
        labelnames = np.append(labelnames,[cspacename+'-Std-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])
        labelnames = np.append(labelnames,[cspacename+'-Mean_Sorted-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])
        labelnames = np.append(labelnames,[cspacename+'-Std_Sorted-Block: ' + ch + ' ' + str(i) for i in range(1,amountBlocks+1)])    

    #generate vectors of block values: [unsorted means, unsorted stds, sorted means, sorted stds]
    blockArr = np.zeros((df.shape[0],np.size(labelnames)))
    blockDesc = df['SKImage'].apply(lambda x: blocksForImg(x,ssiz,resc))
    for i,bl in enumerate(blockDesc):
        blockArr[i,:] = bl
    return blockArr,labelnames
#=================================================
    
@jit
def runGMM(Xseg, n_clusters):
    '''Perform Gaussian Mixture clustering on preprocessed image data
    '''
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
    gmm.fit(Xseg)
    labels = gmm.predict(Xseg)
    return labels

@jit
def prepareImg2GaussM(img):
    '''Prepare image for clustering. image will be transformed into array of size (amount_of_image_pixels,5),
        where each row contains following info on corresponding pixel: [channel_1_value, channel_2_value, channel_3_value, x_position, y_position].
        Thanks to including information about spacial placement of pixels,
        generated clusters will merge neighbouring pixels and will better describe actual objects on image.
    '''
    imgforSeg = np.zeros((img.shape[0]*img.shape[1],5))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgforSeg[(img.shape[0]*j)+i,:] = np.append(img[i][j],[j,i])
    Xseg = np.array(imgforSeg)
    return Xseg

@jit
def getClustersInfo(img, Xseg, labels, n_clusters):
    '''Transform each cluster into vector of features
    '''
    labsDat = pd.DataFrame(data = Xseg)
    labsDat['label'] = labels
    #normalize pixels' coordinates
    labsDat[3] = labsDat[3]/img.shape[0]
    labsDat[4] = labsDat[4]/img.shape[1]
    siz = np.zeros(n_clusters)
    mean = np.zeros((n_clusters, 5))
    #Each cluster is transformed into set of values (mean values for each channel, mean X and Y position of pixels and size)
    #Subsequently, vecotrs are created form clusters sorted among each of values. those sorted vectors are later concatenated
    for i in range(n_clusters):
        siz[i] = labsDat.loc[labsDat['label']==i,0].shape[0]/labsDat.shape[0]
        mean[i,:] = labsDat.loc[labsDat['label']==i].mean(axis=0)[:-1]
    centrsDF = pd.DataFrame(data = mean)
    centrsDF['size'] = siz

    #concatenate sorted values
    centrsDF = centrsDF.sort_values(by=0).append(centrsDF.sort_values(by=1)).append(
    centrsDF.sort_values(by=2)).append(centrsDF.sort_values(by=3)).append(
    centrsDF.sort_values(by=4)).append(centrsDF.sort_values(by='size'))

    centrDataArr = centrsDF.values.flatten()
    return centrDataArr

def prepareLabels(n_clusters, cspace):
    '''Prepare labels for vectors of clusters' characteristics
    '''
    centrDataLabels = []
    for i in range(n_clusters):
        for s in ['sortCH1','sortCH2','sortCH3','sortX','sortY','sortSize']:
            centrDataLabels = np.append(
                centrDataLabels, ['Cluster: ' + str(i)+' cspace: ' + cspace + '; ' + s + '; ' + var for var in
                             ['CH1', 'CH2', 'CH3', 'X', 'Y', 'SIZE']])
    return centrDataLabels

@jit
def processImageGM(img, n_clusters, scale_factor = 1):
    '''Perform Gaussian Mixture clustering for image
    '''
    if scale_factor != 1:
        img = resize(img,(int(img.shape[0]/scale_factor),int(img.shape[1]/scale_factor),3))
    data2seg = prepareImg2GaussM(img)
    labels = runGMM(data2seg, n_clusters)
    return getClustersInfo(img, data2seg, labels, n_clusters)

def getSegmentInfo(df, cspace, n_clusters, scale_factor = 1):
    '''Generate clusters' data for records in dataframe
    '''
    colNames = prepareLabels(n_clusters, cspace)
    tqdm.pandas(ncols=50)
    segArr = np.zeros((df.shape[0],np.size(colNames)))
    allImgsSegInfo = df.SKImage.progress_apply(lambda x: processImageGM(x, n_clusters, scale_factor))
    for i,seg in enumerate(allImgsSegInfo):
        segArr[i,:] = seg
    return segArr,colNames

@jit
def addRawPixValsImg(img,new_width=32,new_height=32):
    '''Scale image and get raw pixels' values
    '''
    img = resize(img,(new_width,new_height,3))
    return img.flatten()

def addRawPixValsImgDf(df,cspace,new_width=32,new_height=32):
    '''Get raw pixels' values for records in dataframe
    '''
    #Generate labels
    rawPixLabels = ['RawPix_' + cspace + str(i) for i in range(new_width*new_height*3)]
    rawPxDF = df['SKImage'].apply(lambda x: addRawPixValsImg(x,new_width,new_height))
    rawPicsVects = np.zeros((df.shape[0],np.size(rawPixLabels)))
    for i,flatIms in enumerate(rawPxDF):
        rawPicsVects[i,:] = flatIms
    return rawPicsVects,rawPixLabels
#=================================================
    
def describeImgs(df,cspace,bins=50,blocksResiz=160,blocksblSiz=20,n_clusters=5,clustResiz=3,
                 new_width=32,new_height=32):
    '''Extract features for records in dataframe
    '''

    #Extract histograms
    histV,histCols = genHistArrays(df,cspace,bins)
    histDF = pd.DataFrame(data = histV, columns = histCols)

    #Extract pixel blocks
    blocksV, blocksCols = blocksVals(df,blocksblSiz,blocksResiz, cspace)
    blocksDF = pd.DataFrame(data = blocksV, columns = blocksCols)

    #Extract clusters
    segV, segCols = getSegmentInfo(df,cspace,n_clusters, clustResiz)
    segDF = pd.DataFrame(data = segV, columns = segCols)

    #Extract raw pixels
    rawPicsV, rawPixCols = addRawPixValsImgDf(df,cspace,new_width,new_height)
    rawPicsDF = pd.DataFrame(data = rawPicsV, columns = rawPixCols)

    #Concatenate features
    return pd.concat([histDF,blocksDF,segDF,rawPicsDF], axis=1)

def extractVarsForML(df,bins=50,blocksResiz=160,blocksblSiz=20,n_clusters=5,clustResiz=3,new_width=32,new_height=32):
    '''Extract features for RGB, HSV, CIE and YdBdR'''
    resDF = describeImgs(df,'RGB',bins,blocksResiz,blocksblSiz,n_clusters,clustResiz,
                         new_width,new_height)
    cspdict = {'HSV': cl.rgb2hsv, 'CIE': cl.rgb2rgbcie, 'YDBDR': cl.rgb2ydbdr}
    for csp, colFunc in cspdict.items():
        df['SKImage'] = df['SKImage'].apply(colFunc)
        resDF = pd.concat([resDF, describeImgs(df,csp,bins,blocksResiz,blocksblSiz,n_clusters,clustResiz,
                                               new_width,new_height)], axis=1)
    return resDF