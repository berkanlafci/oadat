#-----
# Description   : Data saver
# Date          : February 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

import os
import h5py
import logging
import numpy as np
import PIL.Image as img
import scipy.io as sio

#--------------------------------#
# Save images as png files

def saveImagePng(reconObject=None, pngPath=None, saveName=None, imageRecon=None):
    """
    Save images in png files

    :param pngPath: Folder path to save images
    :param saveName: Saving name for individual file
    :param imageRecon: Reconstructed image to save
    
    """
    logging.info('  Function    "saveImagePng"      : %s', __name__)

    # loop trhorugh repetitions
    for repInd in range(0, reconObject.numRepetitions):

        # loop through wavelengths
        for waveInd in range(0, reconObject.numWavelengths):

            maxValue = np.amax(imageRecon[:,:,waveInd,repInd])
            minValue = np.amin(imageRecon[:,:,waveInd,repInd])

            # normalize image
            normRecon = 255*(imageRecon[:,:,waveInd,repInd]-minValue)/(maxValue-minValue)

            # convert to rgb
            arrayRecon = img.fromarray(normRecon)
            if arrayRecon.mode != 'RGB':
                rgbRecon = arrayRecon.convert('RGB')
            
            rgbRecon.save(pngPath + '/' + saveName + '_' + str(reconObject.wavelengths[waveInd]) + '.jpg')

#--------------------------------#
# Save images as mat files

def saveImageMat(reconObject=None, matPath=None, saveName=None, imageRecon=None):
    """
    Save images in mat files

    :param matPath: Folder path to save images
    :param saveName: Saving name for individual file
    :param imageRecon: Reconstructed image to save
    
    """
    logging.info('  Function    "saveImageMat"      : %s', __name__)

    # loop trhorugh repetitions
    for repInd in range(0, reconObject.numRepetitions):

        # loop through wavelengths
        for waveInd in range(0, reconObject.numWavelengths):
            imageDict = {"imageRecon": imageRecon[:,:,waveInd,repInd]}

            sio.savemat((matPath + '/' + saveName + '_' + str(reconObject.wavelengths[waveInd]) + '.mat'), imageDict)

#--------------------------------#
# Save signals as png files

def saveSignalPng(reconObject=None, pngPath=None, saveName=None, signalRecon=None):
    """
    Save individual signals in png files

    :param pngPath: Folder path to save images
    :param saveName: Saving name for individual file
    :param signalRecon: Reconstructed image to save
    
    """
    logging.info('  Function    "saveSignalPng"     : %s', __name__)

    # loop through wavelengths
    for waveInd in range(0, reconObject.numWavelengths):

        maxValue = np.amax(signalRecon[:,:,waveInd])
        minValue = np.amin(signalRecon[:,:,waveInd])

        # normalize image
        normRecon = 255*(signalRecon[:,:,waveInd]-minValue)/(maxValue-minValue)

        # convert to rgb
        arrayRecon = img.fromarray(normRecon)
        if arrayRecon.mode != 'RGB':
            rgbRecon = arrayRecon.convert('RGB')
        
        rgbRecon.save(pngPath + '/' + saveName + '_' + '.jpg')

#--------------------------------#
# Save signals as mat files

def saveSignalMat(reconObject=None, matPath=None, saveName=None, signalRecon=None):
    """
    Save individual signals in mat files

    :param matPath:         Folder path to save images
    :param saveName:        Saving name for individual file
    :param signalRecon:     Reconstructed image to save
    
    """
    logging.info('  Function    "saveSignalMat"     : %s', __name__)

    # loop through wavelengths
    for waveInd in range(0, reconObject.numWavelengths):
        imageDict = {"signalRecon": signalRecon[:,:,waveInd]}

        sio.savemat((matPath + '/' + saveName + '.mat'), imageDict)

#--------------------------------#
# Save images as h5 files

def saveImageH5(h5Path=None, saveName=None, imageRecon=None):

	savePath 	= os.path.join(h5Path, saveName)
	data 		= {}

	with h5py.File(savePath, 'a', libver='latest') as h5File:
		
		data['imageRecon'] = h5File.create_dataset('imageRecon', data=imageRecon)

