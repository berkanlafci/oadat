#-----
# Description   : Data reader
# Date          : February 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

# import Python libraries
import os
import h5py
import time
import logging
import numpy as np
import scipy.io as sio

class oaReader():
    """
    Optoacoustic data reader

    :param folderPath:      3D array (samples x channels x repetition) of signals
    :param scanName:        Name of data file inside the folder
    :param averaging:       Whether to apply averaging or not (default = False)
    :param averagingAxis:   If averaging True, average through this axis

    :return:            oaReader object
    """
    
    # initialize the class
    def __init__(self, folderPath=None, scanName=None, averaging=False, averagingAxis=2):
        logging.info('  Class       "oaReader"          : %s', __name__)
        
        self._folderPath    = folderPath
        self._scanName      = scanName
        self._averaging     = averaging
        self._averagingAxis = averagingAxis
 
        # print info about process
        print('***** reading data *****')
        startTime        = time.time()
            
        if self.folderPath==None or scanName==None:
            print('WARNING: Data path is not valid creating random data for test')
            self.sigMat             = np.random.uniform(low=-1, high=1, size=(2032,512,1))
            self.acquisitionInfo    = {}
        else:           
            # read data using h5py
            signalFile       = h5py.File(os.path.join(self.folderPath, (self.scanName+'.mat')), 'r')
            
            # check availability of sigMat
            if not any(keyCheck== 'sigMat' for keyCheck in signalFile.keys()):
                raise AssertionError('No sigMat variable key found in data!')

            # read acquisitionInfo and sigMat
            for keyValue in signalFile.keys():
                if keyValue == 'sigMat':
                    self.sigMat             = np.transpose(signalFile['sigMat'])

            # WARNING: If mat file is not saved with -v7.3 use this method
            # signalFile              = sio.loadmat(filePath)
            # self.acquisitionInfo    = signalFile['acquisitionInfo']
            # self.sigMat             = signalFile['sigMat']

        # expand dimensions and average
        if averaging == True:
            if np.ndim(self.sigMat) == 2:
                self.sigMat = np.expand_dims(self.sigMat, axis=2)
            else:
                self.sigMat = averageSignals(self.sigMat, axis=self.averagingAxis)
                self.sigMat = np.expand_dims(self.sigMat, axis=2)
        else:
            if np.ndim(self.sigMat) == 2:
                self.sigMat = np.expand_dims(self.sigMat, axis=2)
            else:
                self.sigMat = self.sigMat

        # remove first 2 samples as they do not have signals
        self.sigMat = self.sigMat[2:,...]
        
        endTime = time.time()
        print('time elapsed: %.2f' %(endTime-startTime))

    #--------------------------------#
    #---------- properties ----------#
    #--------------------------------#

    #--------------------------------#
    # Path to folder

    @property
    def folderPath(self):
        return self._folderPath

    @folderPath.setter
    def folderPath(self, value):
        self._folderPath = value

    @folderPath.deleter
    def folderPath(self):
        del self._folderPath
    
    #--------------------------------#
    # Scan name inside the folder

    @property
    def scanName(self):
        return self._scanName

    @scanName.setter
    def scanName(self, value):
        self._scanName = value

    @scanName.deleter
    def scanName(self):
        del self._scanName

    #--------------------------------#
    # Bool for averaging or not

    @property
    def averaging(self):
        return self._averaging

    @averaging.setter
    def averaging(self, value):
        self._averaging = value

    @averaging.deleter
    def averaging(self):
        del self._averaging

    #--------------------------------#
    # Axis to average signals

    @property
    def averagingAxis(self):
        return self._averagingAxis

    @averagingAxis.setter
    def averagingAxis(self, value):
        self._averagingAxis = value

    @averagingAxis.deleter
    def averagingAxis(self):
        del self._averagingAxis
