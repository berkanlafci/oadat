#-----
# Description   : Reconstruction functions on CPU
# Date          : February 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

# import libraries
import time
import h5py
import math
import logging
import numpy as np
import pkg_resources as pkgr
from scipy.sparse.linalg import lsqr
from scipy.sparse import vstack

#----------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ backprojection ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#

# recon backprojection class
class cpuBP():
    """
    Back projection reconstruction class on CPU for optoacoustic imaging

    :param fieldOfView:     Field of view in x and y direction in meters (default: 0.03)
    :param pixelNumber:     Number of pixels in image (default: 128)
    :param xSensor:         X position of transducer elements (default: Ring array positions)
    :param ySensor:         Y position of transducer elements (default: Ring array positions)
    :param cupType:         Array type used in experiment (default: Ring array)
    :param speedOfSound:    Estimated speed of sound (m/s) based on temperature (default: 1540 m/s)
    :param nSamples:        Number of samples in sigMat (default: 2032)
    :param fSampling:       Sampling frequency of signals (default: 40e6)
    :param reconType:       Reconstruction type for backprojection (default: full)
    :param delayInSamples:  Active delay of signals (default: 61)
    :param wavelengths:     Wavelengths used in illumination (default: 800 nm)
    :param numWavelengths:  Number of wavelengths used in acquisition
    :param numRepetitions:  Number of repetitions
    :param lowCutOff:       Low cut off frequency of bandpass filter in MHz (default: 0.1e6 MHz)
    :param highCutOff:      High cut off frequency of bandpass filter in MHz (default: 6e6 MHz)
    :param fOrder:          Butterworth filter order

    :return:                cpuBP object
    """

    # initialization for class cpuBP
    def __init__(self):
        logging.info('  Class       "cpuBP"             : %s', __name__)

        self._fieldOfView       = 0.03
        self._pixelNumber       = 128
        self._xSensor           = {}
        self._ySensor           = {}
        self._cupType           = 'ring'
        self._speedOfSound      = 1540
        self._nSamples          = 2032
        self._fSampling         = 40e6
        self._reconType         = 'full'
        self._delayInSamples    = 0
        self._wavelengths       = [800]
        self.__numWavelengths   = 1
        self.__numRepetitions   = 1
        self._lowCutOff         = 0.1e6
        self._highCutOff        = 6e6
        self._fOrder            = 3
    
    #--------------------------------#
    #---------- properties ----------#
    #--------------------------------#

    #--------------------------------#
    # field of view

    @property
    def fieldOfView(self):
        return self._fieldOfView

    @fieldOfView.setter
    def fieldOfView(self, value):
        logging.info('  Property    "fieldOfView"       : %.4f m', value)
        self._fieldOfView = value

    @fieldOfView.deleter
    def fieldOfView(self):
        del self._fieldOfView

    #--------------------------------#
    # pixel number

    @property
    def pixelNumber(self):
        return self._pixelNumber

    @pixelNumber.setter
    def pixelNumber(self, value):
        logging.info('  Property    "pixelNumber"       : %d ', value)
        self._pixelNumber = value

    @pixelNumber.deleter
    def pixelNumber(self):
        del self._pixelNumber
    
    #--------------------------------#
    # x positions of sensor

    @property
    def xSensor(self):
        return self._xSensor

    @xSensor.setter
    def xSensor(self, value):
        self._xSensor = value

    @xSensor.deleter
    def xSensor(self):
        del self._xSensor
    
    #--------------------------------#
    # y positions of sensor

    @property
    def ySensor(self):
        return self._ySensor

    @ySensor.setter
    def ySensor(self, value):
        self._ySensor = value

    @ySensor.deleter
    def ySensor(self):
        del self._ySensor
    
    #--------------------------------#
    # array properties

    @property
    def arrayData(self):
        return self.__arrayData

    @arrayData.setter
    def arrayData(self, value):
        self.__arrayData = value

    @arrayData.deleter
    def arrayData(self):
        del self.__arrayData
       
    #--------------------------------#
    # cup type

    @property
    def cupType(self):
        return self._cupType

    @cupType.setter
    def cupType(self, value):
        logging.info('  Property    "cupType"           : %s', value)
        self._cupType       = value
        self.__arrayDir     = pkgr.resource_filename('pyoat', 'arrays/'+self._cupType+'Cup.mat')
        self.__arrayData    = h5py.File(self.__arrayDir, 'r')
        self.xSensor        = self.__arrayData['transducerPos'][0,:]
        self.ySensor        = self.__arrayData['transducerPos'][1,:]

    @cupType.deleter
    def cupType(self):
        del self._cupType
        del self.xSensor
        del self.ySensor

    #--------------------------------#
    # speed of sound

    @property
    def speedOfSound(self):
        return self._speedOfSound

    @speedOfSound.setter
    def speedOfSound(self, value):
        logging.info('  Property    "speedOfSound"      : %d m/s', value)
        self._speedOfSound = value

    @speedOfSound.deleter
    def speedOfSound(self):
        del self._speedOfSound

    #--------------------------------#
    # number of samples

    @property
    def nSamples(self):
        return self._nSamples

    @nSamples.setter
    def nSamples(self, value):
        logging.info('  Property    "nSamples"          : %d ', value)
        self._nSamples = value

    @nSamples.deleter
    def nSamples(self):
        del self._nSamples

    #--------------------------------#
    # sampling frequency

    @property
    def fSampling(self):
        return self._fSampling

    @fSampling.setter
    def fSampling(self, value):
        logging.info('  Property    "fSampling"         : %d Hz', value)
        self._fSampling = value

    @fSampling.deleter
    def fSampling(self):
        del self._fSampling

    #--------------------------------#
    # reconstruction type

    @property
    def reconType(self):
        return self._reconType

    @reconType.setter
    def reconType(self, value):
        logging.info('  Property    "reconType"         : %s ', value)
        self._reconType = value

    @reconType.deleter
    def reconType(self):
        del self._reconType

    #--------------------------------#
    # delay in samples

    @property
    def delayInSamples(self):
        return self._delayInSamples

    @delayInSamples.setter
    def delayInSamples(self, value):
        logging.info('  Property    "delayInSamples"    : %.4f ', value)
        self._delayInSamples = value

    @delayInSamples.deleter
    def delayInSamples(self):
        del self._delayInSamples
    
    #--------------------------------#
    # wavelengths

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        logging.info('  Property    "wavelengths"       : {} nm'.format(', '.join(map(str, value))))
        self._wavelengths   = value
        self.numWavelengths = len(value)

    @wavelengths.deleter
    def wavelengths(self):
        del self._wavelengths
    
    #--------------------------------#
    # number of wavelengths

    @property
    def numWavelengths(self):
        return self.__numWavelengths

    @numWavelengths.setter
    def numWavelengths(self, value):
        self.__numWavelengths = value

    @numWavelengths.deleter
    def numWavelengths(self):
        del self.__numWavelengths
    
    #--------------------------------#
    # number of repetitions

    @property
    def numRepetitions(self):
        return self.__numRepetitions

    @numRepetitions.setter
    def numRepetitions(self, value):
        self.__numRepetitions = value

    @numRepetitions.deleter
    def numRepetitions(self):
        del self.__numRepetitions

    #--------------------------------#
    # low cutoff frequency of filter

    @property
    def lowCutOff(self):
        return self._lowCutOff

    @lowCutOff.setter
    def lowCutOff(self, value):
        logging.info('  Property    "lowCutOff"         : %d Hz', value)
        self._lowCutOff = value

    @lowCutOff.deleter
    def lowCutOff(self):
        del self._lowCutOff

    #--------------------------------#
    # high cutoff frequency of filter

    @property
    def highCutOff(self):
        return self._highCutOff

    @highCutOff.setter
    def highCutOff(self, value):
        logging.info('  Property    "highCutOff"        : %d Hz', value)
        self._highCutOff = value

    @highCutOff.deleter
    def highCutOff(self):
        del self._highCutOff

    #--------------------------------#
    # order of filter

    @property
    def fOrder(self):
        return self._fOrder

    @fOrder.setter
    def fOrder(self, value):
        logging.info('  Property    "fOrder"            : %d ', value)
        self._fOrder = value

    @fOrder.deleter
    def fOrder(self):
        del self._fOrder

    #-------------------------------#
    #---------- functions ----------#
    #-------------------------------#

    #-------------------------------#
    # reconstruction function

    def reconBP(self, sigMat):
        """
        Backprojection reconstruction function on CPU for optoacoustic imaging

        :param sigMat:  Array that contains signals

        :return:        4D reconstructed image (height x width x wavelengths x repetition)
        """
        logging.info('  Function    "reconBP"           : %s', __name__)

        from pyoat import sigMatFilter, sigMatNormalize

        if np.ndim(sigMat) == 2:
            sigMat = np.expand_dims(sigMat, axis=2)

        pixelNumber         = self.pixelNumber
        xSensor             = self.xSensor
        ySensor             = self.ySensor
        fSampling           = self.fSampling
        self.numRepetitions = int(np.ceil(np.shape(sigMat)[2]/self.numWavelengths))
        self.nSamples       = np.shape(sigMat)[0]

        # filter sigMat
        sigMatF         = (-1)*sigMatFilter(sigMat, self.lowCutOff, self.highCutOff, fSampling, self.fOrder, 0.5)
        
        # normalize mean of sigMat around 0
        sigMatN         = sigMatNormalize(sigMatF)

        #++++++++++++++++++++++++++++++++#
        # beginning of reconstruction

        print('***** reconstruction *****')
        startTime       = time.time()

        timePoints      = np.linspace(0, (self.nSamples)/fSampling, self.nSamples) + self.delayInSamples/fSampling

        # reconstructed image (output of this function)
        imageRecon = np.zeros((pixelNumber, pixelNumber, self.numWavelengths, self.numRepetitions))

        # length of one pixel
        Dxy = self.fieldOfView/(pixelNumber-1)
        
        # define imaging grid
        x = np.linspace(((-1)*(pixelNumber/2-0.5)*Dxy),((pixelNumber/2-0.5)*Dxy),pixelNumber)
        y = np.linspace(((-1)*(pixelNumber/2-0.5)*Dxy),((pixelNumber/2-0.5)*Dxy),pixelNumber)
        meshX, meshY = np.meshgrid(x,y)

        # loop through repetitions
        for repInd in range(0, self.numRepetitions):
            
            # loop through wavelengths
            for waveInd in range(0, self.numWavelengths):

                # loop through all transducer elements
                for sensorInd in range(0,len(xSensor)):

                    # take corresponding signal for transducer element
                    singleSignal    = sigMatN[:,sensorInd,[(repInd*self.numWavelengths)+waveInd]]

                    # calculate derivative of the signal for 'derivative' and 'full' methods
                    diffSignal      = np.concatenate((singleSignal[1:]-singleSignal[0:-1], [[0]]), axis=0)
                    derSignal       = np.multiply(diffSignal, np.expand_dims(timePoints, axis=1))*fSampling

                    # distance of detector to image grid
                    distX           = meshX - xSensor[sensorInd]
                    distY           = meshY - ySensor[sensorInd]
                    dist            = np.sqrt(distX**2 + distY**2)

                    # find corresponding sample value for distance
                    timeSample      = np.ceil((dist*fSampling)/self.speedOfSound - self.delayInSamples)
                    timeSample      = timeSample.astype(int)

                    # apply number of samples bounds
                    timeSample[timeSample<=0]               = 0
                    timeSample[timeSample>=self.nSamples-2] = self.nSamples-3
                    
                    # reconstruct image based on the defined method
                    if self.reconType == 'direct':
                        imageRecon[:,:,waveInd,repInd]    = imageRecon[:,:,waveInd,repInd] + np.squeeze(singleSignal[timeSample])
                    elif self.reconType == 'derivative':
                        imageRecon[:,:,waveInd,repInd]    = imageRecon[:,:,waveInd,repInd] - np.squeeze(derSignal[timeSample])
                    elif self.reconType == 'full':
                        imageRecon[:,:,waveInd,repInd]    = imageRecon[:,:,waveInd,repInd] + np.squeeze(singleSignal[timeSample] - derSignal[timeSample])
                    else:
                        imageRecon[:,:,waveInd,repInd]    = imageRecon[:,:,waveInd,repInd] + np.squeeze(singleSignal[timeSample] - derSignal[timeSample])

        endTime = time.time()
        print('time elapsed: %.2f' %(endTime-startTime))
        
        # end of reconstruction
        #++++++++++++++++++++++++++++++++#
        
        return imageRecon

#-----------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------- model based --------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#

# recon model based class
class cpuMB():
    """
    Model based reconstruction class on CPU for optoacoustic imaging

    :param fieldOfView:     Field of view in x and y direction in meters (default: 0.03)
    :param nAngles:         Number of angles used in model
    :param pixelNumber:     Number of pixels in image (default: 128)
    :param xSensor:         X position of transducer elements (default: Ring array positions)
    :param ySensor:         Y position of transducer elements (default: Ring array positions)
    :param rSensor:         Radius of transducer array
    :param angleSensor:     Angles of individual transducer elements
    :param cupType:         Array type used in experiment (default: Ring array)
    :param speedOfSound:    Estimated speed of sound (m/s) based on temperature (default: 1540 m/s)
    :param nSamples:        Number of samples in sigMat (default: 2032)
    :param fSampling:       Sampling frequency of signals (default: 40e6)
    :param delayInSamples:  Active delay of signals (default: 61)
    :param wavelengths:     Wavelengths used in illumination (default: 800 nm)
    :param numWavelengths:  Number of wavelengths used in acquisition
    :param numRepetitions:  Number of repetitions
    :param lowCutOff:       Low cut off frequency of bandpass filter in MHz (default: 0.1e6 MHz)
    :param highCutOff:      High cut off frequency of bandpass filter in MHz (default: 6e6 MHz)
    :param fOrder:          Butterworth filter order
    :param numIteration:    Number of iterations in minimization function
    :param regMethod:       Regularization method

    :return:                cpuMB object
    """

    # initialization for class cpuBP
    def __init__(self):
        logging.info('  Class       "cpuMB"             : %s', __name__)

        self._fieldOfView       = 0.03
        self._nAngles           = 256
        self._pixelNumber       = 128
        self._xSensor           = {}
        self._ySensor           = {}
        self._rSensor           = {}
        self._angleSensor       = {}
        self._cupType           = 'ring'
        self._speedOfSound      = 1540
        self._nSamples          = 2032
        self._fSampling         = 40e6
        self._delayInSamples    = 0
        self._wavelengths       = [800]
        self._numWavelengths    = 1
        self._numRepetitions    = 1
        self._lowCutOff         = 0.1e6
        self._highCutOff        = 6e6
        self._fOrder            = 3
        self._numIterations     = 5
        self._regMethod         = None
        self._lambdaReg         = 15e6

    #--------------------------------#
    #---------- properties ----------#
    #--------------------------------#

    #--------------------------------#
    # field of view

    @property
    def fieldOfView(self):
        return self._fieldOfView

    @fieldOfView.setter
    def fieldOfView(self, value):
        logging.info('  Property    "fieldOfView"       : %.4f m', value)
        self._fieldOfView = value

    @fieldOfView.deleter
    def fieldOfView(self):
        del self._fieldOfView

    #--------------------------------#
    # number of angles

    @property
    def nAngles(self):
        return self._nAngles

    @nAngles.setter
    def nAngles(self, value):
        self._nAngles = value

    @nAngles.deleter
    def nAngles(self):
        del self._nAngles

    #--------------------------------#
    # pixel number

    @property
    def pixelNumber(self):
        return self._pixelNumber

    @pixelNumber.setter
    def pixelNumber(self, value):
        logging.info('  Property    "pixelNumber"       : %d ', value)
        self._pixelNumber   = value
        self._nAngles       = 2*value

    @pixelNumber.deleter
    def pixelNumber(self):
        del self._pixelNumber

    #--------------------------------#
    # x positions of sensor

    @property
    def xSensor(self):
        return self._xSensor

    @xSensor.setter
    def xSensor(self, value):
        self._xSensor = value

    @xSensor.deleter
    def xSensor(self):
        del self._xSensor
    
    #--------------------------------#
    # y positions of sensor

    @property
    def ySensor(self):
        return self._ySensor

    @ySensor.setter
    def ySensor(self, value):
        self._ySensor = value

    @ySensor.deleter
    def ySensor(self):
        del self._ySensor

    #--------------------------------#
    # radius of sensor

    @property
    def rSensor(self):
        return self._rSensor

    @rSensor.setter
    def rSensor(self, value):
        self._rSensor = value

    @rSensor.deleter
    def rSensor(self):
        del self._rSensor
    
    #--------------------------------#
    # sensor angle

    @property
    def angleSensor(self):
        return self._angleSensor

    @angleSensor.setter
    def angleSensor(self, value):
        self._angleSensor = value

    @angleSensor.deleter
    def angleSensor(self):
        del self._angleSensor
    
    #--------------------------------#
    # cup type

    @property
    def cupType(self):
        return self._cupType

    @cupType.setter
    def cupType(self, value):
        logging.info('  Property    "cupType"           : %s', value)
        self._cupType       = value
        self.__arrayDir     = pkgr.resource_filename('pyoat', 'arrays/'+self._cupType+'Cup.mat')
        self.__arrayData    = h5py.File(self.__arrayDir, 'r')
        self.xSensor        = self.__arrayData['transducerPos'][0,:]
        self.ySensor        = self.__arrayData['transducerPos'][1,:]
        self.rSensor        = np.sqrt(self.xSensor**2 + self.ySensor**2)
        self.angleSensor    = np.arctan2(self.ySensor,self.xSensor) + 2*math.pi*(np.multiply((self.xSensor>0),(self.ySensor<0)))

    @cupType.deleter
    def cupType(self):
        del self._cupType
        del self.xSensor
        del self.ySensor
        del self.rSensor

    #--------------------------------#
    # speed of sound

    @property
    def speedOfSound(self):
        return self._speedOfSound

    @speedOfSound.setter
    def speedOfSound(self, value):
        logging.info('  Property    "speedOfSound"      : %d m/s', value)
        self._speedOfSound = value

    @speedOfSound.deleter
    def speedOfSound(self):
        del self._speedOfSound

    #--------------------------------#
    # number of samples

    @property
    def nSamples(self):
        return self._nSamples

    @nSamples.setter
    def nSamples(self, value):
        logging.info('  Property    "nSamples"          : %d ', value)
        self._nSamples = value

    @nSamples.deleter
    def nSamples(self):
        del self._nSamples

    #--------------------------------#
    # sampling frequency

    @property
    def fSampling(self):
        return self._fSampling

    @fSampling.setter
    def fSampling(self, value):
        logging.info('  Property    "fSampling"         : %d Hz', value)
        self._fSampling = value

    @fSampling.deleter
    def fSampling(self):
        del self._fSampling

    #--------------------------------#
    # delay in samples

    @property
    def delayInSamples(self):
        return self._delayInSamples

    @delayInSamples.setter
    def delayInSamples(self, value):
        logging.info('  Property    "delayInSamples"    : %.4f ', value)
        self._delayInSamples = value

    @delayInSamples.deleter
    def delayInSamples(self):
        del self._delayInSamples
    
    #--------------------------------#
    # wavelengths

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        logging.info('  Property    "wavelengths"       : {} nm'.format(', '.join(map(str, value))))
        self._wavelengths   = value
        self.numWavelengths = len(value)


    @wavelengths.deleter
    def wavelengths(self):
        del self._wavelengths
    
    #--------------------------------#
    # number of wavelengths

    @property
    def numWavelengths(self):
        return self._numWavelengths

    @numWavelengths.setter
    def numWavelengths(self, value):
        self._numWavelengths = value

    @numWavelengths.deleter
    def numWavelengths(self):
        del self._numWavelengths
    
    #--------------------------------#
    # number of repetitions

    @property
    def numRepetitions(self):
        return self._numRepetitions

    @numRepetitions.setter
    def numRepetitions(self, value):
        self._numRepetitions = value

    @numRepetitions.deleter
    def numRepetitions(self):
        del self._numRepetitions

    #--------------------------------#
    # low cutoff frequency of filter

    @property
    def lowCutOff(self):
        return self._lowCutOff

    @lowCutOff.setter
    def lowCutOff(self, value):
        logging.info('  Property    "lowCutOff"         : %d Hz', value)
        self._lowCutOff = value

    @lowCutOff.deleter
    def lowCutOff(self):
        del self._lowCutOff

    #--------------------------------#
    # high cutoff frequency of filter

    @property
    def highCutOff(self):
        return self._highCutOff

    @highCutOff.setter
    def highCutOff(self, value):
        logging.info('  Property    "highCutOff"        : %d Hz', value)
        self._highCutOff = value

    @highCutOff.deleter
    def highCutOff(self):
        del self._highCutOff

    #--------------------------------#
    # order of filter

    @property
    def fOrder(self):
        return self._fOrder

    @fOrder.setter
    def fOrder(self, value):
        logging.info('  Property    "fOrder"            : %d ', value)
        self._fOrder = value

    @fOrder.deleter
    def fOrder(self):
        del self._fOrder

    #--------------------------------#
    # iteration number

    @property
    def numIterations(self):
        return self._numIterations

    @numIterations.setter
    def numIterations(self, value):
        logging.info('  Property    "numIterations"     : %d ', value)
        self._numIterations = value

    @numIterations.deleter
    def numIterations(self):
        del self._numIterations

    #--------------------------------#
    # regularization method

    @property
    def regMethod(self):
        return self._regMethod

    @regMethod.setter
    def regMethod(self, value):
        logging.info('  Property    "regMethod"         : %s ', value)
        self._regMethod = value

    @regMethod.deleter
    def regMethod(self):
        del self._regMethod

    #--------------------------------#
    # regularization method

    @property
    def lambdaReg(self):
        return self._lambdaReg

    @lambdaReg.setter
    def lambdaReg(self, value):
        logging.info('  Property    "lambdaReg"         : %s ', value)
        self._lambdaReg = value

    @lambdaReg.deleter
    def lambdaReg(self):
        del self._lambdaReg

    #-------------------------------#
    #---------- functions ----------#
    #-------------------------------#

    #-------------------------------#
    # model matrix calculation

    def calculateModelMatrix(self):
        """
        Model matrix calculator function on CPU for optoacoustic imaging

        Calls modelOA class

        :return: Optoacoustic model matrix
        """
        logging.info('  Function    "calculateModelMatrix"  : %s', __name__)

        from pyoat import modelOA

        print('***********************************')
        print('*** Calculating model matrix... ***')

        # initialize optoacoustic model
        mOA                 = modelOA()

        # set OA model parameters
        mOA.fieldOfView     = self.fieldOfView
        mOA.pixelNumber     = self.pixelNumber
        mOA.speedOfSound    = self.speedOfSound
        mOA.nSamples        = self.nSamples
        mOA.fSampling       = self.fSampling
        mOA.regMethod       = self.regMethod
        mOA.rSensor         = self.rSensor
        mOA.angleSensor     = self.angleSensor
        mOA.nAngles         = self.nAngles
        mOA.delayInSamples  = self.delayInSamples
        
        timePoints      = np.linspace(0, (self.nSamples-2)/self.fSampling, self.nSamples-2) + self.delayInSamples/self.fSampling

        modelMatrix         = mOA.calculateModel(timePoints)
        
        return modelMatrix

    #-------------------------------#
    # recon matrix calculation

    def calculateReconMatrix(self, modelMatrix):
        """
        Merge model matrix and regularization matrix

        :return: Recon matrix which is combination of model matrix and regularization matrix
        """
        logging.info('  Function    "calculateReconMatrix"  : %s', __name__)

        from pyoat import modelOA

        print('***** Merging matrices *****')

        # initialize optoacoustic model
        mOA                 = modelOA()

        # set regularization parameters
        mOA.pixelNumber     = self.pixelNumber
        mOA.lambdaReg       = self.lambdaReg

        if self.regMethod == 'tikonov':
            regMatrix       = mOA.calculateRegularizationMatrix()
            reconMatrix     = vstack((modelMatrix, regMatrix))
        else:
            reconMatrix     = modelMatrix
        
        return reconMatrix

    #-------------------------------#
    # reconstruction function

    def recon(self, sigMat, reconMatrix):
        """
        Model based reconstruction function on CPU for optoacoustic imaging

        :param sigMat:          Array that contains signals
        :param reconMatrix:     Combined model matrix and regularization matrix

        :return: 4D reconstructed image (height x width x wavelengths x repetition)
        """
        logging.info('  Function    "recon"             : %s', __name__)

        from pyoat import sigMatFilter, sigMatNormalize

        if np.ndim(sigMat) == 2:
            sigMat = np.expand_dims(sigMat, axis=2)

        pixelNumber         = self.pixelNumber
        self.numRepetitions = int(np.ceil(np.shape(sigMat)[2]/self.numWavelengths))

        # filter sigMat
        sigMatF         = (-1)*sigMatFilter(sigMat, self.lowCutOff, self.highCutOff, self.fSampling, self.fOrder, 0.5)
        
        # normalize mean of sigMat around 0
        sigMatN         = sigMatNormalize(sigMatF)

        # reconstructed image (output of this function)
        imageRecon = np.zeros((pixelNumber, pixelNumber, self.numWavelengths, self.numRepetitions))

        #++++++++++++++++++++++++++++++++#
        # beginning of reconstruction

        print('***** reconstruction *****')
        startTime       = time.time()
        
        # loop through repetitions
        for repInd in range(0, self.numRepetitions):
            
            # loop through wavelengths
            for waveInd in range(0, self.numWavelengths):
            
                sigMatVec       = np.expand_dims(np.transpose(sigMatN[:,:,[(repInd*self.numWavelengths)+waveInd]]).reshape(-1),axis=1)

                if self.regMethod == 'tikonov':
                    bVec        = np.concatenate((sigMatVec, np.zeros((pixelNumber*pixelNumber, 1)) ))
                else:
                    bVec        = sigMatVec

                recon, reasonTerm, iterNum, normR = lsqr(reconMatrix, bVec, iter_lim=self.numIterations)[:4]

                imageRecon[:,:,waveInd, repInd]    = np.reshape(recon, (pixelNumber, pixelNumber))

        endTime = time.time()
        print('time elapsed: %.2f' %(endTime-startTime))
        
        # end of reconstruction
        #++++++++++++++++++++++++++++++++#

        return imageRecon
