#-----
# Description   : Forward model for optoacoustic
# Date          : March 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- optoacoustic model ----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class modelOA():

    # initialization for class modelOA
    def __init__(self):

        self._fieldOfView       = 0.03
        self._nAngles           = 256
        self._pixelNumber       = 128
        self._rSensor           = {}
        self._angleSensor       = {}
        self._speedOfSound      = 1540
        self._nSamples          = 2032
        self._fSampling         = 40e6
        self._delayInSamples    = 61
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
        self._pixelNumber   = value
        self._nAngles       = 2*value 

    @pixelNumber.deleter
    def pixelNumber(self):
        del self._pixelNumber

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
    # speed of sound

    @property
    def speedOfSound(self):
        return self._speedOfSound

    @speedOfSound.setter
    def speedOfSound(self, value):
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
        self._delayInSamples = value

    @delayInSamples.deleter
    def delayInSamples(self):
        del self._delayInSamples

    #--------------------------------#
    # lambda for regularization

    @property
    def lambdaReg(self):
        return self._lambdaReg

    @lambdaReg.setter
    def lambdaReg(self, value):
        self._lambdaReg = value

    @lambdaReg.deleter
    def lambdaReg(self):
        del self._lambdaReg
    
    #-------------------------------#
    #---------- functions ----------#
    #-------------------------------#

    #-------------------------------#
    # calculate projections
    
    def calculateProjection(self,xPoint,yPoint,rPoint,theta,nRows,i):
        
        resolutionXY        = self.pixelNumber
        reconDimsXY         = self.fieldOfView

        nCols               = resolutionXY*resolutionXY
        lt                  = np.shape(xPoint)[1]                 # length of the time vector
        nAngles             = np.shape(xPoint)[0]                 # number of points on the curve
        pixelSize           = reconDimsXY/(resolutionXY-1)        # sampling distance in x and y
        
        ##### map points to original grid #####
        xPointUnrotated     = xPoint*np.cos(theta) - yPoint*np.sin(theta); # horizontal position of the points of the curve in the original grid (not rotated)
        yPointUnrotated     = xPoint*np.sin(theta) + yPoint*np.cos(theta); # vertical position of the points of the curve in the original grid (not rotated)
        
        # pad zeros to x and y positions
        xPaddedRight    = np.concatenate((xPointUnrotated, np.zeros((1,lt))))
        yPaddedRight    = np.concatenate((yPointUnrotated, np.zeros((1,lt))))
        xPaddedLeft     = np.concatenate((np.zeros((1,lt)), xPointUnrotated))
        yPaddedLeft     = np.concatenate((np.zeros((1,lt)), yPointUnrotated))
        
        distToOrigin    = np.sqrt(np.power((xPaddedRight - xPaddedLeft),2) + np.power((yPaddedRight - yPaddedLeft),2))
        distToOrigin    = distToOrigin[1:nAngles,:]
        del xPaddedRight, xPaddedLeft, yPaddedRight, yPaddedLeft
        
        vecIntegral = (1/2)*np.divide((np.concatenate((distToOrigin, np.zeros((1,lt))))+np.concatenate((np.zeros((1,lt)), distToOrigin))),rPoint) # vector for calculating the integral
        del distToOrigin
        
        xNorm = (xPointUnrotated + (reconDimsXY/2))/pixelSize+1 # horizontal position of the points of the curve in normalized coordinates
        del xPointUnrotated
        
        yNorm = (yPointUnrotated + (reconDimsXY/2))/pixelSize+1 # vertical position of the points of the curve in normalized coordinates
        del yPointUnrotated
        
        xBefore = np.floor(xNorm)   # horizontal position of the point of the grid at the left of the point (normalized coordinates)
        xAfter  = np.floor(xNorm+1) # horizontal position of the point of the grid at the right of the point (normalized coordinates)
        xDiff   = xNorm-xBefore
        del xNorm

        yBefore = np.floor(yNorm)      # vertical position of the point of the grid below of the point (normalized coordinates)
        yAfter  = np.floor(yNorm+1)    # vertical position of the point of the grid above of the point (normalized coordinates)
        yDiff   = yNorm - yBefore
        del yNorm
        
        ##### define square #####

        # position of the first point of the square
        xSquare1 = xBefore.astype(int)
        ySquare1 = yBefore.astype(int)
        
        # position of the second point of the square
        xSquare2 = xAfter.astype(int)
        ySquare2 = yBefore.astype(int)
        
        # position of the third point of the square
        xSquare3 = xBefore.astype(int)
        ySquare3 = yAfter.astype(int)
        
        # position of the fourth point of the square
        xSquare4 = xAfter.astype(int)
        ySquare4 = yAfter.astype(int)
        
        ##### decide points are inside or outside of the rectangle #####
        inPoint1 = (xSquare1>0) & (xSquare1<=resolutionXY) & (ySquare1>0) & (ySquare1<=resolutionXY) # boolean to decide 1. point of the square is inside the grid
        inPoint2 = (xSquare2>0) & (xSquare2<=resolutionXY) & (ySquare2>0) & (ySquare2<=resolutionXY) # boolean to decide 2. point of the square is inside the grid
        inPoint3 = (xSquare3>0) & (xSquare3<=resolutionXY) & (ySquare3>0) & (ySquare3<=resolutionXY) # boolean to decide 3. point of the square is inside the grid
        inPoint4 = (xSquare4>0) & (xSquare4<=resolutionXY) & (ySquare4>0) & (ySquare4<=resolutionXY) # boolean to decide 4. point of the square is inside the grid

        inVec1 = np.transpose(inPoint1).reshape(1, -1) # convert to vector
        inVec2 = np.transpose(inPoint2).reshape(1, -1) # convert to vector
        inVec3 = np.transpose(inPoint3).reshape(1, -1) # convert to vector
        inVec4 = np.transpose(inPoint4).reshape(1, -1) # convert to vector
        
        ##### define points on grid #####
        pos1 = resolutionXY*(xSquare1-1)+ySquare1 # one dimensional position of the first points of the squares in the grid
        pos2 = resolutionXY*(xSquare2-1)+ySquare2 # one dimensional position of the first points of the squares in the grid
        pos3 = resolutionXY*(xSquare3-1)+ySquare3 # one dimensional position of the first points of the squares in the grid
        pos4 = resolutionXY*(xSquare4-1)+ySquare4 # one dimensional position of the first points of the squares in the grid
        del xSquare1, xSquare2, xSquare3, xSquare4, ySquare1, ySquare2, ySquare3, ySquare4
        
        ##### convert to vector format #####
        posVec1 = np.transpose(pos1).reshape(1, -1) # Pos_triang_1_t in vector form
        posVec2 = np.transpose(pos2).reshape(1, -1) # Pos_triang_1_t in vector form
        posVec3 = np.transpose(pos3).reshape(1, -1) # Pos_triang_1_t in vector form
        posVec4 = np.transpose(pos4).reshape(1, -1) # Pos_triang_1_t in vector form
        del pos1, pos2, pos3, pos4
        
        weight1 = (1-xDiff)*(1-yDiff)*vecIntegral # weight of the first point of the triangle
        weight2 = (xDiff)*(1-yDiff)*vecIntegral # weight of the second point of the triangle
        weight3 = (1-xDiff)*(yDiff)*vecIntegral # weight of the third point of the triangle
        weight4 = (xDiff)*(yDiff)*vecIntegral # weight of the fourth point of the triangle
        
        weightVec1 = np.transpose(weight1).reshape(1, -1) # weight_sq_1 in vector form
        weightVec2 = np.transpose(weight2).reshape(1, -1) # weight_sq_1 in vector form
        weightVec3 = np.transpose(weight3).reshape(1, -1) # weight_sq_1 in vector form
        weightVec4 = np.transpose(weight4).reshape(1, -1) # weight_sq_1 in vector form
        del weight1, weight2, weight3, weight4
        
        rowMatrix       = np.transpose( np.transpose(np.expand_dims(np.linspace(1,lt,lt, dtype=int),axis=0)) * np.ones((1,nAngles), dtype=int) ) # rows of the sparse matrix
        rowMatrixVec    = np.transpose(np.transpose(rowMatrix).reshape(-1, 1))-1 # rows of the sparse matrix in vector form
        del rowMatrix

        rowMat              = np.concatenate((rowMatrixVec[inVec1]+(i*lt), rowMatrixVec[inVec2]+(i*lt),                                                 rowMatrixVec[inVec3]+(i*lt), rowMatrixVec[inVec4]+(i*lt)))

        posMat              = np.concatenate((posVec1[inVec1]-1, posVec2[inVec2]-1, posVec3[inVec3]-1,                                                  posVec4[inVec4]-1))

        weightMat           = np.concatenate((weightVec1[inVec1], weightVec2[inVec2], weightVec3[inVec3],                                               weightVec4[inVec4]))

        projectionMatrix    = csc_matrix((weightMat, (rowMat, posMat)), shape=(nRows, nCols))
        
        return projectionMatrix


    #-------------------------------#
    # calculate model matrix

    def calculateModel(self,timePoints):

        resolutionXY    = self.pixelNumber
        reconDimsXY     = self.fieldOfView
        speedOfSound    = self.speedOfSound
        rSensor         = self.rSensor
        angleSensor     = self.angleSensor
        nAngles         = self.nAngles

        nCols           = resolutionXY*resolutionXY                 # number of columns of the matrix
        nRows           = len(timePoints)*len(angleSensor)          # number of rows of the matrix
        pixelSize       = reconDimsXY/(resolutionXY-1)              # one pixel size
        dt              = 1e-15                                     # diferential of time to perform derivation
        tPlusdt         = timePoints+dt                             # time instants for t+dt
        tMinusdt        = timePoints-dt                             # time instants for t-dt
        
        # max angle required to cover all grid for each of the transducers
        angleMax = np.arcsin(((reconDimsXY+2*pixelSize)*np.sqrt(2))/(2*np.amin(rSensor)))
        
        minusDistSensor     = speedOfSound*tMinusdt
        plusDistSensor      = speedOfSound*tPlusdt

        angles              = np.transpose(np.expand_dims(np.linspace(-angleMax,angleMax,nAngles),axis=0))*np.ones((1,len(timePoints)))
        
        for i in range(0,len(angleSensor)):
            
            print('Projection Number: {}'.format(i+1))
            
            theta               = angleSensor[i]                        # angle to (0,0) point
            
            rMinus              = np.ones((nAngles,1))*minusDistSensor  # -t distance from sensor to curve
            rPlus               = np.ones((nAngles,1))*plusDistSensor   # +t distance from sensor to curve
            
            xMinust             = rSensor[i]-(rMinus)*np.cos(angles)   # x distance at -t based on (0,0) to transducer coordinate system
            yMinust             = (rMinus)*np.sin(angles)              # y distance at +t based on (0,0) to transducer coordinate system
            
            xPlust              = rSensor[i]-(rPlus)*np.cos(angles)    # x distance at +t based on (0,0) to transducer coordinate system
            yPlust              = (rPlus)*np.sin(angles)               # y distance at +t based on (0,0) to transducer coordinate system

            projectionMinust    = self.calculateProjection(xMinust, yMinust, rMinus, theta, nRows, i)
            projectionPlust     = self.calculateProjection(xPlust, yPlust, rPlus, theta, nRows, i)

            if i > 0:
                modelMatrix     = modelMatrix + (1/(2*dt))*(projectionPlust - projectionMinust)
            else:
                modelMatrix     = (1/(2*dt))*(projectionPlust - projectionMinust)
        
        # clear variables
        del xMinust, yMinust, rMinus, xPlust, yPlust, rPlus
        
        return modelMatrix

    #-------------------------------#
    # calculate regularization matrix

    def calculateRegularizationMatrix(self):

        pixelNumber     = self.pixelNumber
        lambdaReg       = self.lambdaReg
    
        nRows       = pixelNumber*pixelNumber
        nCols       = pixelNumber*pixelNumber
        rows        = np.linspace(0,nRows-1, nRows)
        cols        = np.linspace(0,nCols-1, nCols)
        
        matrixVal   = np.ones((nRows,))*lambdaReg
        regMatrix   = csc_matrix((matrixVal, (rows, cols)), shape=(nRows, nCols))
        
        return regMatrix

