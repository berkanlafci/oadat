#-----
# Description   : Function to average signals in given axis
# Date          : March 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

import logging
import time
import numpy as np

def averageSignals(sigMat, axis=2):
    """
    Function average sigMat channels

    :param sigMat:          Signal array
    :param averageAxis:     Averaging axis of array

    :return:                Averaged signal values
    """
    logging.info('  Function    "averageSignals"    : %s', __name__)

    #++++++++++++++++++++++++++++++++#
    # averaging function

    print('***** averaging *****')
    startTime       = time.time()

    sigMatAverage = np.sum(sigMat, axis=axis)

    endTime = time.time()
    print('time elapsed: %.2f' %(endTime-startTime))

    return sigMatAverage

    
