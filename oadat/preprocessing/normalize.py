#-----
# Description   : Normalization around mean value of each channel
# Date          : February 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

import logging
import time
import numpy as np

def sigMatNormalize(sigMatIn):
    """
    Band pass filtering for signals

    :param sigMatIn: 3D array (samples x channels x repetition) of signals

    :return: 3D array normalized around mean value of the channels
    """
    logging.info('  Function    "sigMatNormalize"   : %s', __name__)

   
    print('***** normalization *****')
    startTime       = time.time()

    sigMatOut = np.zeros(np.shape(sigMatIn))

    for i in range(np.shape(sigMatIn)[2]):
        singleF             = sigMatIn[:,:,i]
        meanF               = np.mean(singleF, axis=0)
        sigMatOut[:,:,i]    = singleF - np.tile(meanF, (np.shape(singleF)[0], 1))

    endTime = time.time()
    print('time elapsed: %.2f' %(endTime-startTime))

    return sigMatOut