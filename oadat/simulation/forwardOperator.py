#-----
# Description   : Function to apply forward model
# Date          : March 2022
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

# import libraries
import logging
import numpy as np

def forward(inputImage, modelMatrix, nSamples=2030, autoResize=True):
	"""
	Apply forward model on the given image
	
	:param inputImage: image to apply forward model
	:param modelMatrix: model to apply on the image
	:param nSamples: number of time points in acquisition
	:param autoResize: resize image to match model matrix size
	
	:return: signal matrix (sigMat)
	"""
	logging.info('  Function    "forward"      		: %s', __name__)
	
	#++++++++++++++++++++++++++++++++#
	# forward operator function

	# pixel number in model matrix
	numElements = int(np.shape(modelMatrix)[0]/nSamples)

	# pixel number in model matrix
	pixelNumberMatrix = int(np.sqrt(np.shape(modelMatrix)[1]))

	# pixel number in image
	pixelNumberImage = np.shape(inputImage)[0]

	# resize image
	if pixelNumberMatrix != pixelNumberImage:
		if autoResize:
			inputImage = np.resize(inputImage, (pixelNumberMatrix, pixelNumberMatrix))
		else:
			raise AssertionError('Sizes of image and model matrix do not match!')

	# flatten the image
	inputImage 		= np.array(inputImage)
	imageFlat		= inputImage.flatten()

	# multiply with model matrix
	sigMatVec 	= modelMatrix*imageFlat
	sigMat 		= np.transpose(sigMatVec.reshape(numElements, -1))

	return sigMat