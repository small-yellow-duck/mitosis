'''
readme

datasets = make_train_valid()
model = do_dnn1(datasets, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400)
#or reload model from pkl
#model = load_pkl_model('model_batchsize400.pkl')

#generate probability map for mitoses in each image (or load the map from the pkl file)
#proba_map_train0 = do_pred(model, datasets[0], 0, 'dnn1')
#proba_map_train0 = load_pkl_model('sample_proba_map_train0.pkl')
im3 = Image.fromarray(255.0*proba_map_train0)
im3.show()

#generate coords for training images for dnn2
dataset2 = construct_dataset2(datasets)
model2 = do_dnn2(datasets2, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400)

#generate probability maps for validation data
proba_map[0,:, :] = int(np.round( do_pred(model, datasets[1], 0, 'dnn2') ))
do_pred_all(datasets)

'''

import matplotlib
import pandas as pd
import numpy as np
import os
import glob
import sys
import time
import copy
import pickle

import theano
import theano.tensor as T
import convolutional_mlp as c_mlp
import logistic_sgd
from theano.tensor.nnet import conv


from PIL import Image
#from matplotlib import pyplot as plt

import glob

#311 images in total
#tot = count_imgs(range(1,13))
def count_imgs(patients, quiet=True):
	tot = 0
	for patient in patients:
		patient = str(patient)
		if len(patient) == 1:
			patient = '0'+patient
		dir = 'mitos_img/'+ patient + '/'
		files = glob.glob(dir+'*.tif')
		if not quiet:
			print patient, len(files)	
		tot = tot+len(files)
	if not quiet:	
		print tot
	return tot
	

#550 mitoses in total
#tot = count_mitoses(range(1,13))
def count_mitoses(patients, quiet=True):
	tot = 0
	for patient in patients:
		totp = 0
		patient = str(patient)
		if len(patient) == 1:
			patient = '0'+patient
		dir = 'mitos_img/'+ patient + '/'
		files = glob.glob(dir+'*.csv')
		
		for file in files:
			d = np.loadtxt(file, delimiter=',')
			d = d.reshape((-1,2))
			m = d.shape[0]
			totp += m
		if not quiet:	
			print patient, totp
		tot = tot+ totp
	if not quiet:	
		print tot
	return tot
# number of mitoses in all frames for each patient	
# 01 73
# 02 37
# 03 18
# 04 224
# 05 6
# 06 96
# 07 68
# 08 3
# 09 2
# 10 0
# 11 15
# 12 8
# 550
#train: everything except the validate sets 
#valid: 01,09,11, 12		
#98 mitoses in validation, 452 in train


def shared_dataset(data_x, data_y, borrow=True):
	""" Function that loads the dataset into shared variables

	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""

	#data = theano.shared(np.asarray(datasets[0][0][0], dtype=theano.config.floatX), borrow=borrow)
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)						 
	shared_y = theano.shared(np.asarray(data_y,
										   dtype=theano.config.floatX),
							 borrow=borrow)
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return shared_x, T.cast(shared_y, 'int32')
	#return shared_x, shared_y
		
		
#datasets = make_train_valid()
def make_train_valid():
	print 'generating training data'
	#train_img_stack, train_coords, train_labels = gen_Sd(['08'], window_size=101, r=10)
	train_img_stack, train_coords, train_labels = gen_Sd(['02', '03', '04', '05', '06', '07', '08', '10'], window_size=101, r=10)
	print 'generating validation data'
	#valid_img_stack, valid_coords, valid_labels = gen_Sd(['12'], window_size=101, r=10)
	valid_img_stack, valid_coords, valid_labels = gen_Sd(['01', '09', '11', '12'], window_size=101, r=10)

	#test_set_x, test_set_y = shared_dataset(test_set)
	#valid_img_stack, valid_coords, valid_labels = shared_dataset(valid_img_stack[0:1000], valid_coords[0:1000], valid_labels[0:1000])
	#train_img_stack, train_coords, train_labels = shared_dataset(train_img_stack, train_coords, train_labels)
	
	#print 'size train data (Gb) ', train_img_stack.get_value(borrow=True).nbytes/(1000.0**3)
	print 'size train data (Gb) ', train_img_stack.nbytes/(1000.0**3)
	return [[train_img_stack, train_coords, train_labels], [valid_img_stack, valid_coords, valid_labels]]





#generate windows with true positives and an equal number of random windows of true negatives
#window_size for the training set Sd is not given in the paper, so I just picked 45 since it seems to be the upper limit on cell size
#data, labels = gen_Sd(['02'], ['01', '09', '11', '12'], window_size=45, r=10)
#data.dump('Sd_train.pkl')
#labels.dump(Sd_train_labels.pkl')
#valid_img_stack, valid_coords, valid_labels = gen_Sd(['01', '09', '11', '12'], window_size=101, r=10)
#train_img_stack, train_coords, train_labels = gen_Sd(['02', '03', '04', '05', '06', '07', '08', '10'], window_size=51, r=10)
#train_data, train_labels = gen_Sd(['02', '03', '04', '05', '06', '07', '08', '10'], window_size=101, r=10)
def gen_Sd(train_patients, window_size=101, r=10):
	if np.mod(window_size, 2) == 0:
		print 'window size must be odd'
		returnr
		
	r = r-1
	w = (window_size-1)/2
	
#	valid = [7,9, 11,12]
#	train = list(set(range(1,13)) - set(valid))
#	valid = [(i<10)*str('0')+str(i) for i in valid]
#	train = [(i<10)*str('0')+str(i) for i in train]
	
	mitoses_in_train = count_mitoses(train_patients)
	approx_number_positive_mitosis = np.ceil(mitoses_in_train*np.pi*(2*(r+1))**2/4.0)
	#data_positive = np.zeros((approx_number_positive_mitosis, 3, window_size,window_size))
	
	tot_images = count_imgs(train_patients)
	negative_samples_per_image = np.floor(1.0*approx_number_positive_mitosis/tot_images)
	#data_negative = np.zeros((negative_samples_per_image*tot_images, 3, window_size,window_size))
	number_negative_mitosis = negative_samples_per_image*tot_images
	#data = np.zeros((number_negative_mitosis+approx_number_positive_mitosis, 3, window_size,window_size))
	
	img_stack = np.zeros((tot_images, 3, 2000+2*w,2000+2*w), dtype=np.int8)
	#print 'data shape ', data.shape
	print 'window size ', window_size
	print 'total images ', tot_images
	print 'negative_samples_per_image ', negative_samples_per_image
	print 'number_negative_mitosis ', number_negative_mitosis
	print 'number_negative_mitosis+approx_number_positive_mitosis ', number_negative_mitosis+approx_number_positive_mitosis

	coords = np.zeros((number_negative_mitosis+approx_number_positive_mitosis, 5), dtype=np.int16)
	labels = np.zeros(number_negative_mitosis+approx_number_positive_mitosis, dtype=np.int8)
	labels[number_negative_mitosis:] = 1
	
	# windows with true positives
	j = 0
	img_num = 0
	for patient in train_patients:
		dir = 'mitos_img/'+ patient + '/'
	
		csvfiles = glob.glob(dir+'*.csv')
		imgfiles = glob.glob(dir+'*.tif')
	
		#print len(cvsfiles)
		
		for imgfile in imgfiles:
			#print patient, imgfile
			csvfile = imgfile[0:-4]+'.csv'

			im = Image.open(imgfile)
			imarray = np.array(im)			

			neg_coords = np.random.randint(0,imarray.shape[1], 4*2*negative_samples_per_image)
			if neg_coords.shape[0] % 2 == 1:
				neg_coords = neg_coords[0:-1]
			neg_coords = neg_coords.reshape((-1,2)) + w
			#print neg_coords.shape			
						
			#print imarray.shape
			imarray = np.append(imarray[w+1:0:-1, :,:], imarray, axis=0)
			imarray = np.append(imarray, imarray[-1:-w:-1, :,:], axis=0)
			imarray = np.append(imarray[:, w+1:0:-1,:], imarray, axis=1)
			imarray = np.append(imarray, imarray[:,-1:-w:-1,:], axis=1)			
			imarray = np.rollaxis(imarray,-1)
			#print imarray.shape
			img_stack[img_num, :, :, :] = imarray
			

			#generate sub-images containing mitosis events
			if os.path.exists(csvfile):			
				m_pos = np.loadtxt(csvfile, delimiter=',')
				m_pos = m_pos.reshape((-1,2))
				#print 'mpos shape ', m_pos.shape
													
				no_overlaps = np.ones(neg_coords.shape[0])												
				for i in range(m_pos.shape[0]): 
					#print m_pos.shape
					mx = m_pos[i,0] + w#2*w
					my = m_pos[i,1] + w#2*w
					no_overlaps = no_overlaps*( (neg_coords[:,0]-mx)**2 + (neg_coords[:,1]-my)**2 > (r+1)**2)
					
					if False:
						d = imarray[:, mx-w:mx+w+1, my-w:my+w+1]
						img = Image.fromarray(d[1,:,:])
						img.show()	
						raw_input("Press Enter to continue...")
						img.close()
						break	
					
					for x in xrange(-r, r+1):
						for y in xrange( -1-int(np.sqrt(r**2-x**2)), 1+int(np.sqrt(r**2-x**2))+1 ):
							#try:
							
							if np.max([img_num, mx-w+x, mx+w+1+x, my-w+y, my+w+1+y]) <= 2000+w and np.min([img_num, mx-w+x, mx+w+1+x, my-w+y, my+w+1+y]) >= 0:
								coords[number_negative_mitosis+j, :] = [img_num, mx-w+x, mx+w+1+x, my-w+y, my+w+1+y]
								#data[number_negative_mitosis+j, :, :, :] = imarray[:, mx-w+x:mx+w+1+x, my-w+y:my+w+1+y]
								j += 1
								if np.max(coords[number_negative_mitosis+j, :]) > 2000+2*w or np.min(coords[number_negative_mitosis+j, :]) < 0:
									print 'uh oh ', imgfile, mx-w+x, mx+w+1+x, my-w+y, my+w+1+y
							
#							except:
#								print imarray[:, mx-w+x:mx+w+1+x, my-w+y:my+w+1+y].shape
#								print 'window not in image', mx, x, my, y, imarray.shape

				#toss out the randomly generated coords for negative samples if they are actually in positive samples
				neg_coords = neg_coords[no_overlaps!=0]
				
			#generate sub-images containing no mitosis events			
			neg_coords = neg_coords[0:negative_samples_per_image, :]
			#print 'shortfall  ', neg_coords.shape[0], negative_samples_per_image																								
			for i in range(neg_coords.shape[0]):	
				mx = neg_coords[i,0]
				my = neg_coords[i,1] 
				#data[i+img_num*negative_samples_per_image, :, :, :] = imarray[:, mx-w:mx+w+1, my-w:my+w+1] 
				coords[i+img_num*negative_samples_per_image, :] = [img_num, mx-w, mx+w+1, my-w, my+w+1]

			img_num += 1

	print 'number neg mitosis ', number_negative_mitosis, img_num*negative_samples_per_image
	print 'negative number empty ', np.sum(np.sum(coords[0:int(number_negative_mitosis)], axis=1) == 0)
	print 'positive number empty ', np.sum(np.sum(coords[int(number_negative_mitosis):int(number_negative_mitosis+ j-1)], axis=1) == 0)
	print 'j -1', j-1, number_negative_mitosis
	data_order = range(int(j-1 + number_negative_mitosis))
	np.random.shuffle(data_order)
				
#	calculate the number of chunks to break the data into (1.4 Gb apiece?)
#	randomly reorder the data in the chunks
#	write chunks to memory
#	total_size = data[0:(j-1 + number_negative_mitosis), :, :, :].nbytes/(1024**3)
#	n_chunks = int(np.ceil(total_size/1.4))
#	chunk_size = int(1.0*(j-1 + number_negative_mitosis)/n_chunks)
#	for i in xrange(n_chunks-1):
#		print 'writing chunk ', i
#		start = int(i*chunk_size)
#		end = int(np.min([(i+1)*chunk_size, (j-1 + number_negative_mitosis)]))
#		l = data_order[start:end]
#		data[l,:,:,:].dump('Sd/Sd_train_chunk_data_'+str(i)+'.pkl')
#		labels[l].dump('Sd/Sd_train_chunk_labels_'+str(i)+'.pkl')
	
#	return data[data_order, :, :, :],  labels[data_order]
	return img_stack, coords[data_order, :],  labels[data_order]


#data = make_chunk(datasets[0][0], datasets[0][1][0:10]) #, datasets[0][2][0:10])
def make_chunk(imgs, coords):
	w = coords[0, -1] -  coords[0, -2]
	#print coords[0]
	data = np.zeros((coords.shape[0], 3*w*w), dtype=np.int8)
	for i in xrange(coords.shape[0]):
		data[i, :] = 1.0*imgs[coords[i,0], :, coords[i,-4]:coords[i,-3], coords[i,-2]:coords[i,-1]].reshape(-1)
# 		if True and l[i] ==1:
# 			d = imgs[coords[i,0], :, coords[i,-4]:coords[i,-3], coords[i,-2]:coords[i,-1]]
# 			img = Image.fromarray(d[1,:,:])
# 			img.show()	
# 			raw_input("Press Enter to continue...")
# 			img.close()
			
	return data	



#model = do_dnn1(datasets, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400, load_model_from_pkl=False)
def do_dnn1(datasets, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400, load_model_from_pkl=False):

	""" 
	model in the paper
	layer		output size			filter size
	input:		3x101x101			-
	0 C			16 x 100 x 100		2 x 2
	0 MP		16 x 50 x 50		2 x 2	#3 x 3
	1 C			16 x 48 x 48		3 x 3	#2 x 2
	1 MP		16 x 24 x 24		2 x 2
	2 C			16 x 22 x 22		3 x 3
	2 MP		16 x 11 x 11		2 x 2
	3 C			16 x 10 x 10		2 x 2
	3 MP		16 x 5 x 5			2 x 2
	4 C			16 x 4 x 4			2 x 2
	4 MP		16 x 2 x 2			2 x 2
	5 FC		100					-
	6 FC		1
	
					-
	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
						  gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: path to the dataset used for training /testing (MNIST here)

	:type nkerns: list of ints
	:param nkerns: number of kernels on each layer
	"""

	rng = np.random.RandomState(23455)
	
	max_chunksize_gb = 7.0 #Gb
	#gpu stores arrays as 64 bit floats
	chunk_size = int( max_chunksize_gb / ( 64*3*(datasets[0][1][0, -1] - datasets[0][1][0, -2])**2/(1000.0**3) ) )
	print 'chunk size ', chunk_size
	total_chunks = int(np.ceil(1.0*datasets[0][1].shape[0]/chunk_size))
	print 'total chunks ', total_chunks
	
	img_dim = int(datasets[0][1][0, -1] - datasets[0][1][0, -2]) # train_set_x.get_value(borrow=True).shape[1:]
	img_dim = np.array([3, img_dim, img_dim])


	# allocate symbolic variables for the data
	index = T.lscalar()	 # index to a [mini]batch

	# start-snippet-1
	x = T.matrix('x')	# the data is presented as rasterized images
	y = T.ivector('y')	# the labels are presented as 1D vector of
			
	if load_model_from_pkl:
		'print loading existing model from pkl'
		[batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5, layer6]	= load_pkl_model('dnn1_model_batchsize400.pkl')
		
	else:		
		#filter size
		fs = np.array([[2,2],[3,3],[3,3],[2,2],[2,2]], dtype=np.int8)
		#pool size
		ps = np.array([[2,2],[2,2],[2,2],[2,2],[2,2]], dtype=np.int8)
		#im_shape
		im_s = np.zeros((6,2), dtype=np.int8)
		im_s[0,:] = [img_dim[1], img_dim[2]]
		window_size = img_dim[1]
	
		print 'layer 0 input shape', img_dim[1]
		for i in range(1,im_s.shape[0]):
			l = im_s[i-1,0]
			l = (l-fs[i-1,0]+1)/ps[i-1,0]
			im_s[i,:] = np.array([l,l])
			print 'layer ', i, ' input shape', l
	
		######################
		# BUILD ACTUAL MODEL #
		######################
		print '... building the model'

		# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
		# to a 4D tensor, compatible with our LeNetConvPoolLayer
		# (28, 28) is the size of MNIST images.
		layer0_input = x.reshape((batch_size, img_dim[0], img_dim[1], img_dim[2]))

		# Construct the first convolutional pooling layer:
		# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
		# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
		# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
		layer0 = c_mlp.LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, img_dim[0], img_dim[1], img_dim[2]), filter_shape=(nkerns[0], 3, fs[0,0], fs[0,1]), poolsize=(ps[0,0], ps[0,1]))
		layer1 = c_mlp.LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], im_s[1,0], im_s[1,1]), filter_shape=(nkerns[1], nkerns[0], fs[1,0], fs[1,1]), poolsize=(ps[1,0], ps[1,1]))
		layer2 = c_mlp.LeNetConvPoolLayer(rng, input=layer1.output, image_shape=(batch_size, nkerns[1], im_s[2,0], im_s[2,1]), filter_shape=(nkerns[2], nkerns[1], fs[2,0], fs[2,1]), poolsize=(ps[2,0], ps[2,1]))
		layer3 = c_mlp.LeNetConvPoolLayer(rng, input=layer2.output, image_shape=(batch_size, nkerns[2], im_s[3,0], im_s[3,1]), filter_shape=(nkerns[3], nkerns[2], fs[3,0], fs[3,1]), poolsize=(ps[3,0], ps[3,1]))
		layer4 = c_mlp.LeNetConvPoolLayer(rng, input=layer3.output, image_shape=(batch_size, nkerns[3], im_s[4,0], im_s[4,1]), filter_shape=(nkerns[4], nkerns[3], fs[4,0], fs[4,1]), poolsize=(ps[4,0], ps[4,1]))
		layer5_input = layer4.output.flatten(2)

		# construct a fully-connected sigmoidal layer
		layer5 = c_mlp.HiddenLayer(rng, input=layer5_input, n_in=nkerns[4] * im_s[5,0] * im_s[5,1], n_out=100, activation=T.tanh)

		# classify the values of the fully-connected sigmoidal layer
		# last layer is softmax
		layer6 = c_mlp.LogisticRegression(input=layer5.output, n_in=100, n_out=2)




	# the cost we minimize during training is the NLL of the model
	cost = layer6.negative_log_likelihood(y)

	# create a list of all model parameters to be fit by gradient descent
	params =   layer6.params + layer5.params +layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

	# create a list of gradients for all model parameters
	grads = T.grad(cost, params)

	# train_model is a function that updates the model parameters by
	# SGD Since this model has many parameters, it would be tedious to
	# manually create an update rule for each model parameter. We thus
	# create the updates list by automatically looping over all
	# (params[i], grads[i]) pairs.
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]




	###############
	# TRAIN MODEL #
	###############
	print '... training'
	# early-stopping parameters
	patience = 5000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant

	best_validation_loss = np.inf
	best_params = None
	best_iter = 0
	test_score = 0.
	start_time = time.clock()
		

	epoch = 0
	done_looping = False

	valid_set_x, valid_set_y = shared_dataset(make_chunk(datasets[1][0], datasets[1][1][0:1*chunk_size]), datasets[1][2][0:1*chunk_size], borrow=True)
	# create a function to compute the mistakes that are made by the model
	validate_model = theano.function(
		[index],
		layer6.errors(y),
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)	

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
	
		
		for c in xrange(total_chunks):
			#print 'starting chunk ', c
			train_set_x, train_set_y = shared_dataset(make_chunk(datasets[0][0], datasets[0][1][c*chunk_size:(c+1)*chunk_size]), datasets[0][2][c*chunk_size:(c+1)*chunk_size], borrow=True)

			# compute number of minibatches for training, validation and testing
			n_train_batches = train_set_x.get_value(borrow=True).shape[0]
			n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
			#n_test_batches = test_set_x.get_value(borrow=True).shape[0]
			n_train_batches /= batch_size
			n_valid_batches /= batch_size
			#n_test_batches /= batch_size

			validation_frequency = min(n_train_batches, patience / 2)
			# go through this many minibatches before checking the network
			# on the validation set. if patience is large, then we check every epoch

			train_model = theano.function(
				[index],
				cost,
				updates=updates,
				givens={
					x: train_set_x[index * batch_size: (index + 1) * batch_size],
					y: train_set_y[index * batch_size: (index + 1) * batch_size]
				}
			)

			
				
			for i in [90, 90,90,90]:
				train_set_x = T.reshape(train_set_x, [chunk_size, img_dim[0], img_dim[2], img_dim[2]])
				train_set_x = train_set_x[:, :, :, range(img_dim[2]-1,-1,-1)]
				train_set_x = train_set_x.dimshuffle(0,1,3,2)
				train_set_x = T.reshape(train_set_x, [chunk_size, img_dim[0]*img_dim[2]*img_dim[2]])
											
# 				train_set_x = T.reshape(train_set_x, [chunk_size, img_dim[0], img_dim[2], img_dim[2]])
# 				for i in range(np.random.randint(0,4)):
# 					train_set_x = train_set_x[:, :, :, range(img_dim[2]-1,-1,-1)]
# 					train_set_x = train_set_x.dimshuffle(0,1,3,2)
# 				train_set_x = T.reshape(train_set_x, [chunk_size, img_dim[0]*img_dim[2]*img_dim[2]])
	
				for minibatch_index in xrange(n_train_batches):

					iter = (epoch - 1) * n_train_batches + minibatch_index

					cost_ij = train_model(minibatch_index)

					if (iter + 1) % validation_frequency == 0:

						# compute zero-one loss on validation set
						validation_losses = [validate_model(i) for i
											 in xrange(n_valid_batches)]
						this_validation_loss = np.mean(validation_losses)
						print('epoch %i, chunk %i, minibatch %i/%i, validation error %f %%' %
							  (epoch, c, minibatch_index + 1, n_train_batches,
							   this_validation_loss * 100.))

						# if we got the best validation score until now
						if this_validation_loss < best_validation_loss:

							#improve patience if loss improvement is good enough
							if this_validation_loss < best_validation_loss *  \
							   improvement_threshold:
								patience = max(patience, iter * patience_increase)

							# save best validation score and iteration number
							best_validation_loss = this_validation_loss
							best_iter = iter
							best_params = copy.deepcopy(params)

							# test it on the test set
		# 					test_losses = [
		# 						test_model(i)
		# 						for i in xrange(n_test_batches)
		# 					]
		# 					test_score = np.mean(test_losses)
		# 					print(('	 epoch %i, minibatch %i/%i, test error of '
		# 						   'best model %f %%') %
		# 						  (epoch, minibatch_index + 1, n_train_batches,
		# 						   test_score * 100.))

					if patience <= iter:
						done_looping = True
						break
		#save a pickle of the model each epoch
		model = [batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5, layer6]
		pkl_model(model, 'dnn1_model_batchsize400.pkl')

	end_time = time.clock()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, ' %
		  (best_validation_loss * 100., best_iter + 1))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))					  

	return [batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5, layer6]

def pkl_model(model, fname):
	f = open(fname, 'w')
	pickle.dump(model, f)
	f.close()

#model = load_pkl_model('model_batchsize400.pkl')
#data = load_pkl_model('sample_proba_map_train0.pkl')
def load_pkl_model(fname):
	f = open(fname)
	data = pickle.load(f)
	f.close()
	return data


def rotate_mirror_chunk(chunk, img_shape, angle, mirror_in_x=False, mirror_in_y=False):
	'''
	chunk is a theano tensor of images
	'''
	n = chunk.shape[0]
	
	#if chunk is a theano 2D tensor with dimensions: (number of images) by (size of images = colours*width*height)
	[c, w, h] = img_shape

	
	if angle % 90 != 0:
		print 'rotation angle not valid: should be multiple of 90 (quarter turn)'
		return chunk
		
	if chunk.shape == [n, c, w, h]:
		already_shaped = True
	else:	
		chunk = T.reshape(chunk, [n, c, w, h])
		already_shaped = False
		
	r = (angle/90) % 4
	
	if mirror_in_x:
		chunk = chunk[:, :, range(w-1,-1,-1), :]
	if mirror_in_y:
		chunk = chunk[:, :, :, range(h-1,-1,-1)]
	
	for i in xrange(r):
		chunk = chunk[:, :, :, range(h-1,-1,-1)]
		chunk = chunk.dimshuffle(0,1,3,2)
		[w, h] = [h, w]
	
	if already_shaped:
		return chunk
	else:	
		return T.reshape(chunk, [n, c*w*h])
		

def make_proba_maps_dnn1():
	datasets = make_train_valid()
	model = load_pkl_model('dnn1_model_batchsize400.pkl')
	
	for i in range(datasets[0][0].shape[0]):
		
		proba_map_train = do_pred(model, datasets[0], i, 'dnn1')
		im3 = Image.fromarray(1.0*proba_map_train0)
		print 'saving proba_map_dnn1/proba_map_train'+str(i)+'.tif'
		im3.save('proba_map_dnn1/proba_map_train'+str(i)+'.tif')
		
		


#proba_map_train = do_pred(model, datasets[0], 0, 'dnn1')
def do_pred(model, dataset, i, model_type):
	if model_type == 'dnn1':
		[batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5, layer6] = model
	if model_type == 'dnn2':	
		[batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5] = model
								  
	window_size = int(dataset[1][0, -1] - dataset[1][0, -2])
	index = T.lscalar()				
								  
	#make probability maps for individual images
	proba_map = np.zeros((dataset[0].shape[3]-window_size+1, dataset[0].shape[3]-window_size+1))
	
	start_time = time.clock()
	#for i in xrange(1): #dataset[0].shape[0]):
	print 'making prediction image ', i
	#sub_images = make_chunk_from_single_img(datasets[0][0][0], 101, 0, 2000**2)
	#img = theano.shared(np.asarray(make_chunk_from_single_img(datasets[0][0][i], window_size, 0, 2000**2), dtype=theano.config.floatX), borrow=borrow)
	img = theano.shared(np.asarray(dataset[0][i], dtype=theano.config.floatX), borrow=True)

				
	for j in xrange(0, window_size):
		nx = int(np.floor(1.0*(img.shape.eval()[1]-j)/window_size))
		print j
		
		if j % 10:
			#pkl_model(proba_map, 'proba_map_partway.pkl')
			im = Image.fromarray(1.0*proba_map)
			im.save('proba_map_partway.tif')
			
		for k in xrange(0, window_size):
			print k
			ny = int(np.floor(1.0*(img.shape.eval()[1]-k)/window_size))

			t1 = time.clock()
			t = T.reshape(img[:, j:(j+nx*window_size),k:(k+ny*window_size) ], [3, nx, window_size,ny, window_size])
			t = t.dimshuffle(0,1,3,2,4)
			t = T.reshape(t, [3, nx*ny, window_size, window_size])
			t = t.dimshuffle(1,0,2,3)		
			#check a random sub-image to make sure the images are correctly sliced	
# 			q = Image.fromarray(t[230,0, :,:].eval())
# 			q.show()
# 			break
			
			t = T.reshape(t, [nx*ny, 3*window_size**2])
			
			#T.reshape(T.reshape(T.reshape(img[:, j:(j+nx*window_size),k:(k+ny*window_size) ], [3, nx, window_size,ny, window_size]).dimshuffle(0,1,3,2,4), [3, nx*ny, window_size, window_size]).dimshuffle(1,0,2,3), [nx*ny, 3*window_size**2])
			
			t = T.tile(t, [2,1])
 			t = t[0:400]
 			
 			t2 = time.clock()
 			print 'time 2 - time 1', t2-t1
 			
			#print 't shape ', t.shape.eval(), nx, ny

			if model_type == 'dnn1':
				make_pred = theano.function(
					[index],
					layer6.pred_proba(),
					givens={
						x: t[index*batch_size:(index+1)*batch_size]
					}
				)
				
			if model_type == 'dnn2':
				make_pred = theano.function(
					[index],
					layer5.pred_proba(),
					givens={
						x: t[index*batch_size:(index+1)*batch_size]
						#x: T.reshape(T.reshape(T.reshape(img[:, j:(j+nx*window_size),k:(k+ny*window_size) ], [3, nx, window_size,ny, window_size]).dimshuffle(0,1,3,2,4), [3, nx*ny, window_size, window_size]).dimshuffle(1,0,2,3), [nx*ny, 3*window_size**2])[index*batch_size:(index+1)*batch_size]
	
					}
				)
			
			t3 = time.clock()
			print 'time 3 - time 2', t3-t2	
# 				p = np.zeros((nx*ny))
# 				for l in xrange(nx*ny/batch_size):
# 					p[l*batch_size:(l+1)*batch_size] = make_pred(l)

			#for testing, use red pixel at centre
			#p = t[:, 5100].eval()
			p = make_pred(0)
			t4 = time.clock()
			print 'time 4 - time 3', t4-t3			
			
			#print nx, ny, proba_map_train.shape
			proba_map[j::window_size, k::window_size] = p[0:nx*ny].reshape((nx, ny))
			
			#code in regular python
			#b = np.array(range(3*6*8)).reshape((3,6,8))
			#ws = 2
			#t =b.reshape((3,b.shape[1]/ws, ws,b.shape[2]/ws, ws)).swapaxes(2,3).reshape((3,b.shape[1]*b.shape[2]/ws**2,ws,ws))	
			#np.swapaxes(t, 0,1)[0] 
		
	end_time = time.clock()
	print 'time for ', i, ' iterations: ', (end_time - start_time)/60.0
	#time for one image on laptop 982 minutes
				
	return proba_map				
	

#train_dataset2 = construct_dataset2(datasets[0])
#valid_dataset2 = construct_dataset2(datasets[0])
def construct_dataset2(datasets):
	f_positive_mitosis = 0.05
	n_pos_mitosis = int(1.0/f_positive_mitosis)
	ws = datasets[0][1][0,-1] - datasets[0][1][0,-2] # window_size
	
	coords2 = np.zeros((n_pos_mitosis*datasets[1].shape[0], 5), dtype=np.int16)
	labels2 = np.zeros(n_pos_mitosis*datasets[1].shape[0], dtype=np.int8)
	j = 0
	
	for img_num in [0]: #xrange(datasets[0].shape[0]):
		
		#proba_map = load_pkl_model('sample_proba_map_train'+str(img_num)+'.pkl')	
		proba_map = np.array(Image.open('proba_map_train'+str(img_num)+'.tif'))
		new_coords, new_labels = construct_images_likely_cell_centres(proba_map, datasets, img_num, ws, f_positive_mitosis, dset='train')
		#if j+new_coords.shape[0] < coords2.shape[0]:
		coords2[j:j+new_coords.shape[0], :] = new_coords
		labels2[j:j+new_labels.shape[0]] = new_labels
		j += new_coords.shape[0]
			
	return coords2[0:j], labels2[0:j]
		
	
def construct_images_likely_cell_centres(proba_map, datasets, img_num, ws, f_positive_mitosis, dset='train'):
	cutoff = 0.5
	n_pos_mitosis = int(1.0/f_positive_mitosis -1)
	#ws = window_size
	
	#select only the true mitoses from the dnn1 training data
	if dset=='train':
		img = datasets[0][0][img_num]
		ind = np.where(datasets[0][1][:,0] == img_num)
		coords = datasets[0][1][ind]
		labels = datasets[0][2][ind]
		ind = np.where(labels==1)
		coords = coords[ind]
		
	if set=='valid':
		img = datasets[1][0][img_num]
		ind = np.where(datasets[1][1][:,0] == img_num)
		coords = datasets[1][1][ind]
		labels = datasets[1][2][ind]
		ind = np.where(labels==1)
		coords = coords[ind]
	
	#construct the false mitoses set by	excluding all the true mitoses from the possible mitoses 				
	coords2 = np.array(np.where(proba_map > cutoff), dtype=np.int16).T + (ws-1)/2
	
	i = 0
	while i < coords.shape[0]:
# 		if i % 100 == 0:
# 			print coords2.shape[0]
		#print np.min((0.5*(coords[i,1] +coords[i,3])- coords2[:,0])**2 + (0.5*(coords[i,2]+coords[i,4]) - coords2[:,1])**2)
		use = ((0.5*(coords[i,1] +coords[i,3])- coords2[:,0])**2 + (0.5*(coords[i,2]+coords[i,4]) - coords2[:,1])**2 > (2.0)**2)
		
		coords2 = coords2[use==True]
		i += 1
	
	print coords2.shape
	#select the number of negative samples
	n_coords2 = np.min([coords2.shape[0], np.max([n_pos_mitosis*coords.shape[0], 2*n_pos_mitosis])])
	ind = range(coords2.shape[0])
	np.random.shuffle(ind)
	ind = ind[0:n_coords2]
	
	
	#make coord array with upper and lower corners, add image number too coord array
	coords2 = coords2[ind]
	print coords2.shape
	coords2 = np.hstack((coords2[:,0].reshape((-1,1))-ws, coords2[:, 0].reshape((-1,1))+ws+1, coords2[:,1].reshape((-1,1))-ws, coords2[:,1].reshape((-1,1))+ws+1))
	
	coords2 = np.hstack((img_num*np.ones((coords2.shape[0],1)), coords2))
	
	#glue positive and negative samples together in one array
	labels2 = np.vstack(( labels.reshape((-1,1)), np.zeros((coords2.shape[0],1)) ))
	coords2 = np.vstack(( coords, coords2 ))
	n_coords2 = coords2.shape[0]
	ind = range(n_coords2)
	np.random.shuffle(ind)
	
	return coords2[ind], labels2[ind].reshape(-1)
	
	
	
	
def stest():
	i = 0
	j = 10
	while i < j:
		print i, j
		i += 1
		j -= 1	
	
	
#model = do_dnn2(datasets2, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400)
def do_dnn2(datasets2,learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400):

	""" 
	model in the paper
	layer		output size			filter size
	input:		3x101x101			-
	0 C			16 x 98 x 98		4 x 4
	0 MP		16 x 49 x 49		2 x 2	#3 x 3
	1 C			16 x 46 x 46		4 x 4	#2 x 2
	1 MP		16 x 23 x 23		2 x 2
	2 C			16 x 20 x 20		4 x 4
	2 MP		16 x 10 x 10		2 x 2
	3 C			16 x 8 x 8			3 x 3
	3 MP		16 x 4 x 4			2 x 2
	4 FC		100					-
	5 FC		2
	
					-
	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
						  gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: path to the dataset used for training /testing (MNIST here)

	:type nkerns: list of ints
	:param nkerns: number of kernels on each layer
	"""

	rng = np.random.RandomState(23455)


	
	max_chunksize_gb = 4.0 #Gb
	#gpu stores arrays as 64 bit floats
	chunk_size = int( max_chunksize_gb / ( 64*3*(datasets[0][1][0, -1] - datasets[0][1][0, -2])**2/(1000.0**3) ) )
	print 'chunk size ', chunk_size
	total_chunks = int(np.ceil(1.0*datasets[0][1].shape[0]/chunk_size))
	print 'total chunks ', total_chunks
	
	img_dim = int(datasets[0][1][0, -1] - datasets[0][1][0, -2]) # train_set_x.get_value(borrow=True).shape[1:]
	img_dim = np.array([3, img_dim, img_dim])


	# allocate symbolic variables for the data
	index = T.lscalar()	 # index to a [mini]batch

	# start-snippet-1
	x = T.matrix('x')	# the data is presented as rasterized images
	y = T.ivector('y')	# the labels are presented as 1D vector of
						# [int] labels
	#filter size
	fs = np.array([[4,4],[4,4],[4,4],[3,3]], dtype=np.int8)
	#pool size
	ps = np.array([[2,2],[2,2],[2,2],[2,2]], dtype=np.int8)
	#im_shape
	im_s = np.zeros((6,2), dtype=np.int8)
	im_s[0,:] = [img_dim[1], img_dim[2]]
	window_size = img_dim[1]
	
	print 'layer 0 input shape', img_dim[1]
	for i in range(1,im_s.shape[0]):
		l = im_s[i-1,0]
		l = (l-fs[i-1,0]+1)/ps[i-1,0]
		im_s[i,:] = np.array([l,l])
		print 'layer ', i, ' input shape', l
	
	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
	# to a 4D tensor, compatible with our LeNetConvPoolLayer
	# (28, 28) is the size of MNIST images.
	layer0_input = x.reshape((batch_size, img_dim[0], img_dim[1], img_dim[2]))

	# Construct the first convolutional pooling layer:
	# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
	# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
	# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
	layer0 = c_mlp.LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, img_dim[0], img_dim[1], img_dim[2]), filter_shape=(nkerns[0], 3, fs[0,0], fs[0,1]), poolsize=(ps[0,0], ps[0,1]))

	layer1 = c_mlp.LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], im_s[1,0], im_s[1,1]), filter_shape=(nkerns[1], nkerns[0], fs[1,0], fs[1,1]), poolsize=(ps[1,0], ps[1,1]))

	layer2 = c_mlp.LeNetConvPoolLayer(rng, input=layer1.output, image_shape=(batch_size, nkerns[1], im_s[2,0], im_s[2,1]), filter_shape=(nkerns[2], nkerns[1], fs[2,0], fs[2,1]), poolsize=(ps[2,0], ps[2,1]))

	layer3 = c_mlp.LeNetConvPoolLayer(rng, input=layer2.output, image_shape=(batch_size, nkerns[2], im_s[3,0], im_s[3,1]), filter_shape=(nkerns[3], nkerns[2], fs[3,0], fs[3,1]), poolsize=(ps[3,0], ps[3,1]))
	
	# the HiddenLayer being fully-connected, it operates on 2D matrices of
	# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
	# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
	# or (500, 50 * 4 * 4) = (500, 800) with the default values.
	layer4_input = layer3.output.flatten(2)

	# construct a fully-connected sigmoidal layer
	layer4 = c_mlp.HiddenLayer(rng, input=layer4_input, n_in=nkerns[3] * im_s[4,0] * im_s[4,1], n_out=100, activation=T.tanh)

	# classify the values of the fully-connected sigmoidal layer
	# last layer is softmax
	layer5 = c_mlp.LogisticRegression(input=layer4.output, n_in=100, n_out=2)

	# the cost we minimize during training is the NLL of the model
	cost = layer5.negative_log_likelihood(y)


	# create a list of all model parameters to be fit by gradient descent
	params =   layer5.params +layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

	# create a list of gradients for all model parameters
	grads = T.grad(cost, params)

	# train_model is a function that updates the model parameters by
	# SGD Since this model has many parameters, it would be tedious to
	# manually create an update rule for each model parameter. We thus
	# create the updates list by automatically looping over all
	# (params[i], grads[i]) pairs.
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]




	###############
	# TRAIN MODEL #
	###############
	print '... training'
	# early-stopping parameters
	patience = 5000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant

	best_validation_loss = np.inf
	best_params = None
	best_iter = 0
	test_score = 0.
	start_time = time.clock()
		

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		#use part of training data as the validation set
		valid_set_x, valid_set_y = shared_dataset(make_chunk(datasets[0][0], datasets[0][1][0:1*chunk_size]), datasets[0][2][0:1*chunk_size], borrow=True)
		# create a function to compute the mistakes that are made by the model
		validate_model = theano.function(
			[index],
			layer6.errors(y),
			givens={
				x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)		
		
		for c in xrange(total_chunks):
			print 'starting chunk ', c
			train_set_x, train_set_y = shared_dataset(make_chunk(datasets[0][0], datasets[0][1][c*chunk_size:(c+1)*chunk_size]), datasets[0][2][c*chunk_size:(c+1)*chunk_size], borrow=True)

			# compute number of minibatches for training, validation and testing
			n_train_batches = train_set_x.get_value(borrow=True).shape[0]
			n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
			#n_test_batches = test_set_x.get_value(borrow=True).shape[0]
			n_train_batches /= batch_size
			n_valid_batches /= batch_size
			#n_test_batches /= batch_size

			validation_frequency = min(n_train_batches, patience / 2)
			# go through this many minibatches before checking the network
			# on the validation set. if patience is large, then we check every epoch

			train_model = theano.function(
				[index],
				cost,
				updates=updates,
				givens={
					x: train_set_x[index * batch_size: (index + 1) * batch_size],
					y: train_set_y[index * batch_size: (index + 1) * batch_size]
				}
			)

			
	
			#print 'train set shape ', train_set_x.shape.eval()
				
			for minibatch_index in xrange(n_train_batches):

				iter = (epoch - 1) * n_train_batches + minibatch_index
				#print 'training @ iter = ', iter
				
				cost_ij = train_model(minibatch_index)

				if (iter + 1) % validation_frequency == 0:

					# compute zero-one loss on validation set
					validation_losses = [validate_model(i) for i
										 in xrange(n_valid_batches)]
					this_validation_loss = np.mean(validation_losses)
					print('epoch %i, chunk %i, minibatch %i/%i, validation error %f %%' %
						  (epoch, c, minibatch_index + 1, n_train_batches,
						   this_validation_loss * 100.))

					# if we got the best validation score until now
					if this_validation_loss < best_validation_loss:

						#improve patience if loss improvement is good enough
						if this_validation_loss < best_validation_loss *  \
						   improvement_threshold:
							patience = max(patience, iter * patience_increase)

						# save best validation score and iteration number
						best_validation_loss = this_validation_loss
						best_iter = iter
						best_params = copy.deepcopy(params)

						# test it on the test set
	# 					test_losses = [
	# 						test_model(i)
	# 						for i in xrange(n_test_batches)
	# 					]
	# 					test_score = np.mean(test_losses)
	# 					print(('	 epoch %i, minibatch %i/%i, test error of '
	# 						   'best model %f %%') %
	# 						  (epoch, minibatch_index + 1, n_train_batches,
	# 						   test_score * 100.))

				if patience <= iter:
					done_looping = True
					break

		#save a pickle of the model each epoch
		model = [batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5, layer6]
		pkl_model(model, 'dnn2_model_batchsize400.pkl')
		
	end_time = time.clock()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, ' %
		  (best_validation_loss * 100., best_iter + 1))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))					  

	return [batch_size, x, y, layer0, layer1, layer2, layer3, layer4, layer5]	
	




def do_pred_all(datasets):
	[train_dataset, test_dataset] = datasets

	train_patients = ['02', '03', '04', '05', '06', '07', '08', '10']
	test_patients = ['01', '09', '11', '12']
	
	i = 0
	patient_look_up = {}
	
	for patient in train_patients + test_patients:
		dir = 'mitos_img/'+ patient + '/'
	
		imgfiles = glob.glob(dir+'*.tif')
		for imgfile in imagefiles:
			patient_look_up[i] = [patient, imagefile[0:-4]]
			i += 1

	
	for dset in [0,1]:	
		for i in xrange(datasets[dset].shape):
			proba_map =  do_pred(model, datasets[dset][0], i, 'dnn2')
			im = Image.fromarray(1.0*proba_map)
			if dset==0:
				im.save('proba_map_train'+str(i)+'.tif')
			if dset==1:
				im.save('proba_map_test'+str(i)+'.tif')				
			coords = np.array(np.where(proba_map > 0.5)).T
			
			np.savetxt('pred_'+str(i)+'_patient'+str(patientlookup[i][0])+'_'+str(patientlookup[i][1])+'.csv', coords)



if __name__ == "__main__":
#    nr.seed(6)

#     op = ConvNet.get_options_parser()
# 
#     op, load_dic = IGPUModel.parse_options(op)
#     model = ConvNet(op, load_dic)
#     model.start()	

	'''
	readme
	
	#command line execution with gpu
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mitos5.py

	
	datasets = make_train_valid()
	model = do_dnn1(datasets, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400, load_model_from_pkl=True)
	
	'''
	make_proba_maps_dnn1()
	'''
	
	#or reload model from pkl
	#model = load_pkl_model('model_batchsize400.pkl')

	#generate probability map for mitoses in each image (or load the map from the pkl file)
	#proba_map_train0 = do_pred(model, datasets[0], 0, 'dnn1')
	#proba_map_train0 = load_pkl_model('sample_proba_map_train0.pkl')
	im3 = Image.fromarray(255.0*proba_map_train0)
	im3.show()

	#generate coords for training images for dnn2
	#train_dataset2 = construct_dataset2(datasets[0])
	#valid_dataset2 = construct_dataset2(datasets[0])
	dataset2 = [train_dataset2, valid_dataset2]
	model2 = do_dnn2(datasets2, learning_rate=0.1, n_epochs=1, nkerns=[16,16,16,16,16], batch_size=400)

	#generate probability maps for validation data
	proba_map[0,:, :] = int(np.round( do_pred(model, datasets[1], 0, 'dnn2') ))
	do_pred_all(datasets)

	'''
	
	'''
	
	#proba_map_train0 = load_pkl_model('sample_proba_map_train0.pkl')
	proba_map_train0 = np.array(Image.open('proba_map_train0.tif'))
	print 'mean ', np.mean(proba_map_train0)
	im3 = Image.fromarray(1.0*proba_map_train0)
	im3.save('proba_map_train0.tif')
	im3 = Image.fromarray(255.0*proba_map_train0)
	im3.show()
	'''
	
	'''
	datasets = make_train_valid()
#	DON'T BOTHER PICKLING DATA - it's faster just to generate it than to pickle and unpickle	
# 	print 'pickling datasets'
# 	pkl_model(datasets, 'datasets.pkl')
#	print 'loading datasets'
#	datasets = load_pkl('datasets.pkl')

	#model = do_dnn1(datasets, learning_rate=0.1, n_epochs=10, nkerns=[16,16,16,16,16], batch_size=400)
	'''