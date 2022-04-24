#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script creates images with a sentence embedding.
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# PARAMETERS
#MODEL_URL = 'D:/workspace/image_generation_models/bigbigan-resnet50/'
MODEL_URL = '/data1/image_generators/bigbigan-resnet50/'

# Load model and input data
model = tf.saved_model.load(MODEL_URL)
poem_embedding = np.load('poem.npy')
# Create the mapping matrix
proj = np.random.normal(size=(512,120))
z = np.matmul(poem_embedding.reshape(1,512), proj)
z = tf.convert_to_tensor(z, dtype=float)
# Generate image
gen_samples = model.signatures['generate'](z)
images = np.array(gen_samples['upsampled'])
# Rescale image
tmp = np.flip(images[0,:,:,:],2)
tmp = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
# Save image
plt.imsave('poem.jpg', tmp)
