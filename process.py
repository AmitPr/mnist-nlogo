#! /usr/bin/python3

# reads CSV file, creates 10 floats, writes results to CSV
import sys
import tensorflow as tf
import numpy as np
f=open(sys.argv[1],'r')
lines = f.read().split('\n')
f.close()

invals = [float(x) for x in lines[0].split(',')]
arr = np.asarray(invals)
arr = arr.reshape(1,28,28,1)
model = tf.keras.models.load_model('mnist_model.h5')
results = model.predict(arr).tolist()
# write out result
f=open(sys.argv[2],'w')
f.write(','.join(['{:.20f}'.format(x) for x in results[0]]))
f.close()
