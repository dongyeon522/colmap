from dataloader import colmap_loader

import os
import sys
import argparse
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('seq_path', help='the path to the sequence of images.')
parser.add_argument('--input_views', help='comma-separated list of the indices in the sequence to use as inputs')
args = parser.parse_args()


input_views = []
if args.input_views is not None:
	input_views = [int(i) for i in args.input_views.split(',')]

print(args)

img_list, list_cam_params \
    = colmap_loader.COLMAPData.extract_data_to_param(args.seq_path)

#if len(input_views) > 0:
#    list_img = [list_img[i] for i in input_views]
#    list_depth = [list_depth[i] for i in input_views]
#    list_cam_params = [list_cam_params[i] for i in input_views]

#print('list_img', list_img)
#print('list_depth', list_depth)


fig = plt.figure(1)
ax = fig.gca(projection='3d')

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

X = []
Y = []
Z = []

ext = []

R = np.zeros((3,len(list_cam_params)))
T = np.zeros((3,len(list_cam_params)))

for i in range(len(list_cam_params)):
	x = list_cam_params[i]['extrinsic'][0][3]
	y = list_cam_params[i]['extrinsic'][1][3]
	z = list_cam_params[i]['extrinsic'][2][3]
	X.append(x)
	Y.append(y)
	Z.append(z)
	print(img_list[i])
	print('intrinsic')
	print(list_cam_params[i]['intrinsic'])
	print('extrinsic')
	print(list_cam_params[i]['extrinsic'])
	ext_array = list_cam_params[i]['extrinsic']

	ext_array = ext_array.reshape(4,4)
	rotation = ext_array[0:3, 0:3]
	position = ext_array[0:3, 3]
	vec = np.array([0., 0., 0.])

	cv2.Rodrigues(rotation, vec, jacobian=None)
	norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
	vec = vec/norm *0.1

	R[:,i] = vec
	T[:,i] = position

print(R.transpose())
print(T.transpose())

# ax.scatter(X, Y, Z)

ax.quiver(T[0,:], T[1,:], T[2,:], T[0,:]+ R[0,:], T[1,:] + R[1,:], T[2,:]+ R[2,:], length = 0.5, normalize = True)

for i, txt in enumerate(img_list):
	#label = '(%d, %d, %d), text=%s' % (X[i], Y[i], Z[i], txt)
	#ax.text(X[i], Y[i], Z[i], txt, 'x')
	ax.text(X[i], Y[i], Z[i], i, 'x')
		
	#ax.annotate(txt, (X[i], Y[i]))

plt.show()

# print('list_cam_params', list_cam_params)
print(len(list_cam_params))
