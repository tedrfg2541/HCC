#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision.models.vgg import VGG
import random
import sys

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String,Float64, Bool
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import os

from object_detection.srv import *

class FCN16s(nn.Module):

	def __init__(self, pretrained_net, n_class):
		super(FCN16s,self).__init__()
		self.n_class = n_class
		self.pretrained_net = pretrained_net
		self.relu    = nn.ReLU(inplace = True)
		self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn1     = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn2     = nn.BatchNorm2d(256)
		self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn3     = nn.BatchNorm2d(128)
		self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn4     = nn.BatchNorm2d(64)
		self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn5     = nn.BatchNorm2d(32)
		self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

	def forward(self, x):
		output = self.pretrained_net(x)
		x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
		x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

		score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
		score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
		score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
		score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
		score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
		score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
		score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

		return score

class VGGNet(VGG):
	def __init__(self, cfg, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
		super(VGGNet,self).__init__(self.make_layers(cfg[model]))
		ranges = {
			'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
			'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
			'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
			'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
		}
		self.ranges = ranges[model]

		if pretrained:
			exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

		if not requires_grad:
			for param in super().parameters():
				param.requires_grad = False

		if remove_fc:  # delete redundant fully-connected layer params, can save memory
			del self.classifier

		if show_params:
			for name, param in self.named_parameters():
				print(name, param.size())

	def forward(self, x):
		output = {}

		# get the output of each maxpooling layer (5 maxpool in VGG net)
		for idx in range(len(self.ranges)):
			for layer in range(self.ranges[idx][0], self.ranges[idx][1]):      
				x = self.features[layer](x)
			output["x%d"%(idx+1)] = x
		return output
	def make_layers(self, cfg, batch_norm=False):
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)

class task1_2(object):
 	def __init__(self):

		self.predict_ser = rospy.Service("prediction", task1out, self.prediction_cb)
		self.cv_bridge = CvBridge() 

		# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
		self.cfg = {
			'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
			'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		}
		self.means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
		self.h, self.w  = 480, 640
		self.n_class = 4
		model_dir = "/root/sis_mini_competition_2018"   # 
		model_name = "sis_99epoch.pkl"
		self.vgg_model = VGGNet(self.cfg, requires_grad=True, remove_fc=True)
		self.fcn_model = FCN16s(pretrained_net=self.vgg_model, n_class=self.n_class)

		use_gpu = torch.cuda.is_available()
		num_gpu = list(range(torch.cuda.device_count()))
		rospy.loginfo("Cuda available: %s", use_gpu)

		if use_gpu:
			ts = time.time()
			self.vgg_model = self.vgg_model.cuda()
			self.fcn_model = self.fcn_model.cuda()
			self.fcn_model = nn.DataParallel(self.fcn_model, device_ids=num_gpu)
			print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
		state_dict = torch.load(os.path.join(model_dir, model_name))
		self.fcn_model.load_state_dict(state_dict)
		
		self.mask1 = np.zeros((self.h, self.w))
		self.MAXAREA = 18000
		self.MINAREA = 1000
		self.brand = ['doublemint', 'kinder', 'kusan']
		####
		self.pub_pos = rospy.Publisher("/HCC/position",np.array,queue_size=10)
		rospy.loginfo("Service ready!")


	def prediction_cb(self, req):
		resp = task1outResponse()
		im_msg = rospy.wait_for_message('/camera/rgb/image_rect_color', Image, timeout=None)
		resp.pc = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2, timeout=None)
		rospy.loginfo("Get image.")
		resp.org_image = im_msg
		try:
			img = self.cv_bridge.imgmsg_to_cv2(im_msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		origin  = img
		img     = img[:, :, ::-1]  # switch to BGR

		img = np.transpose(img, (2, 0, 1)) / 255.
		img[0] -= self.means[0]
		img[1] -= self.means[1]
		img[2] -= self.means[2]

		now = rospy.get_time()
		# convert to tensor
		img = img[np.newaxis,:]
		img = torch.from_numpy(img.copy()).float() 

		output = self.fcn_model(img)
		output = output.data.cpu().numpy()

		N, _, h, w = output.shape
		mask = output.transpose(0, 2, 3, 1).reshape(-1, self.n_class).argmax(axis = 1).reshape(N, h, w)[0]
		rospy.loginfo("Predict time : %f", rospy.get_time() - now)
		now = rospy.get_time()

		show_img = np.asarray(origin)
		count = np.zeros(3)
		self.mask1[:,:] = 0
		self.mask1[mask != 0] = 1
		labels = self.adj(self.mask1)
		mask = np.asarray(mask, np.uint8)
		mask2 = np.zeros((h, w))
		for i in range(1, self.n_class):
			self.mask1[:,:] = 0
			self.mask1[mask == i] = 1
			self.mask1 = np.asarray(self.mask1, np.uint8)
			self.mask1 = cv2.GaussianBlur(self.mask1, (5, 5), 0)

			cnts = cv2.findContours(self.mask1.copy(), cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[1]
			sd = ShapeDetector()
			for c in cnts:
				if self.MAXAREA >= cv2.contourArea(c) >= self.MINAREA:
					shape = sd.detect(c)
					if shape is "rectangle":
						M = cv2.moments(c)
						if M["m00"] == 0 :
							break
						cX = int(M["m10"] / M["m00"])
						cY = int(M["m01"] / M["m00"])

						rect = cv2.minAreaRect(c)
						box = cv2.boxPoints(rect)
						box = np.int0(box)

						cv2.drawContours(show_img,[box], 0, (255, 0, 0), 2)
						cv2.putText(show_img, self.brand[i - 1], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 0), 2)
						#_, labels = cv2.conectedComponents(mask)
						p = labels[c[0,0,1],c[0,0,0]]
						mask2[labels == p] = i
						count[i - 1] += 1
		rospy.loginfo("Image processing time : %f", rospy.get_time() - now)
		#cv2.imshow('My image', show_img)
		cv2.imwrite("/hosthome/test.jpg", show_img)
		resp.process_image = self.cv_bridge.cv2_to_imgmsg(show_img, "bgr8")
		resp.mask = self.cv_bridge.cv2_to_imgmsg(mask2, "64FC1") 
		print(count)

		return resp

	def adj(self, _img, _level = 8):
		colomn, row = self.h, self.w
		_count = 0
		_pixel_pair = []
		label = np.zeros((colomn,row))
		for i in range(colomn):
			for j in range(row):
				if (_img[i,j] == 1 and label[i,j] == 0):
				    _pixel_pair.append([i,j])
				    _count += 1
				while len(_pixel_pair) != 0:
					pair = _pixel_pair.pop()
					a = pair[1] + 1
					b = pair[1] - 1
					c = pair[0] + 1
					d = pair[0] - 1
					if a == 640 : a -= 1
					if b == -1  : b += 1
					if c == 480 : c -= 1
					if d == -1  : d += 1

					if _img[pair[0],a] == 1 and label[pair[0],a] == 0:
					    _pixel_pair.append([pair[0],a])
					if _img[pair[0],b] == 1 and label[pair[0],b] == 0:
					    _pixel_pair.append([pair[0],b])
					if _img[c,pair[1]] == 1 and label[c,pair[1]] == 0:
					    _pixel_pair.append([c,pair[1]])
					if _img[d,pair[1]] == 1 and label[d,pair[1]] == 0:
					    _pixel_pair.append([d,pair[1]])
					if _level == 8:
						if _img[c,a] == 1 and label[c,a] == 0:
							_pixel_pair.append([c,a])
						if _img[d,a] == 1 and label[d,a] == 0:
							_pixel_pair.append([d,a])
						if _img[d,b] == 1 and label[d,b] == 0:
							_pixel_pair.append([d,b])
						if _img[c,b] == 1 and label[c,b] == 0:
							_pixel_pair.append([c,b])
					label[pair[0],pair[1]] = _count

		print("Num of classes for connected components : ", _count)
		return label

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	

class ShapeDetector:
	def __init__(self):
		pass
	def detect(self, c):

		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		c1 = cv2.convexHull(c)
		approx = cv2.approxPolyDP(c1, 0.04 * peri, True)

		if len(approx) == 3:
			shape = "triangle"

		elif len(approx) == 4:
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			shape = "square" if ar >= 0.9 and ar <= 1.1 else "rectangle"
			if h < 5 or w < 5:
				shape = "unidentified" 
		return shape

if __name__ == '__main__': 
	rospy.init_node('task1_2',anonymous=False)
	task1_2 = task1_2()
	rospy.on_shutdown(task1_2.onShutdown)
	rospy.spin()
