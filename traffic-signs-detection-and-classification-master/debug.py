from __future__ import absolute_import, division, print_function, unicode_literals

import utils
import preprocessing as pre
import numpy as np
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import NN as nn


train_data_dir, train_labels_path = "data/gtsrb-german-traffic-sign/Train", "./data/gtsrb-german-traffic-sign/Train.csv"
train_data_set = nn.Dataset(train_data_dir, train_labels_path, data='train', n_samples=2000)

test_data_dir, test_labels_path = "data/gtsrb-german-traffic-sign/Test", "./data/gtsrb-german-traffic-sign/Test.csv"
test_data_set = nn.Dataset(test_data_dir, test_labels_path, data='test')

reshaped_train_images = []
for image in train_data_set.X:
    dst = pre.eqHist(image)
    dst = pre.reshape(dst)
    reshaped_train_images.append(dst)
train_data_set.X = reshaped_train_images

reshaped_test_images = []
for image in test_data_set.X:
    dst = pre.eqHist(image)
    dst = pre.reshape(dst)
    reshaped_test_images.append(dst)
test_data_set.X = reshaped_test_images

reshaped_train_images = []
for image in train_data_set.X:
    dst = image.reshape(-1,1)
    reshaped_train_images.append(dst)
train_data_set.X = np.asarray(reshaped_train_images)

reshaped_test_images = []
for image in test_data_set.X:
    dst = image.reshape(-1,1)
    reshaped_test_images.append(dst)
test_data_set.X = np.asarray(reshaped_test_images)

mlp = nn.MLP("NN.dat", train_data_set, print_step=1, verbose=1)
#mlp.train(n_epochs=10, learning_rate=2, decay=1.)
#mlp.make_plot()
#
#mlp.setdataset(test_data_set)
#mlp.print_accuracy()