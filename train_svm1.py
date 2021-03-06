from time import time
from sklearn.metrics import accuracy_score
from sklearn import svm, linear_model
import numpy as np
from Config import *
from six.moves import cPickle as pickle

acc = []
nums = np.load('{}/features.npy'.format(FLAGS.feature_dir))

for num in nums:
    X_train = np.load('{}/features{}_train.npy'.format(FLAGS.feature_dir, num))
    y_train = np.load('{}/label{}_train.npy'.format(FLAGS.feature_dir, num))
    feature_ids = '{}/feature_ids{}_train.npy'.format(FLAGS.feature_dir, num)

    print("Fitting the classifier to the training set")
    t0 = time()
    C = 1000.0
    clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    y_pred = clf.predict(X_train)
    print("Predicting on training ...")
    difference = np.where((y_train-y_pred) !=0)
    corresponding_feature_ids = feature_ids[difference]
    feature_ids_file = '{}/feature_ids{}_train_negative.npy'.format(FLAGS.feature_dir, num)
    np.save(feature_ids_file, corresponding_feature_ids)

