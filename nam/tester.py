import numpy as np
from torch.utils.data import random_split
import torch
from nam.wrapper import NAMClassifier, MultiTaskNAMClassifier

random_state = 2016

from tensorflow import keras
# Load the CIFAR-10 dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


subset_size = 2000
train_images = train_images[:subset_size]
test_images = test_images[:subset_size]
train_labels = train_labels[:subset_size]
test_labels = test_labels[:subset_size]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28*28)
test_images = test_images.reshape(-1, 28*28)
print(train_images.shape)

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
print(test_labels.shape)

model = NAMClassifier(
            num_epochs=200,
            num_learners=1,
            metric='accuracy',
            early_stop_mode='max',
            monitor_loss=True,
            n_jobs=1,
            random_state=random_state,
            loss_func=torch.nn.CrossEntropyLoss(reduction="none")
        )

from time import perf_counter
print("training")
start = perf_counter()
model.fit(train_images, train_labels)
print(perf_counter()-start)

def evaluate_acc(preds, labels):
	preds = np.argmax(preds, axis=1)
	labels = np.argmax(labels, axis=1)
	# print(preds)
	# print(labels)
	return np.sum(preds==labels)/len(preds)

preds = model.predict_proba(test_images)
print(evaluate_acc(preds,test_labels))