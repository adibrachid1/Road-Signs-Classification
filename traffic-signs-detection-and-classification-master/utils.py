import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import csv
import numpy as np


#plot images
def plotImages(images_arr, cmap=None):
    fig, axes = plt.subplots(2, 11, figsize=(20,8))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


 # This function will be used only in the notebook
def classDistribution(classLabels):
    '''
    Used to plot a histogram of sign classes
    '''
    # Count number of occurrences of each value
    classCount = np.bincount(classLabels.reshape(-1))
    classes = np.arange(len(classCount))
    plt.figure(figsize=(10, 4))
    plt.bar(classes, classCount, 0.8, color='red', label='Inputs per Class')
    plt.xlabel('Class Number')
    plt.ylabel('Samples per Class')
    plt.title('Distribution of Training Samples Among Classes')
    plt.show()


class Dataset:
    def __init__(self, data_dir, labels_path, data='train'):
        self.X = []
        self.y = []

        data_dir = Path(data_dir)
        with open(labels_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                label = row[-2]
                img_name = Path(row[-1]).name
                if data == 'train':
                    img_path = data_dir/label/img_name
                else:
                    img_path = data_dir/img_name
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.X.append(img)
                self.y.append(int(label))

        self.len = len(self.y)    # Number of samples in the dataset (accessible through len(dataset))
        self.indices = list(range(self.len)) # List of indices used to pick samples in a random order
    def __len__(self):
        return self.len
    
    def next_sample(self):
        if self.index == self.len: # If we arrived at the end of the dataset...
            self.index = 0 # then reset the pointer
            np.random.shuffle(self.indices) # and shuffle the dataset
        i = self.index
        i = self.indices[i]
        self.index += 1
        return (list(self.X[i]), list(self.y[i]))

def data_info(data_dir, train_dir, test_dir):
    data_dir = Path(data_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    class_paths = [dI for dI in train_dir.iterdir() if dI.is_dir()]

    total_train = 0
    total_test = len(list(test_dir.iterdir())) - 1
    total_classes = len(class_paths)

    for class_path in class_paths:
      total_train += len(list(class_path.iterdir()))

    print('total classes', total_classes)
    print('total train',total_train)
    print('total test',total_test)

# d = Dataset("data/gtsrb-german-traffic-sign/Train/", "data/gtsrb-german-traffic-sign/Train.csv")