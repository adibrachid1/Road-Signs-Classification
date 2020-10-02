# Road-Signs-Classification
Road-Signs-Classification Machine Learning

A car company’s commercial project would be a combination of detection and classification of road signs inside the car software. This project is highly recommended for autonomous cars and even to automate some car functions such as alerting drivers on a limit speed or other road signs.
However, in this project, the objective will be to work on only classifying road signs into their correct classes ex: speed limit, no stopping, no entry… The difficulty can increase by knowing more information about these classes ex: speed limit value, maximum height value… and then, by detecting the road signs as a further step. This problem can be considered as a computer vision problem so deep learning may be required to solve the classification in order to extract features from the images and use them to correctly classify the image to its exact class. 

# Dataset

There are many data sets online related to traffic signs, but we chose to work on the ‘German Traffic Sign Benchmark’ which is a dataset containing single image in a multiclass environment. It has 43 different traffic sign classes labeled from 0 to 42.
Dataset can be downloaded using the following URL link from Kaggle website:
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
The dataset contains:
</br>3 subfolders:
 - Meta: contains metadata of the different classes.
 - Test: contains around 25000 images to be tested which are from different classes.
 - Train: contains a directory for each class, containing a number of images for the specified class.
</br>3 csv label files:
 - Meta.csv: labels of the Meta folder
 - Test.csv: labels of the Test folder
 - Train.csv: labels of the Train folder

# Run Steps

First of all, we create a new directory called data and unzip the downloaded file from Kaggle which contains 3 csv files and 3 directories (Test, Train and Meta):
 - Run the python script augmentation.py which will create a new directory called augmented under data having the train directory with augmented data.
 - In the Jupyter notebook containing the code you might have to modify the path of the cur path to your project directory.

Project report is attached
