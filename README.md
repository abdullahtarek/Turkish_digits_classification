# Turkish_digits_classification

This project classifies images of Turkish digits using CNN developed by tensorflow. The archecture of the CNN used is a Lnet model but with some tweaks, like replacing the activation function with relu and adding dropouts.

## Requirments
packages used:
* tensorflow
* opencv-python
* numpy
* matplotlib

You can install all the requirements with the bellow command   
pip install -r requirements.txt

## Steps to run inference
* clone the repository
* cd Turkish_digits_classification  
* Do one of the following   
1. Run inference through terminal
   * run --> python inference.py PATH_TO_IMG
   * example: python inference Dataset/2/IMG_1120.JPG
   * ouput will be the last line of output after warning messages. It will tell the predicted label for the given image   
2. Run inference through jupyter notebook
   * open inference.ipynb 
   * change the image path and run the code cells 
   * It will display the image and the correspoing label
   

## Steps to run training
* clone the repository
* cd Turkish_digits_classification
1. Train Through terminal 
* open train.py and edit any path if needed (you won't have to edit any path if you left the datset folder as is)
* run --> python train.py
2. Train through jupyter notebook
* open train.ipynb
* you can see my chosen paramters and the dataset path, you can change them if needed
* run the cells you will find the model started training and saving checkpoints


## Preprocessing
a couple of image preprocessing techniques were used while training and they are:
* random_flip_left_right
* random_saturation
* random_brightness
* random_contrast


## Accuracy reached 
* I reached arroung 95 % accuracy in both the training and arround 94% testing set.

