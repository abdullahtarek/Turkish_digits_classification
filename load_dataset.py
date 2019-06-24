from glob import glob
import cv2
import numpy as np
from random import shuffle


def load_dataset(Dataset_paths,img_size,num_channels,num_classes):
    Dataset_paths = Dataset_paths.strip()
    if Dataset_paths[-1] !="/":
        Dataset_paths+="/"
    Dataset_paths+="*/*"
    imgs_path = glob(Dataset_paths)
    shuffle(imgs_path)
    imgs =[]
    labels = []

    for img_path in imgs_path:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img , (img_size,img_size))

        label = np.zeros(num_classes)
        label[int(img_path.split('/')[-2])]=1

        imgs.append(img)
        labels.append(label)

    imgs = np.array(imgs)
    labels = np.array(labels)
    
    return imgs , labels

def train_test_split(data , labels , train_percentage):
    train_data   =  data[0:int(len(data)*0.8)]
    train_labels =  labels[0:int(len(labels)*0.8)]

    val_data   =  data[int(len(data)*0.8):]
    val_labels =  labels[int(len(labels)*0.8):]
    
    return train_data,train_labels,val_data,val_labels