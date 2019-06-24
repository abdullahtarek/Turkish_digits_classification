from load_dataset import load_dataset, train_test_split
from trainer import trainer


Dataset_paths="Dataset/"
img_size,num_channels,num_classes = 28,3,10
train_percentage = 0.8
keep_prop= 0.75

batch_size = 32
num_of_epochs= 16

data , labels = load_dataset(Dataset_paths,img_size,num_channels,num_classes)
train_data,train_labels,val_data,val_labels = train_test_split(data,labels,train_percentage)
trainer(batch_size,num_of_epochs,keep_prop, train_data,train_labels,val_data,val_labels)