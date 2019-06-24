from utils import conv_2d , maxpool , flatten , dense , relu , softmax , dropout,normalize_inputs
import tensorflow as tf
def build_model(input_x , keep_prob , training):
    
    if training != 1:
        input_x_aug = tf.image.random_hue(input_x, 0.08)
        input_x_aug = tf.image.random_saturation(input_x_aug, 0.6, 1.6)
        input_x_aug = tf.image.random_brightness(input_x_aug, 0.05)
        input_x_aug = tf.image.random_contrast(input_x_aug, 0.7, 1.3)
    else:
        input_x_aug = input_x
        
    input_x_normalized = normalize_inputs(input_x)
    
    conv1 = conv_2d(input_x ,[5,5,3,6],padding="SAME")
    conv1 = dropout(conv1 , keep_prob)
    conv1 = relu(conv1)
    maxpool1 =  maxpool(conv1,kernel=[1,2,2,1])
    
    conv2 = conv_2d(maxpool1 ,[5,5,6,16],padding="VALID")
    conv2 = dropout(conv2 , keep_prob)
    conv2 = relu(conv2)
    maxpool2 =  maxpool(conv2,kernel =[1,2,2,1])
    
    flat = flatten(maxpool2)
    
    dense_1 = dense(flat,400,120,"Dense1")
    dense_1 =  dropout(dense_1 , keep_prob)
    dense_1 = relu(dense_1)
    
    dense_2 = dense(dense_1,120,84,"Dense2")
    dense_2 =  dropout(dense_2 , keep_prob)
    dense_2 = relu(dense_2)
    
    output = dense(dense_2,84,10,"output")
    
    return output