import cv2
import numpy as np
import tensorflow as tf

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def inference(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img , (28,28))
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('Best_model/checkpoint_15.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('Best_model/'))
    
    with tf.Session() as sess:    
        saver = tf.train.import_meta_graph('Best_model/checkpoint_15.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('Best_model/'))

        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_data:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        op_to_restore = graph.get_tensor_by_name("output:0")
        predict =sess.run(op_to_restore,feed_dict={input_x: np.expand_dims(img,axis=0 ),
                                    keep_prob:1})
        
        print("The predicted class is : {}".format(softmax(predict).argmax()))