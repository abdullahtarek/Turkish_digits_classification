import numpy as np
import tensorflow as tf
from model import build_model
import tensorflow as tf


def trainer(batch_size,num_of_epochs,dropout_prob, train_data,train_labels,val_data,val_labels):
    
    
    tf.reset_default_graph()
    img_size = train_data.shape[1]
    num_channels = train_data.shape[3]
    num_classes = train_labels.shape[1]
    
    keep_prob =  tf.placeholder(tf.float32,name="keep_prob")
    train_flag = tf.placeholder(tf.bool,name="train_flag")
    
    input_x = tf.placeholder(tf.float32,shape= [None,img_size,img_size,num_channels],name="input_data")
    GT_y = tf.placeholder(tf.float32,shape= [None,num_classes],name="output_GT")

    predictions = build_model(input_x,keep_prob , train_flag)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels = GT_y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    y_pred_cls = tf.argmax(predictions, dimension=1)
    y_true_cls = tf.argmax(GT_y, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(num_of_epochs):
            train_accuracy=[]
            train_cost=[]
            for i in range(0,len(train_data),batch_size):
                x_batch, y_true_batch = train_data[i:i+batch_size] , train_labels[i:i+batch_size]
                _,trainig_batch_cost,training_batch_acc= sess.run([optimizer,cost,accuracy] , feed_dict = {input_x:x_batch , GT_y: y_true_batch , keep_prob:dropout_prob , train_flag:1})

                train_cost.append(trainig_batch_cost)
                train_accuracy.append(training_batch_acc)

            train_accuracy = np.array(train_accuracy)
            val_cost,val_acc= sess.run([cost,accuracy] , feed_dict = {input_x:val_data , GT_y: val_labels, keep_prob:1, train_flag:0})

            print("epoch {}: Training accuracy:: {} , Validation accuracy:: {} ".format(epoch, train_accuracy.mean() , val_acc)) 
            
            if epoch %5==0 and epoch !=0:
                saver = tf.train.Saver()
                saver.save(sess,'./ckpt/checkpoint_'+str(epoch)+'.ckpt')
                tf.train.write_graph(sess.graph.as_graph_def(), './ckpt/', 'checkpoint_'+str(epoch)+'.pbtxt', as_text=True)

    
    