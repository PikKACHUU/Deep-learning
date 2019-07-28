import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

#Download and load dataset
mnist_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)     
m = tf.placeholder(tf.float32, shape=[None, 784])                       
n= tf.placeholder(tf.float32, shape=[None, 10]) 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#initialise weight and bias
 # m is the trained image and the n is the label of the image
#print dataset information
train_images = mnist_dataset.train.images
print('train_images',train_images.shape)
train_labels = mnist_dataset.train.labels
print('train_labels',train_labels.shape)
test_images = mnist_dataset.test.images
print('test_images',test_images.shape)
test_labels = mnist_dataset.test.labels
print('test_labels',test_labels.shape)

#define a function to initialise all weight W
def weight(s):
  initial = tf.truncated_normal(s, stddev=0.1)
  return tf.Variable(initial)

#define a function to initialise all bias variables b
def bias(s):
  initial = tf.constant(0.1, shape=s)
  return tf.Variable(initial)
  
#define a function to construct convolution layer
def convolution_layer(m, W):
  return tf.nn.conv2d(m, W, strides=[1, 1, 1, 1], padding='SAME')

#define a function to construct a pool layer
def pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#construct the CNN neural network
m_image = tf.reshape(m, [-1,28,28,1])        

W_conv1 = weight([5, 5, 1, 32])      
b_conv1 = bias([32])       
#First convolutional layer
l_conv1 = tf.nn.relu(convolution_layer(m_image, W_conv1) + b_conv1)     
#First pool layer
l_pool1 = pool(l_conv1)                                  

W_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
#Second convolutional layer
l_conv2 = tf.nn.relu(convolution_layer(l_pool1, W_conv2) + b_conv2)     
#Second pool layer
l_pool2 = pool(l_conv2)                                   

W_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])
l_pool2_vec = tf.reshape(l_pool2, [-1, 7*7*64])              
#First full-connected layer
l_fc1 = tf.nn.relu(tf.matmul(l_pool2_vec, W_fc1) + b_fc1)    

prob = tf.placeholder("float") 
#drop out layer
l_fc1_drop = tf.nn.dropout(l_fc1, prob)                  

W_fc2 = weight([1024, 10])
b_fc2 = bias([10])
#softmax layer
n_predict=tf.nn.softmax(tf.matmul(l_fc1_drop, W_fc2) + b_fc2)
#corss entropy is the loss function
loss = -tf.reduce_sum(n*tf.log(n_predict))     
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
#Gradient descent    
prediction = tf.equal(tf.argmax(n_predict,1), tf.argmax(n,1))    
#calculate accuracy
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))              
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
a=0
while a<3:
     for loop in range(10000):
         batch = mnist_dataset.train.next_batch(50)
         if loop%50 == 0:
             train_step.run(feed_dict={m: batch[0], n: batch[1], prob: 0.5})
             acc = accuracy.eval(feed_dict={m:batch[0], n: batch[1], prob: 1.0})
     test_acc=accuracy.eval(feed_dict={m: mnist_dataset.test.images, n: mnist_dataset.test.labels, prob: 1.0})
     print("test accuracy",test_acc)
     a+=1
def predict(c):
  pre=tf.argmax (n_predict, 1).eval(session=sess,feed_dict={m: mnist_dataset.test.images,n:mnist_dataset.test.labels,prob:1.0})
  print('the prediction of front {} images is: '.format(c))
  for i in range(c):
       print(pre[i],end=',')
       if int((i+1)%5) ==0:
         print('\t')
  return pre
predict(20)
#predict the label of front twenty images
def original(c):
    org = tf.argmax(n,1).eval(session=sess,feed_dict={m:mnist_dataset.test.images,n:mnist_dataset.test.labels,prob:1.0})
    print('the front {} images is: '.format(c))
    for i in range(c):
         print(org[i],end=',')
         if int((i+1)%5) ==0:
           print('\t')
    return org
original(20)

#original label of fron twenty images