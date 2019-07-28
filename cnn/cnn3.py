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
def pool(m, k):
  return tf.nn.max_pool(m, ksize=[1, k, k, 1],strides=[1, k, k, 1], padding='SAME')

def norm(m,size=4):
   return tf.nn.lrn(m,size,bias=1.0,alpha=0.001/9.0,beta =0.75) 
#construct the CNN neural network


m_image = tf.reshape(m, [-1,28,28,1])        

    
#First pool layer
l_pool1 = pool(m_image , 2)   
norm1 = norm(l_pool1,size=4)                               

W_conv2 = weight([5,5,1,192])
b_conv2 = bias([192])
#First convolutional layer
l_conv2 = tf.nn.relu(convolution_layer(norm1, W_conv2) + b_conv2)     
#Second pool layer
l_pool2 = pool(l_conv2 , 2)
norm2 = norm(l_pool2 , size=4)                                   

W_conv3 = weight([3,3,192,384])
b_conv3 = bias([384])
#second convolutional layer
l_conv3 = tf.nn.relu(convolution_layer(norm2, W_conv3) + b_conv3)     
norm3 = norm(l_conv3 , size=4) 

W_conv4 = weight([3,3,384,384])
b_conv4 = bias([384])
#third convolutional layer
l_conv4 = tf.nn.relu(convolution_layer(norm3 , W_conv4) + b_conv4)     
norm4 = norm(l_conv4 , size=4) 

W_conv5 = weight([3,3,384,256])
b_conv5 = bias([256])
#forth convolutional layer
l_conv5 = tf.nn.relu(convolution_layer(norm4 , W_conv5) + b_conv5)  
#fifth loop layer
l_pool5 = pool(l_conv5,2)   
norm5 = norm(l_pool5 , size=4) 

W_fc1 = weight([4*4*256, 4096])
b_fc1 = bias([4096])
l_norm5_vec = tf.reshape(norm5, [-1, 4*4*256])             
#First full-connected layer
l_norm5_vec = tf.nn.relu(tf.matmul(l_norm5_vec, W_fc1) + b_fc1)    
prob = tf.placeholder("float") 
#drop out layer
l_norm5_vec = tf.nn.dropout(l_norm5_vec, prob)                  

#Second full-connected layer
W_fc2 = weight([4096,4096])
b_fc2 = bias([4096])

vec2 = tf.reshape(l_norm5_vec, [-1, 4096])
vec2 = tf.nn.relu(tf.matmul(l_norm5_vec, W_fc2) + b_fc2)
vec2 =  tf.nn.dropout(vec2, prob) 

#out layer
out = tf.matmul(vec2,weight([4096,10])) + bias([10])

#corss entropy is the loss function
loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = n))     
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#Gradient descent    
prediction = tf.equal(tf.argmax(out,1), tf.argmax(n,1))    
#calculate accuracy
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))              
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
a=0
while a<5:
     for loop in range(10000):
         batch = mnist_dataset.train.next_batch(50)
         train_step.run(feed_dict={m: batch[0], n: batch[1], prob: 0.75})
         if loop%100 == 0: 
            acc = accuracy.eval(feed_dict={m:batch[0], n: batch[1], prob: 1.0})
     test_acc=accuracy.eval(feed_dict={m: mnist_dataset.test.images, n: mnist_dataset.test.labels, prob: 1.0})
     print("test accuracy",test_acc)
     a+=1
def predict(c):
  pre=tf.argmax (out, 1).eval(session=sess,feed_dict={m: mnist_dataset.test.images,n:mnist_dataset.test.labels,prob:1.0})
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