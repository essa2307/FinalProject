import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
XTrain = mnist.train.images
YTrain = mnist.train.labels
XTest = mnist.test.images
YTest = mnist.test.labels


#TempImage = XTrain[0].reshape((28,28))
#plt.imshow(TempImage, cmap='gray')
#plt.show()

#------------------Make a modified Dataset with a dot inserted in random images across the training and test dataset
def TransformMNIST (OrigX,OrigY):
    OrigX = XTrain
    OrigY = YTrain
    SelectedImages = np.random.choice(int(OrigX.shape[0]),int(OrigX.shape[0]/2),replace=False) #Choose random images to insert sensitive information

    SubX = OrigX[SelectedImages]
    SubX[:,729] = 1
    
    SubY = OrigY[SelectedImages]
    SenYCol = np.zeros((OrigY.shape[0],2))
    NewY = np.concatenate((OrigY,SenYCol),axis=1)

    for num,i in enumerate(SelectedImages):
        OrigX[i]=SubX[num]
        NewY[i][11]=1
    NewX = OrigX
    for i in range(int(XTrain.shape[0])):
        if i not in SelectedImages:
            NewY[i][10]=1
    return (SelectedImages, NewX,NewY)

s,XTrainPrime, YTrainPrime = TransformMNIST(XTrain,YTrain)
_,XTestPrime, YTestPrime = TransformMNIST(XTest,YTest)
#These are our dataset for the rest of the project

#n=s[0]
#X=XTrainPrime[n]
#TempImage = X.reshape((28,28))
#plt.imshow(TempImage, cmap='gray')
#plt.show()

'''Please see paper for architecture'''
XTrial = XTrainPrime[:10]
YTrial = YTrainPrime[:10]
alpha = (1/16)
SensitiveLabels = YTrial[:,10:12]
    
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 12]) # 0-9 digits recognition => 1 class


'''-----------------------------------------Encoder Training----------------------------------------------------'''
  # Input Layer
input_layer = tf.reshape(x,[-1, 28, 28, 1])    #-1 infers the shape according to the batch size user gives

  # Convolutional Layer #1
Econv1Temp = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv1Temp')
    
Econv1BN = tf.layers.batch_normalization(
    inputs=Econv1Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv1BN')
    
Econv1 = tf.nn.relu(Econv1BN)
    
  # Convolutional Layer #2
Econv2Temp = tf.layers.conv2d(
    inputs=Econv1,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv2Temp')
    
Econv2BN = tf.layers.batch_normalization(
    inputs=Econv2Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv2BN')
    
Econv2 = tf.nn.relu(Econv2BN)

  # Pooling Layer #1
    #pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #print (pool1.shape)
    
  # Convolutional Layer #3
Econv3Temp = tf.layers.conv2d(
    inputs=Econv2,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv3Temp')

Econv3BN = tf.layers.batch_normalization(
    inputs=Econv3Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv3BN')
    
Econv3 = tf.nn.relu(Econv3BN)
    
  # Convolutional Layer #4
Econv4Temp = tf.layers.conv2d(
    inputs=Econv3,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv4Temp')

Econv4BN = tf.layers.batch_normalization(
    inputs=Econv4Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv4BN')
    
Econv4 = tf.nn.relu(Econv4BN)

  # Pooling Layer #2
    #pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    #print (pool2.shape)
  # Convolutional Layer #5
Econv5Temp = tf.layers.conv2d(
    inputs=Econv4,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv5Temp')
    
Econv5BN = tf.layers.batch_normalization(
    inputs=Econv5Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv5BN')
    
Econv5 = tf.nn.relu(Econv5BN)
    
  # Convolutional Layer #6
Econv6Temp = tf.layers.conv2d(
    inputs=Econv5,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv6Temp')
    
Econv6BN = tf.layers.batch_normalization(
    inputs=Econv6Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv6BN')
    
Econv6 = tf.nn.relu(Econv6BN)

  # Pooling Layer #3
    #pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

  # Convolutional Layer #7
Econv7Temp = tf.layers.conv2d(
    inputs=Econv6,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv7Temp')

Econv7BN = tf.layers.batch_normalization(
    inputs=Econv7Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv7BN')
    
Econv7 = tf.nn.relu(Econv7BN)    
    
  # Convolutional Layer #8
Econv8Temp = tf.layers.conv2d(
    inputs=Econv7,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv8Temp')
    
Econv8BN = tf.layers.batch_normalization(
    inputs=Econv8Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv8BN')
    
Econv8 = tf.nn.relu(Econv8BN)    

  # Convolutional Layer #9
Econv9Temp = tf.layers.conv2d(
    inputs=Econv8,
    filters=16,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv9Temp')
    
Econv9BN = tf.layers.batch_normalization(
    inputs=Econv9Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv9BN')
    
Econv9 = tf.nn.relu(Econv9BN)    
    
  # Convolutional Layer #10
Econv10Temp = tf.layers.conv2d(
    inputs=Econv9,
    filters=8,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name='Econv10Temp')

Econv10BN = tf.layers.batch_normalization(
    inputs=Econv10Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None,
    name='Econv10BN')
    
Econv10 = tf.nn.relu(Econv10BN)   
    #print (conv10.shape)
    
  # Convolutional Layer #11
Econv11Temp = tf.layers.conv2d(
    inputs=Econv10,
    filters=1,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.tanh,
    reuse=None,
    name='Econv11Temp')
    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #EncImage = conv11Temp.eval(session=sess)
EncoderParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
'''---------------------------------------Classifier Training----------------------------------------'''
  # Input Layer
InputLayer = tf.reshape(Econv11Temp,[-1, 28, 28, 1]) 
print ("One")
    #-1 infers the shape according to the batch size user gives

  # Convolutional Layer #1
Cconv1Temp = tf.layers.conv2d(
inputs=InputLayer,
filters=256,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv1BN = tf.layers.batch_normalization(
inputs=Cconv1Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv1 = tf.nn.relu(Cconv1BN)
   
  # Convolutional Layer #2
Cconv2Temp = tf.layers.conv2d(
inputs=Cconv1,
filters=256,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv2BN = tf.layers.batch_normalization(
inputs=Cconv2Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv2 = tf.nn.relu(Cconv2BN)

  # Convolutional Layer #3
Cconv3Temp = tf.layers.conv2d(
inputs=Cconv2,
filters=256,
kernel_size=[3, 3],
padding="same",
activation=None)
  
Cconv3BN = tf.layers.batch_normalization(
inputs=Cconv2Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv3 = tf.nn.relu(Cconv3BN)

  # Pooling Layer #1
Cpool1 = tf.layers.max_pooling2d(inputs=Cconv3, pool_size=[2, 2], strides=2)
    
  # Convolutional Layer #4
Cconv4Temp = tf.layers.conv2d(
inputs=Cpool1,
filters=512,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv4BN = tf.layers.batch_normalization(
inputs=Cconv4Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv4 = tf.nn.relu(Cconv4BN)
    
  # Convolutional Layer #5
Cconv5Temp = tf.layers.conv2d(
inputs=Cconv4,
filters=512,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv5BN = tf.layers.batch_normalization(
inputs=Cconv5Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv5 = tf.nn.relu(Cconv5BN)

  # Convolutional Layer #7
Cconv7Temp = tf.layers.conv2d(
inputs=Cconv5,
filters=512,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv7BN = tf.layers.batch_normalization(
inputs=Cconv7Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv7 = tf.nn.relu(Cconv3BN)
print (Cconv7.shape)
  # Pooling Layer #2
Cpool7 = tf.layers.max_pooling2d(inputs=Cconv7, pool_size=[2, 2], strides=2)
    #print (pool7.shape)  
  # Fully Connected Layer #1
Cfc1 = tf.layers.dense(inputs=Cpool7,units=4096,activation=tf.nn.relu)
#print (fc1.shape)

  # Fully Connected Layer #2
Cfc2 = tf.layers.dense(inputs=Cfc1,units=4096,activation=tf.nn.relu)
#print (fc1.shape)
    
  # Fully Connected Layer #1
Cfc3 = tf.layers.dense(inputs=Cfc2,units=4096,activation=tf.nn.relu)
dim = int(Cfc3.shape[1]*Cfc3.shape[2]*Cfc3.shape[3])
Cfc3Flat = tf.reshape(Cfc3,[-1,dim])
    #print (fc3Flat.shape)
    
  # Fully Connected Layer #1
OutputLayer = tf.layers.dense(inputs=Cfc3Flat,units=12,activation=None)
EncoderLen = len(EncoderParameters)
ClassifierParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[EncoderLen:]
print (ClassifierParameters)
    
  # Optimization
learning_rate = 0.01
training_epochs = 2
batch_size = 2
display_step = 1
print ('Entering Optimization')
    
pred = tf.nn.softmax(OutputLayer)
predV = pred[:,:10]
predU = pred[:,11:]
CostV = tf.reduce_mean(-tf.reduce_sum(y[:,:10]*tf.log(predV), reduction_indices=1))
CostU = tf.reduce_mean(-tf.reduce_sum(y[:,11:]*tf.log(predU), reduction_indices=1))
cost = CostU - (alpha*CostV)
ClassifierOptimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,var_list=ClassifierParameters)
EncoderOptimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(-cost,var_list=EncoderParameters)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# #print (SensitiveLabels)
#     #print ("Three")
#     # Training cycle
for epoch in range(training_epochs):
	print (epoch)
	avg_cost = 0.
	total_batch = int(int(XTrial.shape[0])/batch_size)
	print ("Epoch=",epoch,'\n')
    #print (total_batch)

	for i in range(total_batch):
		batch_xs = XTrial[(i*batch_size):((i+1)*batch_size)]
		batch_ys = YTrial[(i*batch_size):((i+1)*batch_size)]
		_, c1 = sess.run([ClassifierOptimizer, cost], feed_dict={x: XTrial, y: YTrial})
		_, c2 = sess.run([EncoderOptimizer, cost], feed_dict={x: XTrial, y: YTrial})
		print ('Encoder: For i=',i,' cost =',c1)
		print ('Classifier: For i=',i,' cost =',c2)
    #avg_cost += c / total_batch
    #print (c)

    #if (epoch+1) % display_step == 0:

#        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print ("Optimization Finished!")

    
 




