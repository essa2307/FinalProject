
'''
     /
    /
\  /
 \/
'''
'''
Author: Esha Sarkar
Content: Discrimination Control Using Deep Neural Network
'''
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
XTrain = mnist.train.images
YTrain = mnist.train.labels
XTest = mnist.test.images
YTest = mnist.test.labels
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
XTrain = mnist.train.images[:10000]
YTrain = mnist.train.labels[:10000]
XTest = mnist.test.images
YTest = mnist.test.labels
# Another kind of dataset
def BWTransformMNIST (Percentage,OrigX,OrigY):
    SelectedImages = np.random.choice(int(OrigX.shape[0]),int((Percentage/100)*OrigX.shape[0]),replace=False) #Choose random images to insert sensitive information
    print ('No. of images changed are', SelectedImages.shape[0])
    SubX = 1-OrigX[SelectedImages]
    
    
    SubY = OrigY[SelectedImages]
    SenYCol = np.zeros((OrigY.shape[0],2))
    NewY = np.concatenate((OrigY,SenYCol),axis=1)

    for num,i in enumerate(SelectedImages):
        OrigX[i]=SubX[num]
        NewY[i][11]=1
    NewX = OrigX
    for i in range(int(OrigX.shape[0])):
        if i not in SelectedImages:
            NewY[i][10]=1
    return (SelectedImages, NewX,NewY)

s,XTrainPrime, YTrainPrime = BWTransformMNIST(50,XTrain,YTrain)
_,XTestPrime, YTestPrime = BWTransformMNIST(50,XTest,YTest)
XTrain = mnist.train.images[:10000]
YTrain = mnist.train.labels[:10000]
XTest = mnist.test.images
YTest = mnist.test.labels
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 12]) # 0-9 digits recognition => 1 class + verification of sensitive information


'''-----------------------------------------Encoder Training----------------------------------------------------'''
input_layer = tf.reshape(x,[-1, 28, 28, 1])    #-1 infers the shape according to the batch size user gives

  # Convolutional Layer #1
Econv1Temp = tf.layers.conv2d(
    inputs=input_layer,
    filters=1,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None,
    name=None)
    
Econv1BN = tf.layers.batch_normalization(
    inputs=Econv1Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None)
    
Econv1 = tf.nn.relu(Econv1BN)

Econv2Temp = tf.layers.conv2d(
    inputs=Econv1,
    filters=1,
    kernel_size=[3, 3],
    padding="same",
    activation=None,
    reuse=None)
    
Econv2BN = tf.layers.batch_normalization(
    inputs=Econv2Temp,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    reuse=None)
    
EncImagesTemp = tf.nn.relu(Econv2BN)

c = tf.reduce_max(EncImagesTemp)
d = tf.reduce_min(EncImagesTemp)
#The Encodings need to be scaled so that the final values are between 0 and 1 (like original MNIST)
EncImages = tf.divide((EncImagesTemp-d),(c-d))
#EncoderParameters = [W1,b1,W2,b2,W3,b3]
EncoderParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
'''---------------------------------------Classifier Training----------------------------------------'''
InputLayer = tf.reshape(EncImages,[-1, 28, 28, 1]) 
print ("One")
    #-1 infers the shape according to the batch size user gives

  # Convolutional Layer #1
Cconv1Temp = tf.layers.conv2d(
inputs=InputLayer,
filters=1,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv1BN = tf.layers.batch_normalization(
inputs=Cconv1Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv1 = tf.nn.relu(Cconv1BN)
Cfc3 = tf.layers.dense(inputs=Cconv1,units=1000,activation=tf.nn.relu)
dim = int(Cfc3.shape[1]*Cfc3.shape[2]*Cfc3.shape[3])
Cfc3Flat = tf.reshape(Cfc3,[-1,dim])
    #print (fc3Flat.shape)
    
  # Fully Connected Layer #1
OutputLayer1 = tf.layers.dense(inputs=Cfc3Flat,units=10,activation=None)
EncoderLen = len(EncoderParameters)
ClassifierParameters1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[EncoderLen:]

  # Convolutional Layer #1
Cconv2Temp = tf.layers.conv2d(
inputs=InputLayer,
filters=1,
kernel_size=[3, 3],
padding="same",
activation=None)
    
Cconv2BN = tf.layers.batch_normalization(
inputs=Cconv1Temp,
axis=-1,
momentum=0.99,
epsilon=0.001,trainable=True)
    
Cconv2 = tf.nn.relu(Cconv1BN)
Cfc3 = tf.layers.dense(inputs=Cconv2,units=300,activation=tf.nn.relu)
dim = int(Cfc3.shape[1]*Cfc3.shape[2]*Cfc3.shape[3])
Cfc3Flat = tf.reshape(Cfc3,[-1,dim])
    #print (fc3Flat.shape)
    
  # Fully Connected Layer #1
OutputLayer2 = tf.layers.dense(inputs=Cfc3Flat,units=2,activation=None)
Classifier1Len = len(ClassifierParameters1)
ClassifierParameters2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[Classifier1Len:]
Classifier1Len2 = len(ClassifierParameters2)


#print (ClassifierParameters)
    
  # Optimization
learning_rateU = 0.0001
learning_rateV = 0.0001
learning_rateE1 = 0.0001
training_epochs = 10
batch_size = 10
alpha = 1/16
beta = 1
print ('Entering Optimization')
    
predV = tf.nn.tanh(OutputLayer1)
predU = tf.nn.tanh(OutputLayer2)

CostV = tf.reduce_mean(-tf.reduce_sum(y[:,:10]*tf.log(tf.clip_by_value(predV,1e-5,1.0)), reduction_indices=1))
CostU = tf.reduce_mean(-tf.reduce_sum(y[:,10:]*tf.log(tf.clip_by_value(predU,1e-5,1.0)), reduction_indices=1))


cost = CostU - (alpha*CostV)
#LookSimilarCost = tf.sqrt(tf.reduce_sum((x-EncImages)**2))

ClassifierOptimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rateU).minimize(CostU,var_list=ClassifierParameters2)
ClassifierOptimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rateV).minimize(CostV,var_list=ClassifierParameters1)
EncoderOptimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rateE1).minimize(-cost,var_list=EncoderParameters)
#EncoderOptimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rateE2).minimize(-cost,var_list=EncoderParameters)
correct_predictionV = tf.equal(tf.argmax(predV, 1), tf.argmax(y[:,:10], 1))
correct_predictionU = tf.equal(tf.argmax(predU, 1), tf.argmax(y[:,10:], 1))

#print ((tf.argmax(y[:,10:], 1)).eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime}))
#a=(tf.argmax(predU, 1)).eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime})
#print (XTestPrime.shape)

AccuracyU = tf.reduce_mean(tf.cast(correct_predictionU, tf.float32))
AccuracyV = tf.reduce_mean(tf.cast(correct_predictionV, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print (AccuracyV.shape)

for epoch in range(training_epochs):
    avg_cost1 = 0.
    avg_cost2 = 0.
    avg_cost3 = 0.
    avg_cost4 = 0.
    total_batch = int(int(XTrainPrime.shape[0])/batch_size)
    print ("Epoch=",epoch,'\n')
    #print (total_batch)
    for i in range(total_batch):
        #print ('i=',i)
        batch_xs = XTrainPrime[(i*batch_size):((i+1)*batch_size)]
        batch_ys = YTrainPrime[(i*batch_size):((i+1)*batch_size)]
        _, c3 = sess.run([EncoderOptimizer1, cost], feed_dict={x: batch_xs, y: batch_ys})
        #_, c4 = sess.run([EncoderOptimizer2, LookSimilarCost], feed_dict={x: batch_xs, y: batch_ys})
    avg_cost3 += c3 / total_batch
    #avg_cost4 += c4 / total_batch
    print (avg_cost3) 
    
    for i in range(total_batch):
        #print ('i=',i)
        batch_xs = XTrainPrime[(i*batch_size):((i+1)*batch_size)]
        batch_ys = YTrainPrime[(i*batch_size):((i+1)*batch_size)]
        _, c2 = sess.run([ClassifierOptimizer1, CostV], feed_dict={x: batch_xs, y: batch_ys})
        _, c1 = sess.run([ClassifierOptimizer2, CostU], feed_dict={x: batch_xs, y: batch_ys})

    avg_cost1 += c1 / total_batch
    avg_cost2 += c2 / total_batch

    print (avg_cost1,avg_cost2)
    #print (avg_cost3)


    #if (epoch+1) % display_step == 0:

#        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print ("Optimization Finished!")
print ("Detection rate of Sensitive features (on Test Data):", AccuracyU.eval(session=sess,feed_dict={x: XTrainPrime[:1000], y: YTrainPrime[:1000]}))
print ("Accuracy of Utility (on Test Data):", AccuracyV.eval(session=sess,feed_dict={x: XTrainPrime[:1000], y: YTrainPrime[:1000]}))
EncTest = (EncImages.eval(session=sess,feed_dict={x:XTestPrime,y:YTestPrime})).reshape((-1,784))
print ('Shape of Enc Test', EncTest.shape)
#Training on a better Classifier
# model.add(Conv2D(32, (3, 3), 
                 # padding='valid', 
                 # input_shape=x_train.shape[1:],
                 # activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
sess.close()
'''---------------------------------------Good Classifier Training----------------------------------------'''
x1 = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y1 = tf.placeholder(tf.float32, [None, 12]) # 0-9 digits recognition => 1 class + verification of sensitive information
InputLayer1 = tf.reshape(x1,[-1, 28, 28, 1]) 
print ("Two")
    #-1 infers the shape according to the batch size user gives

  # Convolutional Layer #1
Cconv1Temp1 = tf.layers.conv2d(
inputs=InputLayer1,
filters=32,
kernel_size=[3, 3],
padding="valid",
activation=tf.nn.relu)
    
pool1 = tf.layers.max_pooling2d(inputs=Cconv1Temp1, pool_size=[2, 2], strides=2)

Cconv2Temp1 = tf.layers.conv2d(
inputs=pool1,
filters=32,
kernel_size=[3, 3],
padding="valid",
activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=Cconv2Temp1, pool_size=[2, 2], strides=2)
    

dim = int(pool2.shape[1]*pool2.shape[2]*pool2.shape[3])
Cfc3Flat = tf.reshape(pool2,[-1,dim])
    #print (fc3Flat.shape)
    
  # Fully Connected Layer #1
OutputLayer1Temp = tf.layers.dense(inputs=Cfc3Flat,units=512,activation=tf.nn.relu)
OutputLayer1 = tf.layers.dense(inputs=OutputLayer1Temp,units=10,activation=tf.nn.relu)

ClassifierParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[Classifier1Len2:]
learning_rateU = 0.0001
learning_rateV = 0.0001
learning_rateE1 = 0.0001
training_epochs = 10
batch_size = 100
i=0
alpha = 1/16
beta = 1
print ('Entering Optimization')
    
predV = tf.nn.tanh(OutputLayer1)
#predU = tf.nn.tanh(OutputLayer2)

CostV = tf.reduce_mean(-tf.reduce_sum(y1[:,:10]*tf.log(tf.clip_by_value(predV,1e-5,1.0)), reduction_indices=1))
#CostU = tf.reduce_mean(-tf.reduce_sum(y[:,10:]*tf.log(tf.clip_by_value(predU,1e-5,1.0)), reduction_indices=1))


#cost = CostU - (alpha*CostV)
#LookSimilarCost = tf.sqrt(tf.reduce_sum((x-EncImages)**2))

#ClassifierOptimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rateU).minimize(CostU,var_list=ClassifierParameters2)
ClassifierOptimizer = tf.train.AdamOptimizer(learning_rate=learning_rateV).minimize(CostV,var_list=ClassifierParameters)
#EncoderOptimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rateE1).minimize(-cost,var_list=EncoderParameters)
#EncoderOptimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rateE2).minimize(-cost,var_list=EncoderParameters)
correct_predictionV = tf.equal(tf.argmax(predV, 1), tf.argmax(y1[:,:10], 1))
#correct_predictionU = tf.equal(tf.argmax(predU, 1), tf.argmax(y[:,10:], 1))

#print ((tf.argmax(y[:,10:], 1)).eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime}))
#a=(tf.argmax(predU, 1)).eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime})
#print (XTestPrime.shape)

#AccuracyU = tf.reduce_mean(tf.cast(correct_predictionU, tf.float32))
AccuracyV = tf.reduce_mean(tf.cast(correct_predictionV, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
c2 = sess.run([CostV], feed_dict={x1: EncTest[:10], y1: YTestPrime[:10]})
for epoch in range(training_epochs):
    avg_cost1 = 0.
    avg_cost2 = 0.
    avg_cost3 = 0.
    avg_cost4 = 0.
    total_batch = int(int(XTrainPrime.shape[0])/batch_size)
    print ("Epoch=",epoch,'\n')
    
    for i in range(total_batch):
        #print ('i=',i)
        batch_xs = EncTest[(i*batch_size):((i+1)*batch_size)]
		
        batch_ys = YTestPrime[(i*batch_size):((i+1)*batch_size)]
		
        _, c2 = sess.run([ClassifierOptimizer, CostV], feed_dict={x1: batch_xs, y1: batch_ys})
        #_, c1 = sess.run([ClassifierOptimizer2, CostU], feed_dict={x: batch_xs, y: batch_ys})

    #avg_cost1 += c1 / total_batch
    avg_cost2 += c2 / total_batch

    print (avg_cost2)
    #print (avg_cost3)
    #print ("Detection rate of Sensitive features (on Test Data):", AccuracyU.eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime}))
    #print ("Accuracy of Utility (on Test Data):", AccuracyV.eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime}))

    #if (epoch+1) % display_step == 0:

#        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print ("Optimization Finished!")
print ("Accuracy of Utility (on Test Data):", AccuracyV.eval(session=sess,feed_dict={x1: XTestPrime[:1000], y1: YTestPrime[:1000]}))
