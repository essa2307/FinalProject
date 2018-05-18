
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
XTrain = mnist.train.images
YTrain = mnist.train.labels
XTest = mnist.test.images
YTest = mnist.test.labels
# Another kind of dataset
def MoreTransformMNIST (OrigX,OrigY):


    samples = int(OrigX.shape[0])
    SelectedImages = np.random.choice(samples,int(OrigX.shape[0]),replace=False) #Choose random images to insert sensitive information

    print ('No. of images changed are', SelectedImages.shape[0])
    s = SelectedImages.shape[0]
    SelectedImages1 = SelectedImages[:int(s/2)]
    SelectedImages2 = SelectedImages[int(s/2):]
    
    SubX1 = OrigX[SelectedImages1]
    SubX2 = OrigX[SelectedImages2]
    for o in range(784):
        if o % 13 == 0:
            SubX1[:,o] = 1
        if (o%10) == 0:               
            SubX2[:,o] = 1
        #elif o>=364 and o<378:
        #    SubX1[:,o] = 1



    
    
    SenYCol = np.zeros((OrigY.shape[0],2))
    NewY = np.concatenate((OrigY,SenYCol),axis=1)
    NewX = OrigX
# If plus sign then Y = 01 else Y = 10
    for num,i in enumerate(SelectedImages1):
        NewX[i]=SubX1[num]
        NewY[i][11]=1
    for num,i in enumerate(SelectedImages2):
        NewX[i]=SubX2[num]
        NewY[i][10]=1
    return (SelectedImages1, SelectedImages2, NewX, NewY)

s1,s2,XTrainPrime, YTrainPrime = MoreTransformMNIST(XTrain,YTrain)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
XTrain = mnist.train.images
YTrain = mnist.train.labels
XTest = mnist.test.images
YTest = mnist.test.labels

'''
The architecture is same but there are some added constraints. Now all the encoded images have to have zero mean 
and unit variance. This is done to not saturate the encoder to a constant solution and will try to look similar 
to the original digit without any attribute.
But doing the training on a new dataset
'''

alpha = (1/16)

    
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 12]) # 0-9 digits recognition => 1 class


'''-----------------------------------------Encoder Training----------------------------------------------------'''
W1 = tf.Variable(tf.random_normal([784, 300], mean=0, stddev=1))
b1 = tf.Variable(tf.random_normal([300], mean=0, stddev = 1))
W2 = tf.Variable(tf.random_normal([300, 784],mean=0, stddev=1))
b2 = tf.Variable(tf.random_normal([784],mean=0, stddev=1))

Hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1);
TempLayer = tf.matmul(Hidden1, W2) + b2
MeanTempLayer  = tf.reduce_mean(TempLayer)
m = tf.reduce_mean(TempLayer,axis=1)
m = tf.convert_to_tensor([m]*784)
m = tf.reshape(m,(-1,784))
VarTempLayer  = tf.reduce_mean((TempLayer-m)**2)
EncImagesTemp = tf.nn.tanh(tf.matmul(Hidden1, W2) + b2);
#These are the encodings of the images

c = tf.reduce_max(EncImagesTemp)
d = tf.reduce_min(EncImagesTemp)
#The Encodings need to be scaled so that the final values are between 0 and 1 (like original MNIST)
EncImages = tf.divide((EncImagesTemp-d),(c-d))
L2Norm = tf.sqrt(tf.reduce_sum((EncImages - x)**2))

EncoderParameters = [W1,b1,W2,b2]
    
'''---------------------------------------Classifier Training----------------------------------------'''
#W3,b3,W4,b4 helps in classifying utility
W3 = tf.Variable(tf.random_normal([784, 300], mean=0, stddev=1))
b3 = tf.Variable(tf.random_normal([300], mean=0, stddev = 1))
W4 = tf.Variable(tf.random_normal([300, 10]))
b4 = tf.Variable(tf.random_normal([10]))

Hidden2 = tf.nn.relu(tf.matmul(EncImages, W3) + b3);
OutputLayer1 = tf.matmul(Hidden2, W4) + b4;
ClassifierParameters1 = [W3,b3,W4,b4]

#W5,b5,W6,b6 helps in classifying sensitive features
W5 = tf.Variable(tf.random_normal([784, 300], mean=0, stddev=1))
b5 = tf.Variable(tf.random_normal([300], mean=0, stddev = 1))
W6 = tf.Variable(tf.random_normal([300, 2]))
b6 = tf.Variable(tf.random_normal([2]))

Hidden3 = tf.nn.relu(tf.matmul(EncImages, W5) + b5);
OutputLayer2 = tf.matmul(Hidden3, W6) + b6;
ClassifierParameters2 = [W5,b5,W6,b6]    
  # Optimization
learning_rateU = 0.0001
learning_rateV = 0.0001
learning_rateE = 0.0001
training_epochs = 30
batch_size = 100
display_step = 1
lambda1 = 0.05
lambda2 = 0.05
lambda3 = 0.05
print ('Entering Optimization')
    
predV = tf.nn.softmax(OutputLayer1)
predU = tf.nn.softmax(OutputLayer2)

CostV = tf.reduce_mean(-tf.reduce_sum(y[:,:10]*tf.log(tf.clip_by_value(predV,1e-10,1.0)), reduction_indices=1))
CostU = tf.reduce_mean(-tf.reduce_sum(y[:,10:]*tf.log(tf.clip_by_value(predU,1e-10,1.0)), reduction_indices=1))

#Batch normalization will happen across a batch

#cost = CostU - (alpha*CostV) - (lambda1*MeanTempLayer) - lambda2*(VarTempLayer-1) - lambda3*(L2Norm)
cost = CostU - (alpha*CostV)- lambda3*(L2Norm)
#print (MeanTempLayer,VarTempLayer,L2Norm)
ClassifierOptimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rateV).minimize(CostV,var_list=ClassifierParameters1)
ClassifierOptimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rateU).minimize(CostU,var_list=ClassifierParameters2)
EncoderOptimizer = tf.train.AdamOptimizer(learning_rate=learning_rateE).minimize(-cost,var_list=EncoderParameters)

correct_predictionV = tf.equal(tf.argmax(predV, 1), tf.argmax(y[:,:10], 1))
correct_predictionU = tf.equal(tf.argmax(predU, 1), tf.argmax(y[:,10:], 1))


AccuracyU = tf.reduce_mean(tf.cast(correct_predictionU, tf.float32))
AccuracyV = tf.reduce_mean(tf.cast(correct_predictionV, tf.float32))
#print ("Detection rate of Sensitive features (on Test Data):", AccuracyU.eval(session=sess,feed_dict={x: XTrainPrime, y: YTrainPrime}))
#print ("Accuracy of Utility (on Test Data):", AccuracyV.eval(session=sess,feed_dict={x: XTrainPrime, y: YTrainPrime}))

sess = tf.Session()
sess.run(tf.global_variables_initializer())



EA = []
PA = []
alpha = 1/16
lambda3 = 0.5
training_epochs = 20
XTrial = XTrainPrime
YTrial = YTrainPrime
batch_size = 100
learning_rateU = 0.0005
learning_rateV = 0.0001
learning_rateE = 0.0001
for epoch in range(training_epochs):
    avg_cost1 = 0.
    avg_cost2 = 0.
    avg_cost3 = 0.
    total_batch = int(int(XTrainPrime.shape[0])/batch_size)
    print (total_batch)
    print ("Epoch=",epoch)
    
    for i in range(total_batch):
        #print ('esha')
        batch_xs = XTrainPrime[(i*batch_size):((i+1)*batch_size)]
        batch_ys = YTrainPrime[(i*batch_size):((i+1)*batch_size)]
        _, c3 = sess.run([EncoderOptimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
        #print (c3)
    avg_cost3 += c3 / total_batch
    print ('Encoder cost', avg_cost3) 
    
    for i in range(total_batch):

        batch_xs = XTrainPrime[(i*batch_size):((i+1)*batch_size)]
        batch_ys = YTrainPrime[(i*batch_size):((i+1)*batch_size)]
        _, c1 = sess.run([ClassifierOptimizer1, CostV], feed_dict={x: batch_xs, y: batch_ys})
        _, c2 = sess.run([ClassifierOptimizer2, CostU], feed_dict={x: batch_xs, y: batch_ys})
        #print (c1)
        #print (c2)
    avg_cost1 += c1 / total_batch
    avg_cost2 += c2 / total_batch
    print ('Utility cost:',avg_cost1,'Private attribute cost:',avg_cost2)
    EpochAccuracy = AccuracyV.eval(session=sess,feed_dict={x: XTrial, y: YTrial})
    SenAcc = AccuracyU.eval(session=sess,feed_dict={x: XTrial, y: YTrial})
    print ('Epoch Accuracy',EpochAccuracy)
    print ('Accuracy with which sensitive features are detected', SenAcc)
    EA.append(AccuracyV.eval(session=sess,feed_dict={x: XTrainPrime, y: YTrainPrime}))
    PA.append(AccuracyU.eval(session=sess,feed_dict={x: XTrainPrime, y: YTrainPrime}))
#     if (EpochAccuracy > 0.6):
#         #print ('yippie')
#         learning_rateU = 0.001
#         learning_rateV = 0.0005
#         learning_rateE = 0.001
#     if (EpochAccuracy > 0.8):
#         #print ('yippie')
#         learning_rateU = 0.0001
#         learning_rateV = 0.0001
#         learning_rateE = 0.0005
#         alpha = 1.5
    print ('\n')
print ("Optimization Finished!")
correct_predictionV = tf.equal(tf.argmax(predV, 1), tf.argmax(y[:,:10], 1))
correct_predictionU = tf.equal(tf.argmax(predU, 1), tf.argmax(y[:,10:], 1))

#print ((tf.argmax(y[:,10:], 1)).eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime}))
#a=(tf.argmax(predU, 1)).eval(session=sess,feed_dict={x: XTestPrime, y: YTestPrime})
#print (XTestPrime.shape)

AccuracyU = tf.reduce_mean(tf.cast(correct_predictionU, tf.float32))
AccuracyV = tf.reduce_mean(tf.cast(correct_predictionV, tf.float32))
print ("Detection rate of Sensitive features (on Test Data):", AccuracyU.eval(session=sess,feed_dict={x: XTrial, y: YTrial}))
print ("Accuracy of Utility (on Training Data):", AccuracyV.eval(session=sess,feed_dict={x: XTrial, y: YTrial}))



n=s2[0]
X=XTrainPrime[n]
TempImage = X.reshape((28,28))
plt.imshow(TempImage, cmap='gray')
plt.show()
X1=XTrain[n]
TempImage1 = X1.reshape((28,28))
plt.imshow(TempImage1, cmap='gray')
plt.show()
m=s1[0]
X=XTrainPrime[m]
TempImage = X.reshape((28,28))
plt.imshow(TempImage, cmap='gray')
plt.show()
X1=XTrain[m]
TempImage1 = X1.reshape((28,28))
plt.imshow(TempImage1, cmap='gray')
plt.show()
p=EncImages.eval(session=sess,feed_dict={x:XTrainPrime,y:YTrainPrime})
q1=p[n].reshape((28,28))
q2=p[m].reshape((28,28))

plt.imshow(q1, cmap='gray')
plt.show()
plt.imshow(q2, cmap='gray')
plt.show()