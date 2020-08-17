#!/usr/bin/env python
# coding: utf-8

# In[124]:



from __future__ import print_function
import pandas as pd 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold


#Read file
def read_file(file):
    wine = pd.read_csv(file)
    Data= wine.iloc[:,1:13]    
    Label = np.array(wine.iloc[:,0]).reshape(-1,1)
    
    return Data,Label

# One Hot Encoding. 
def code(data,label):
    oneHot = OneHotEncoder() 
    
    oneHot.fit(data) 
    x = oneHot.transform(data).toarray()   
   
    oneHot.fit(label) 
    y = oneHot.transform(label).toarray() 
    m, n = x.shape 
    
    return x,y,n

#Training model and Test model function
def run_train(train_x, train_y):
    
    
    with tf.Session() as session:
        
        print("\nStart training")
        session.run(init)
    
        avg_cost=0
        
 
        for epoch in range(epochs):
               
      
            session.run(optimizer, feed_dict = {x_p : train_x, y_p : train_y}) 
          
         
            c = session.run(cost, feed_dict = {x_p : train_x, y_p : train_y}) 
          
            
            
            cost_pod.append(sum(sum(c)))
            
                       
        
            if (epoch+1) % display_step == 0:
                
                
                print("Epoch:", '%04d' % (epoch+1), "cost=" + str(cost_pod[-1]))          
                
                  
        correct_prediction = tf.equal(tf.argmax(Y_hat, 1),tf.argmax(y_p, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
                  
        print("Accuracy:", accuracy.eval({x_p: X_test , y_p: y_test})*100, "%")
        
        
        


        
   
#Parameters
file = ""
alpha = 0.01
epochs = 500
display_step = 1

# tf Graph Input
x_p = tf.placeholder(tf.float32, [None, N]) 
y_p = tf.placeholder(tf.float32, [None, 3]) 

# Set model weights
W = tf.Variable(tf.zeros([N, 3])) 
b = tf.Variable(tf.zeros([3])) 
# Construct model
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(x_p, W), b)) 
# Minimize error using cross entropy
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = y_p) 
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate = alpha).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer() 
cost_pod=[]
                
                               

Data_1,Label_1=read_file("wine.csv")
X,Y,N = code(Data_1,Label_1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0) 


    
run_train(X_train, y_train)
    
#Plot Cost per Epoch     
plt.plot(list(range(epochs)), cost_pod) 
plt.xlabel('Epochs') 
plt.ylabel('Cost') 
plt.title('Decrease in Cost with Epochs') 
plt.show() 
  



# In[ ]:




