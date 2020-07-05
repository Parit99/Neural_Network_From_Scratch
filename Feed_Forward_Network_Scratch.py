#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
x=np.array([1,2,3])
weights=np.array([0.2,0.8,-0.5])
bias=2.0
output=np.dot(x,weights.T)+bias
print(output)


# In[9]:


inputs=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases=[2,3,0.5]
layer_op=[]
for neuron_weights,neuron_bias in zip(weights,biases):
    neuron_op=0
    for n_input,weight in zip(inputs,neuron_weights):
        neuron_op+=n_input*weight
    neuron_op+=neuron_bias
    layer_op.append(neuron_op)
print(layer_op)


# In[10]:


output=np.dot(weights,inputs)+biases
print(output)


# In[15]:


#For layer 1 neurons
inputs=np.array([[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]])
weights=np.array([[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]])
layer1_output=np.dot(inputs,weights.T)+biases
print(layer1_output)


# In[17]:


#weight2 is of layer2
weight2=np.array([[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]])
biases2=[-1,2,-0.5]
layer2_output=np.dot(layer1_output,weight2.T)+biases2
print(layer2_output)


# In[11]:


#input data for row
import numpy as np
import nnfs
nnfs.init()




X_train=np.array([[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]])

class Dense_layer:
    def __init__(self,inputs,n_neurons):
        self.inputs=inputs
        #print(self.inputs)
        n_inputs=inputs.shape[1]
        self.weights=0.10*np.random.randn(n_inputs,n_neurons)#randn gaussian distribution
        self.biases=np.zeros((1,n_neurons))
        
        
    def forward(self):
        self.output=np.dot(self.inputs,self.weights)+self.biases
        return self.output
class Relu:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
        
        
        
        
layer1=Dense_layer(X_train,5)
op=layer1.forward()
layer2=Dense_layer(op,2)
op1=layer2.forward()
print(op1)


# In[3]:


inputs=[0,2,-1,3.3,-2.7,1.1,2.2,-100]


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
def create_data(points,classes):
    X=np.zeros((points*classes,2))
    y=np.zeros(points*classes,dtype='uint8')
    for class_no in range(classes):
        ix=range(points*class_no,points*(class_no+1))
        r=np.linspace(0.0,1,points)
        t=np.linspace(class_no*4,(class_no+1)*4,points)+np.random.randn(points)*0.2
        X[ix]=np.c_[r*np.sin(t*2.5)+r*np.cos(t*2.5)]
        y[ix]=class_no
    return X,y
X,y=create_data(100,3)
plt.scatter(X[:,0],X[:,1])


# In[8]:


from nnfs.datasets import spiral_data
X,y=spiral_data(100,3)


# In[16]:


layer1=Dense_layer(X,5)
activation1=Relu()
op=layer1.forward()
#print(op)
activation1.forward(op)
print(activation1.output)


# In[ ]:




