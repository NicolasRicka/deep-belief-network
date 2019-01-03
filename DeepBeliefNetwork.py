# -*- coding: utf-8 -*-
"""
Restricted Boltzmann Machines
-tensorflow code for a general RBM
-Phony data; to see whether it works on handcrafted data
-test on the mnist database (simply change n_v (here, n_v = 28*28) to something else to adapt to another dataset)

@author: RICKA
"""

phony_data_part = True
train_multiple_models = True


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Part I: Settin up the RBM


def sigmoid(x):    
    #activation function
    return 1/(1+np.exp(-x))



def init_model(n_v,n_h):
    #initialize a single RBM
    W  = np.random.normal(0, 0.1, [n_v, n_h]) 
    bh = np.zeros([1, n_h],  np.float32) 
    bv = np.zeros([1, n_v],  np.float32)
    return [W,bh,bv]



def sample(proba):
    #sample a vector of 0's and 1's with respect to a prescribed probability array
    return np.floor(proba + np.random.uniform(0, 1,np.shape(proba)))

def gibbs_step(xg, model, sampling = True, activations = [sigmoid,sigmoid]):
    #perform one Gibbs step
    [W,bh,bv] = model
    hg = sample(activations[0](np.matmul(xg, W) + bh))
    if sample == True:
        xg = sample(activations[1](np.matmul(hg, np.transpose(W)) + bv))
    else:
        xg = (activations[1](np.matmul(hg, np.transpose(W)) + bv))
    return xg


def gibbs_sample(xg,model,k, activations = [sigmoid,sigmoid], sampling = True):
    #perform a Gibbs sample
    x_sample = xg
    for step in range(k):
        if step < k-1:
            x_sample = gibbs_step(x_sample, model, activations = activations, sampling = True)
        else:
            x_sample = gibbs_step(x_sample, model, activations = activations, sampling = sampling)
    return x_sample

def train_step(model,lr, x, CD = 1, activations = [sigmoid,sigmoid], sampling = True):
    
    [W,bh,bv] = model
    x_sample = gibbs_sample(x,model,CD, activations = activations, sampling = sampling) 
    h = sample(activations[0](np.matmul(x, W) + bh)) 
    h_sample = sample(activations[1](np.matmul(x_sample, W) + bh)) 
    
    size_bt = np.shape(X)[0]
    Delta_W  = lr/size_bt* (np.matmul(np.transpose(x), h) - np.matmul(np.transpose(x_sample), h_sample))
    Delta_bv = lr/size_bt* np.sum(x- x_sample, 0)
    Delta_bh = lr/size_bt*np.sum(h-h_sample, 0)
    
    W += Delta_W
    bv += Delta_bv
    bh += Delta_bh
    return [W,bh,bv]

def vis_to_hid(X,model, activations = [sigmoid,sigmoid]):
    [W,bh,bv] = model
    h = (activations[0](np.matmul(X, W) + bh)) 
    return h

def hid_to_vis(H,model, activations = [sigmoid,sigmoid]):
    [W,bh,bv] = model
    xg = (activations[1](np.matmul(H, np.transpose(W)) + bv)) 
    return xg


#Part II: the phony data
if phony_data_part == True:
    n_v = 2
    n_h = 1
    
    #we will create data along a plane in R^dim_v
    
    plane = np.array([[1,0]])
    
    data_size = 500
    X = np.matmul( np.random.normal(5,2,(data_size,1)) , plane)
    
    model = init_model(n_v,n_h)
    
    for i in range(20):
        model1 = train_step(model, 0.2, X)
        print('model at step {}'.format(i),model)
    
    

#Part III: producing eights
    

path = r''
file = 'mnist_small.csv'

data = pd.read_csv(path + '\\' + file, header = None)
data.columns = ['number'] + list(range(28*28))
eights_df = data[data['number'] == 8][list(range(28*28))]
eights_np = eights_df.values
numbers_np = data[list(range(28*28))].values
#
#for i in range(10):
#    img = eights_np[i].reshape((28,28))/255
#    plt.imshow(img, cmap="Greys")
#    plt.show()

    
if train_multiple_models == True:
    '''grid search for an optimal number of hidden neurons'''
    number_models = 4
    
    n_v = 28*28
    n_h = [50*(i+2) for i in range(number_models)]
    activations = [[sigmoid,sigmoid] for i in range(number_models)]
    lr = [0.1 for i in range(number_models)]
    
    models = [init_model(n_v,n_h[i])  for i in range(number_models)]
    X = numbers_np/255
    
    
    last_loss = [28*28 for i in range(number_models)]
    CD = 1
    for i in range(1000):
        
        if i% 20 == 0:
            CD = int(i/200) +1
            print('synthetic data at step {}'.format(i))
            plt.figure(figsize=(5*(number_models+1),5))
            n_ex = np.random.randint(len(X))
            eight_example = X[n_ex]
            plt.subplot(1,number_models+1,1)
            plt.imshow(eight_example.reshape(28,28), cmap = "Greys")
            plt.title("original figure")
            for m in range(len(models)):
                model = models[m]
                xg = X.copy()
            
                [W,bh,bv] = model
#                hg = sample(activations[0](np.matmul(xg, W) + bh)) #Propagate the visible values to sample the hidden values
#                xs = sample(activations[1](np.matmul(hg, np.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
                xs = gibbs_sample(xg, model, CD, activations = activations[m], sampling = False)
                loss = np.mean(np.abs(xg - xs))
                #print('loss : ',loss)
                if last_loss[m] < loss:
                    lr[m] = lr[m]/2
                    print('reduced learning rate of model {} to {}'.format(m,lr[m]))
                    if lr[m] < 0.000001 and loss > 0.1:
                        models[m] = init_model(n_v,n_h[m])
                        lr[m] = 0.1
                        print('retossing model {}'.format(m))
                last_loss[m] = loss
                #hidden = np.random.uniform(0,1,(1,n_h))
                x_syn = (gibbs_step(eight_example, model, sampling = False, activations = activations[m])).reshape((28,28))
                plt.subplot(1,number_models+1,m+2)
                plt.imshow(x_syn, cmap="Greys")
                plt.title('reconstruction {} neurons \n loss = {}'.format(n_h[m],round(loss,4)))
            plt.show()
        for m in range(len(models)):
            model = models[m]
            model = train_step(model, lr[m], X, CD = CD, activations = activations[m], sampling = False)
    
    
    
    '''Generate synthetic data from another character'''
    
    
    
    n_gen = 50
    
    plt.figure(figsize=(8,4*n_gen))
    for i in range(n_gen):
        xo = numbers_np[np.random.randint(len(numbers_np))]
        plt.subplot(n_gen,2,2*i+1)
        plt.imshow(xo.reshape(28,28), cmap = "Greys")
        plt.title("original figure")
        x_syn = (gibbs_step(xo, model, sampling = False)).reshape((28,28))
        plt.subplot(n_gen,2,2*i+2)
        plt.imshow(x_syn, cmap="Greys")
        plt.title('reconstruction')
    plt.show()
    
    CDmax = 50
    plt.figure(figsize=(10,5*CDmax))
    for i in range(CDmax):
        xo = np.zeros(28*28)
        #xo = np.random.uniform(1,0.1,28*28)
        plt.subplot(CDmax,2,2*i+1)
        plt.imshow(xo.reshape(28,28), cmap = "Greys")
        plt.title("original figure")
        x_syn = (gibbs_sample(xo, model, i, sampling = False)).reshape((28,28))
        plt.subplot(CDmax,2,2*i+2)
        plt.imshow(x_syn, cmap="Greys")
        plt.title('reconstruction after CD {}'.format(i))
    plt.show()
    
    



''' Adaptive CD-n and DBM '''

DBMShape = [28*28,200,50]

#training the first layer RBM

X = numbers_np/255

n_v = DBMShape[0]
n_h = DBMShape[1]
activations = [sigmoid, sigmoid]
lr = 0.1
    
model = init_model(n_v,n_h)
    
    
last_loss = 0.5
CD = 1
for i in range(2000):
        if i% 10 == 0:
            print('synthetic data at step {}'.format(i))
            plt.figure(figsize=(10,5))
            n_ex = np.random.randint(len(X))
            eight_example = X[n_ex]
            plt.subplot(1,2,1)
            plt.imshow(eight_example.reshape(28,28), cmap = "Greys")
            plt.title("original figure")
            
            xg = X[0:1000]
            [W,bh,bv] = model
            hg = sample(activations[0](np.matmul(xg, W) + bh)) 
            xs = sample(activations[1](np.matmul(hg, np.transpose(W)) + bv))
                
            loss = np.mean(np.abs(xg - xs))
            if i % 50 == 0:
                if last_loss < loss:
                    lr = lr/2
                    print('reduced learning rate of model to {}'.format(lr))
                    if lr < 0.01:
                        CD += 1
                        lr *= 4
                        print('passing to CD {}'.format(CD))
                last_loss = loss
            
            x_syn = (gibbs_step(eight_example, model, sampling = False, activations = activations)).reshape((28,28))
            plt.subplot(1,2,2)
            plt.imshow(x_syn, cmap="Greys")
            plt.title('reconstruction {} neurons \n loss = {}'.format(n_h,round(loss,4)))
        plt.show()
        
            
        model = train_step(model, lr, X, CD = CD, activations = activations, sampling = False)
[W,bh,bv] = model
Model = [[W.copy(),bh.copy(),bv.copy()]]


#training a second RBM to play the role of the second layer

X1 = vis_to_hid(X,Model[0])

n_v = DBMShape[1]
n_h = DBMShape[2]
activations = [sigmoid,sigmoid]
lr = 0.1
    
model = init_model(n_v,n_h)
    
    
last_loss = 0.5
CD = 1
for i in range(1000):
        if i% 100 == 0:
            print('synthetic data at step {}'.format(i))
            plt.figure(figsize=(10,5))
            n_ex = np.random.randint(len(X))
            eight_example = X[n_ex]
            plt.subplot(1,3,1)
            plt.imshow(eight_example.reshape(28,28), cmap = "Greys")
            plt.title("original figure")
            hid_example = vis_to_hid(eight_example, Model[0])
            plt.subplot(1,3,2)
            plt.imshow(hid_to_vis(hid_example, Model[0]).reshape(28,28), cmap = "Greys")
            plt.title("original (reconstructed) figure")
            
            xg = X1[0:974]
            [W,bh,bv] = model
            hg = sample(activations[0](np.matmul(xg, W) + bh)) 
            xs = sample(activations[1](np.matmul(hg, np.transpose(W)) + bv))
                
            loss = np.mean(np.abs(xg - xs))
            
            if last_loss < loss:
                lr = lr/2
                print('reduced learning rate of model to {}'.format(lr))
                if lr < 0.0001:
                    CD += 1
                    lr *= 4
                    print('passing to CD {}'.format(CD))
            last_loss = loss
            
            x_syn = hid_to_vis((gibbs_step(hid_example, model, sampling = False, activations = activations)), Model[0]).reshape((28,28))
            plt.subplot(1,3,3)
            plt.imshow(x_syn, cmap="Greys")
            plt.title('reconstruction {} neurons \n loss = {}'.format(n_h,round(loss,4)))
        plt.show()
        
            
        model = train_step(model, lr, X1, CD = CD, activations = activations, sampling = False)

Model.append([W.copy(),bh.copy(),bv.copy()])






n_gen = 10

plt.figure(figsize=(8,4*n_gen))
for i in range(n_gen):
    xo = numbers_np[np.random.randint(len(numbers_np))]
    plt.subplot(n_gen,2,2*i+1)
    plt.imshow(xo.reshape(28,28), cmap = "Greys")
    plt.title("original figure")
    h1 = vis_to_hid(xo, Model[0])
    h2 = vis_to_hid(h1,Model[1])
    v2 = hid_to_vis(h2,Model[1])
    xg = hid_to_vis(v2,Model[0])
    plt.subplot(n_gen,2,2*i+2)
    plt.imshow(xg.reshape(28,28), cmap="Greys")
    plt.title('reconstruction')
plt.show()




n_gen = 10

plt.figure(figsize=(8,4*n_gen))
xo = numbers_np[np.random.randint(len(numbers_np))]
for i in range(n_gen):
    plt.subplot(n_gen,2,2*i+1)
    plt.imshow(xo.reshape(28,28), cmap = "Greys")
    plt.title("original figure")
    xg = gibbs_step(v2,Model[0],)
    plt.subplot(n_gen,2,2*i+2)
    plt.imshow(xg.reshape(28,28), cmap="Greys")
    plt.title('reconstruction')
plt.show()


