import torch
import time
import numpy as np

def comput_error(b,w,points):
    totalloss = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalloss += (y-(w*x+b))**2
    return totalloss/float(len(points))

def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N)*(y-((w_current*x)+b_current))
        w_gradient += -(2/N)*x*((w_current*x)+b_current)
    new_b = b_current - (learningRate*b_gradient)
    new_w = w_current - (learningRate*w_gradient)
    return [new_b,new_w]

def run_gradient(points,starting_b,starting_w,learningRate,num_iterate):
    b = starting_b
    w = starting_w
    for i in range(num_iterate):
        b, w =step_gradient(b,w,np.array(points),learningRate)
        print("new_b = ",b)
        print("new_w = ",w)
    return [b,w]

def data_generate(number,max,min):
    x = np.random.randint(max,min,size=number)
    y = np.random.randint(max,min,size=number)
    x = np.reshape(x,(len(x),-1))
    y = np.reshape(y,(len(y),-1))
    print(x.shape )
    points = np.concatenate((x,y),axis=1)
    return points

if __name__=="__main__":
    points = data_generate(100,-10,10)
    learning_rate = 0.001
    init_b = 0
    init_w = 0
    num_itirate = 1000
    print("starting training b={0} , m={1},error={2}".format(init_b,init_w,comput_error(init_b,init_w,points)))
    print("Running....")
    [b,w ]=run_gradient(points,init_b,init_w,learning_rate,num_itirate)
    print("starting training b={0} , m={1},error={2}".format(b,w,comput_error(b,w,points)))


