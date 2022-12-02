import numpy as np
import matplotlib.pylab as plt
import math
import pandas as pd
import seaborn as sns
sns.set()
from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27


class RLS_Filter:
    def __init__(self, num_vars, lam, delta):
        '''
        num_vars: Degree of polinomial
        lam: Forgetting factor, usually very close to 1
        delta: Initation value -> ! Bad initation needs more itteration to reach same accuracy
        '''
        self.num_vars = num_vars
        self.P = delta*np.matrix(np.identity(self.num_vars))

        # Weigths/Coefficent of the system
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1],1)

        # Kalman Gain Factor
        self.g = np.matrix(np.zeros(num_vars))
        self.g = np.reshape(self.g,(num_vars,1))
        
        # Variables needed for add_obs
        self.lam_inv = lam**(-1)
        
        # A priori error
        self.a_priori_error = 0
        
        # Count of number of observations added
        self.num_obs = 0

    def add_obs(self, x, t):
        '''
        Expected value is t, add the new observation as t.
        t is noisy output of the some linear system. Input of the RLS.

        Task is to identify a system, by determining coefficents,
        that outputs a value which is closest to t.

        x is a column vector as a numpy matrix  |   (new inputs to the system)
        t is a real scalar                      |   (expected output to update weigths)
        '''            

        self.g = self.lam_inv*self.P*x/(1+self.lam_inv*(x.T*self.P*x))
        self.P = self.P*self.lam_inv - self.g*(x.T*self.P)
        self.w = self.w + self.g*(t-x.T*self.w)

        self.a_priori_error = t - x.T*self.w
        self.num_obs += 1
        

    def get_error(self):
        '''
        Finds the (instantaneous) error.
        '''
        return self.a_priori_error

    def get_weights(self):
        '''
        Finds the (instantaneous) weigths.
        '''
        return self.w


def gen_regs(x_val,deg,mem_len,prediction):
    regs = np.zeros((mem_len,deg))

    assert np.shape(x_val)      ==  (2,)
    assert np.shape(prediction) ==  (deg,)

    for j in range (mem_len):
        for k in range (deg):
            regs[j][k] = x_val[j]**prediction[k]

    return regs


#Create and update pure signal
def update_availbe_signal(data_in,degree_dim):
    mem_dim =len(data_in)   
    regs = np.zeros((mem_dim,degree_dim))
    c = 0

    for m in range(mem_dim):
        for k in range(degree_dim):
            c = (data_in[m]**k)
            regs[m][k] = c

    return regs

def generate_input(data,mem_len,test_size):
    every_input     = np.zeros((test_size+mem_len,mem_len))
    avilable_input  = np.zeros(mem_len)

    for k in range (mem_len):
        data = np.insert(data,0,0)

    for j in range (test_size):
        for i in range (mem_len):
            avilable_input[i] = data[j+i]
        every_input[j] = avilable_input

    return every_input 



def one_batch(deg,mem_len,prediction,LS_2,test_size,input_data,reference_data):
        # Gets the result of the algorithm that has the data of {test_size} according to given predictions (prediction,mem_len,deg)
    
    """""
    test_size   -- Number of measurements availble
    prediction  -- Prediction of Regressors (Closer Prediction gives better results)
    every_pure_input -- Input measurments with memory
    LS_2        -- Main Object
    deg & mem_len   -- Prediction of degree and memory length of the system
    """""

    assert LS_2.__class__.__name__  == "RLS_Filter"                                      


    num_vars  = deg*mem_len
    pred_e = []
    pred_output = []



    for i in range(test_size):

        every_input = generate_input(input_data,mem_len,len(input_data))

        assert np.shape(every_input)    == (test_size+mem_len,mem_len)

        #regs = gen_regs(x_val=every_pure_input[i][:],deg=deg,mem_len=mem_len,prediction=prediction)
        #regs = regs.reshape(1,num_vars)
        regs = update_availbe_signal(data_in= every_input[i] ,degree_dim=4)
        regs = regs.flatten()
        regs = regs.reshape(1,num_vars)

        LS_2.add_obs(x=regs.T,t=reference_data[i])
        #Single RLS Data has been evaluated

        pred_e.append(LS_2.get_error())     
        #To observe if the error is in fact decreasing

        get_weigths = LS_2.get_weights()
        #print(get_weigths)

        pred_output.append(regs*get_weigths)

    return pred_e,pred_output