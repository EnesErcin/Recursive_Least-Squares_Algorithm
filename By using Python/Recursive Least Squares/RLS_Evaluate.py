import numpy as np
import matplotlib.pylab as plt
import math
import pandas as pd
import seaborn as sns
sns.set()
from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27
import scipy.io
import RLS
f = scipy.io.loadmat('C:\\Users\\Enes\\Desktop\\RLS\\RLS_DATA.mat')
delay = 300
vars = f.keys()


# Extract Data
data_size = f.get("RX_I").size
RX_Q_data = f.get("RX_Q")
RX_I_data = f.get("RX_I")
TX_Q_data = f.get("TX_Q")
TX_I_data = f.get("TX_I")


# Take input from users to choose parameters
    # Parameters are :
        # Prediction (Regressşon exponent)
        # Forgetting Factor (close to one)
        # Test Size

mem_len = int(input("Memory Dimension ? || suggestion: <3"))
est_per= int(input("How many data ? || suggestion <1600"))
lam = float(input("Forgetting Factor ? || suggestion >0.90"))
deg = int(input("Regressor exponent ? Enter 1 -> 1 "))
# Example est_per = 1600 deree_len = 4 regs = [1,2,3,4] mem_len = 2 ,lam = 0.99-0.90
prediction = []

for i in range(deg):
    reg = int(input("Suggested inputs ?"))
    prediction.append(reg)
num_vars = deg*mem_len
i = int(input("start_point || suggestion: <5"))
LS_2 = RLS.RLS_Filter(num_vars,lam,1)
test_size = len(TX_Q_data[i*est_per:(i+1)*(est_per)])


pred_e, pred_output,_,weight = RLS.one_batch(test_size = test_size, 
deg = deg, mem_len = mem_len ,prediction = prediction,LS_2= LS_2 ,
input_data=TX_Q_data[i*est_per:(i+1)*(est_per)],
reference_data=RX_Q_data[delay+i*est_per:(i+1)*(est_per)+delay])




ref = RX_Q_data[delay+i*est_per:(i+1)*(est_per)+delay]
line_refernce = np.reshape(np.linspace(0,test_size,test_size),(test_size,1)) 
out = np.reshape((np.squeeze(pred_output)[0:test_size]),(test_size,1))
plt.plot(line_refernce,out[0:test_size],marker = '*',label='Adapted Output')       
plt.plot(line_refernce,ref[0:test_size],marker = '_',label='Curropted Signal') 
plt.legend()
plt.show()



high_end = 2500
low_end =  1000
condition_1 = np.logical_or(np.greater(out-ref, high_end ), np.less(out-ref, -high_end )) 
condition_2 = np.logical_and(np.less(out-ref, low_end), np.greater(out-ref, -low_end)) 

dif = (out[0:test_size])-(ref[0:test_size])
range_dif = (np.max(ref)-np.min(ref))

print("\n---------------------------------")
print("---------------------------------\n")

print(np.max(ref),"<    Range   <",np.min(ref))
print("Maximum range of the reference",(np.max(ref))-(np.min(ref)))
print("The count of the difference which are larger than |{}| --> ||".format(high_end), np.count_nonzero(np.extract(out-ref,condition_1)),   "||  Ratio '%' :   ",int(100*np.count_nonzero(np.extract(out-ref,condition_1))/len(out-ref)), "//This range is  %:", 100*high_end/range_dif)
print("The count of the difference which are less   than |{}|  --> ||".format(low_end), np.count_nonzero(np.extract(out-ref,condition_2)),   "||  Ratio '%' :   ",int(100*np.count_nonzero(np.extract(out-ref,condition_2))/len(out-ref)), "//This range is %:", 100*low_end/range_dif)
print("Sample Size", len(out-ref))

print("\n---------------------------------")
print("---------------------------------")

rms_error = (np.sum(dif**2)/(test_size-delay+1))**0.5
print("\n*****RMS Error --> ", rms_error)
print("*****Normalized RMS Error  %-->",100*rms_error/range_dif)

print("\n---------------------------------")
print("---------------------------------")

counts, bins = np.histogram(np.abs(dif),8)
plt.stairs(counts/test_size*100, bins)
plt.xlabel("Error Magnitude")
plt.ylabel("Percantage of samples")
plt.title("||   The Error Distribution   || (with avarage output value of   {})".format(range_dif))
plt.show()

line_refernce = np.linspace(0,len(dif),len(dif)) 
plt.plot(line_refernce,dif,marker="+",label='Error') 
line_refernce = np.linspace(0,test_size,test_size) 
plt.plot(line_refernce,RX_Q_data[delay+i*est_per:(i+1)*(est_per)+delay],label='Curropted Signal') 
plt.plot(line_refernce,np.linspace(np.average(np.abs(dif)/2),np.average(np.abs(dif)/2),test_size),marker="_",label="abs dif mean(+)")
plt.plot(line_refernce,np.linspace(-np.average(np.abs(dif)/2),-np.average(np.abs(dif)/2),test_size),marker="_",label="abs dif mean(-)")
plt.legend()
print("Avarege difference / Max Range of data  ==> %{}".format(100*np.average(np.abs(dif))/range_dif))
print("Avarage difference between adapted input and reference data ==> {}".format(np.average(np.abs(dif))))


dif = (out)-(ref[0:test_size])
rms_error = (np.sum(dif**2)/(test_size-delay+1))**0.5
print("RMS Error --> ", rms_error)
print(np.max(ref),"<    Range   <",np.min(ref))
print("Normalized RMS Error  -->",100*rms_error/(np.max(ref)-np.min(ref)))

plt.plot(line_refernce,dif,marker="*")
plt.show()

weigth_array = np.squeeze(np.array(weight).T)
weigth_array = np.reshape(weigth_array  ,(np.shape(weigth_array)[0],np.shape(weigth_array)[1]))


weigth_array = np.squeeze(np.array(weight).T)
weigth_array = np.reshape(weigth_array  ,(np.shape(weigth_array)[0],np.shape(weigth_array)[1]))


fig = plt.figure()
gs = fig.add_gridspec(num_vars,hspace = 0)
axs = gs.subplots(sharex=True)
fig.suptitle("|| Weigths change || Memory -- {} Deg -- {} ".format(mem_len,deg))

for i in range(np.shape(weigth_array)[0]):
    axs[i].plot(line_refernce[50:],weigth_array[i][50:],"C{}".format(i),label="Weigth Apdate {}".format(i))
    axs[i].legend(loc="upper right")
plt.show()

