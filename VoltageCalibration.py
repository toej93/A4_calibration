from __future__ import print_function
from ROOT import TCanvas, TGraph
from ROOT import gROOT
import ROOT
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import iirfilter,lfilter,butter
from scipy import signal
from scipy.fftpack import fft
from scipy import optimize
from scipy.misc import derivative
import numpy as np
import sys
import math
from math import sin
from array import array
from pynverse import inversefunc
from TimingCalibration import partial_derivative,PedestalFix
import matplotlib
#from sympy import *
from scipy.optimize import leastsq
from AutomaticLoadData import LoadDataFromWeb
from termcolor import colored



font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [12, 8]

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

def SineFit3(t,data):

    guess_mean = 0#np.mean(data)
    #guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
    guess_phase = 0
    #guess_freq = 1
    #guess_amp = 1

    # we'll use this to plot our first estimate. This might already be good enough for you
    #data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: 445.0*np.sin(2.0*np.pi*0.218*t+x[0]) + x[1] - data
    est_phase, est_mean = leastsq(optimize_func, [guess_phase, guess_mean])[0]
    print(guess_mean,est_mean)

    """
    plt.figure(0)
    plt.scatter(t,data)
    t_long = np.linspace(0,300,1000)
    plt.plot(t_long,445.0*np.sin(2.0*np.pi*0.218*t_long+est_phase)+est_mean)
    plt.show()
    """

    return(0.218,est_phase,445.0,est_mean)


def SineFunc3(t,k,phi,A): #time, freq, offset, amplitude
    return 445.0*np.sin(2.0*np.pi*k*t+phi)

def FileReader(MyFile,graphname,length,block_type):
    gr1 = MyFile.Get(graphname)
    #print('updated!')
    t_buff = gr1.GetX()
    v_buff = gr1.GetY()
    n = gr1.GetN()

    t_buff.SetSize(n)
    v_buff.SetSize(n)

    v = np.array(v_buff,copy=True)
    t = np.array(t_buff,copy=True)
    #print(t[-1])
    cut = len(v)-length #default here is length of 2 blocks

    #print('cut is', cut)
    if((block_type+1)%2==0):#if true starting block is even, after cutting out first block
        v=v[cut/2:(len(v)-cut/2)]
        t=t[:(len(t)-cut)]
    else:
        v=v[cut:]
        t=t[:(len(t)-cut)]
    #print(len(v),len(t))
    return(t,v)

def sort_vals(t,v):
    args = t.argsort()
    t=t[args]
    v=v[args]
    return(t,v)



def SineFunc(t,k,phi,A): #time, freq, offset, amplitude
    return A*np.sin(2.0*np.pi*k*t-phi)


def SineFit(t,v,freq,A):

    # test = lambda t,k,phi: SineFunc(t,k,phi,A)
    #print(test(3,5,6))

    params, params_covariance = optimize.curve_fit(lambda t,k,phi: SineFunc(t,k,phi,A),t,v,p0=[freq,np.pi/2.0])#,bounds=([0.216,-np.inf,200],[0.220,np.inf,np.inf]))#freq,offset,amplitude,voff


    #if(params[2]<0):
        #params[2]=np.abs(params[2])
    #   params[1]=params[1]+np.pi
    params[1]=params[1]%(np.pi*2)
    while(params[1]<0):
        params[1]=params[1]+np.pi*2.0
    params = np.append(params,[A])
    #print(params)
    return(params)

def SineFunc2(t,k,phi,A,o): #time, freq, offset, amplitude
    return A*np.sin(2.0*np.pi*k*t-phi)+o

def SineFit2(t,v,freq):
    params, params_covariance = optimize.curve_fit(SineFunc2,t,v,p0=[freq,np.pi/2.0,500.0,30])#,bounds=([0.216,-np.inf,200],[0.220,np.inf,np.inf]))#freq,offset,amplitude,voff
    if(params[2]<0):
        params[2]=np.abs(params[2])
        params[1]=params[1]+np.pi
    params[1]=params[1]%(np.pi*2)
    while(params[1]<0):
        params[1]=params[1]+np.pi*2.0
    return(params)

def histogram(vals,string):
    fig, axs = plt.subplots(2,1,facecolor='w')
    counter=0
    for ax in axs.reshape(-1):
        #print('success')
        ax.hist(vals[counter],color='navy',edgecolor='none',bins=50)
        ax.axvline(x=np.mean(vals[counter]),color='red',ls='-',linewidth=2.0)
        #ax.text(230,250,"mean (MHz):  "+str(round(np.mean(freq_array[counter]*1000),2)))
        #ax.set_xlim(200,250)
        #ax.set_ylim(0,300)
        ax.set_xlabel(string)
        ax.set_ylabel('Counts')
        counter = counter +1
    # plt.show()

def SinePlotter(t,v,params,sample,col):
    plt.figure(10,facecolor='w')
    plt.scatter(t,v[sample,:],color=col)
    #print(t)
    #print(v[sample,:])
    #print(params[0])
    t_up = np.linspace(t[0],t[-1],5000)
    plt.plot(t_up,SineFunc(t_up,params[sample,0],params[sample,1],params[sample,2]),color=col,lw=1.9)
    plt.xlabel('Time (ns)')
    plt.ylabel('ADC Counts')
    # plt.show()

def Cubic(a,p0,p1,p2,p3):
    return(p0*a**3+p1*a**2+p2*a+p3)

def plot_voltage(t,adc,p_pos,p_neg,block_num,freq,A):


    start_block = int(block_num)
    #print('starting block is',start_block)
    length = len(adc)
    #print(len(adc))
    piece = np.linspace(start_block*64,(start_block*64+length)%32768,length,dtype=int)
    v = np.zeros(length)

    for k in range(0,length):

        if(k%64==0 and k>0):
            #print('here!')
            start_block=(start_block+1)%512
        #print(start_block,start_block*64+i%64)
        if(adc[k]>0):
            v[k]=Cubic(adc[k],p_pos[start_block*64+k%64,0],p_pos[start_block*64+k%64,1],p_pos[start_block*64+k%64,2],p_pos[start_block*64+k%64,3])
        if(adc[k]<0.0):
            v[k]=Cubic(adc[k],p_neg[start_block*64+k%64,0],p_neg[start_block*64+k%64,1],p_neg[start_block*64+k%64,2],p_neg[start_block*64+k%64,3])


    params = SineFit(t,v,freq,A)

    plt.figure(0,facecolor='w')
    plt.scatter(t[1::2],adc[1::2],color='dodgerblue')
    #plt.scatter(t[1::2],v[1::2],color='maroon')
    plt.scatter(t[1::2],v[1::2],color='maroon')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    #plt.plot(t,v,color='maroon')
    t_up = np.linspace(0,280.0,2000)
    plt.plot(t_up,SineFunc(t_up,params[0],params[1],params[2]),color='maroon')
    #print(params[0],params[1],params[2])
    # plt.show()

def plot_cubic(this_ADC,this_volt,mean_ADC,mean_volt,p_pos,p_neg,i,meanval):
    colors =["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    deg = len(p_pos[i,:])-1


    mx = symbols('mx')

    print('meanval is ', meanval)
    x = np.sort(this_ADC[this_ADC>=-meanval])
    #x=np.append(x,x_int[0])
    x = np.sort(x)
    xn = np.sort(this_ADC[this_ADC<=-meanval])
    #xn=np.append(xn,x_int[0])

    deg = len(p_pos[i,:])
    print('degree is',deg)
    p_p = np.zeros(len(x))
    p_n = np.zeros(len(xn))
    for p in range(0,deg):
        # print(p_pos[i,deg-p-1])
        print(deg-p)
        p_p = p_p+p_pos[i,deg-p-1]*x**p
        p_n = p_n+p_neg[i,deg-p-1]*xn**p

    print(len(x))
    print(len(xn))

    plt.figure(1,facecolor='w')
    plt.scatter(this_ADC,this_volt,s=2,color='black')
    #plt.scatter(mean_ADC,mean_volt,s=10,color='red')
    plt.plot(x,p_p,color=colors[0],lw=2.0)
    plt.plot(xn,p_n,color=colors[1],lw=2.0)
    plt.xlim([-600,600])
    #plt.plot(x,np.multiply(p_pos[i,0],x**3)+np.multiply(p_pos[i,1],x**2)+np.multiply(p_pos[i,2],x)+p_pos[i,3],color=colors[0],lw=2.0)
    #plt.plot(xn,np.multiply(p_neg[i,0],xn**3)+np.multiply(p_neg[i,1],xn**2)+np.multiply(p_neg[i,2],xn)+p_neg[i,3],color=colors[1],lw=2.0)
    plt.xlabel('ADC Counts')
    plt.ylabel('Voltage (mV)')
    # plt.show()


def BlockCorrector(block_nums):
    for i in range(0,len(block_nums)):
        if(block_nums[i]%2==0):
            block_nums[i]=(block_nums[i]+2)%512
        else:
            block_nums[i]=(block_nums[i]+1)%512
    return(block_nums)

def MeanEachBlock(t,v):
    new_v = np.zeros(len(v))
    for k in range(0,len(new_v)/64):
        #print(v[k*64:k*64+64])
        #print(np.mean(v[k*64:k*64+64]))
        new_v[k*64:k*64+64] = v[k*64:k*64+64]-np.mean(v[k*64:k*64+64])
    #plt.figure(1)
    #plt.plot(t,v)
    #plt.plot(t,new_v)
    #plt.show()
    return(new_v)

def CorrectVoltage(station,files, channel,freq):

    ADC_list = [[] for i in range(32768)]#each entry = block_number*64 +sample%64
    volts_list = [[] for i in range(32768)]
    colors =["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    print('Loading RootFiles')
    avg_ADC = []

    A = 445.0 #This is the amplitude for the fitting

    total_samples = 896
    #We now load the root files and extract variables of interest.
    all_times, ADC,block_nums = LoadDataFromWeb(station,files,"0529","2018",int(channel),total_samples,0,1,1,0,1)
    times, ADC_raw,block_nums = LoadDataFromWeb(station,files,"0529","2018",int(channel),total_samples,0,1,0,0,1)

    total_events = len(all_times[:,0])#Number of events
    print('number of events:',total_events)

    total_samples = len(all_times[0,:]) #Number of samples
    odds = np.linspace(1,total_samples-1,total_samples/2,dtype=int) #Define array for odd samples
    evens = np.linspace(0,total_samples-2,total_samples/2,dtype=int) #Define array for even samples

    odd_params=np.zeros([total_events,3]) #Put zeroes in the arrays
    bad_params=np.zeros([total_events,3])
    plt.figure(0)
    # print(times[0,1]-times[0,0])
    plt.plot(all_times[0,:],ADC[0,:])
    plt.plot(times[0,1::2]-times[0,0],ADC_raw[0,1::2])
    plt.show()
    for i in range(0,total_events):#Loop over all waveforms
        if(i%100==0):
            print(i)

        odd_params[i,:] = SineFit(all_times[i,odds],ADC[i,odds],freq,A) #Fit sine to waves
        #print(odd_params[i,:])
        times[i,odds]=times[i,odds]-times[i,0]#Shift to zero
        #bad_params[i,:] = SineFit(times[i,odds],ADC[i,odds],freq,A)

    t_cal = all_times[0]
    #block_nums = BlockCorrector(block_nums)#FROM THIS POINT ON, Block_nums has CORRECT first block!
    #plot_voltage(t_cal,ADC[0,:],p_pos,p_neg,block_nums[0])


    #plt.legend()
    #plt.show()
    print(block_nums)
    print(np.shape(times))
    print(np.shape(ADC))




    #SinePlotter(times[5,1::2],ADC_raw[:,1::2],bad_params,5,colors[0])
    #plt.show()
    #SinePlotter(all_times[5,1::2],ADC[:,1::2],odd_params,5,colors[1])
    #plt.show()
    """
    spacing_e = t_cal[evens[1:]]-t_cal[evens[:-1]]
    spacing_o = t_cal[odds[1:]]-t_cal[odds[:-1]]
    binwidth = 0.02


    plt.figure(0,facecolor='w')
    plt.hist(spacing_o,bins=np.arange(min(spacing_e), max(spacing_o) + binwidth, binwidth),edgecolor='none',color=colors[0],label='Odd Spacing')
    plt.hist(spacing_e,bins=np.arange(min(spacing_e), max(spacing_o) + binwidth, binwidth),edgecolor='none',color=colors[1],label='Even Spacing',alpha=0.7)
    plt.axvline(x=0.625,color='black',lw=3.0)
    #plt.xlim([0.25,1.0])
    plt.ylim([0,150])
    plt.xlabel('Spacing Between Samples (ns)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

    SinePlotter(time,ADC,bad_params,5,colors[0])
    SinePlotter(t_cal,ADC,odd_params,5,colors[1])
    plt.show()
    """


    plot_block = []
    plot_adc = []



    for i in range(62,total_events): #Use only events after event 62 (why?)
        my_block=int(block_nums[i])
        for j in range(0,total_samples):#896
            if(j%64==0 and j>0): #Each block has 64 samples
                #print(j,my_block)
                my_block=(my_block+1)%512

            volt_val = SineFunc(t_cal[j],odd_params[i,0],odd_params[i,1],A) #Retrieve sine function of amplitude A, with parameters from fit
            #print(volt_val,ADC[i,j])
            #print(volt_val)
            #print(my_block*64+j%64)
            #slope = partial_derivative(SineFunc,var=0,point=[t_cal[j],odd_params[i,0],odd_params[i,1],v_amp])
            #slope = v_amp*2*np.pi*odd_params[i,0]*np.cos(2*np.pi*odd_params[i,0]*t_cal[j]-odd_params[i,1])
            if( (ADC[i,j]*volt_val>0.0 or (np.abs(ADC[i,j])<100 and np.abs(volt_val)<100)) ):#If volt_val and ADC have the same sign and magnitudes are both less than 100
                #Now, the voltages are grouped in blocks per event
                ADC_list[(my_block*64+j%64)].append(ADC[i,j])
                volts_list[(my_block*64+j%64)].append(volt_val)
                plot_block.append(my_block*64+j%64)
                plot_adc.append(ADC[i,j])

    tot = 32768
    counts = 0
    degree = 3
    p_pos=np.zeros([32768,degree+1])
    p_neg=np.zeros([32768,degree+1])
    plot_blocks = []
    chi2_p = np.zeros([32768])
    chi2_n = np.zeros([32768])

    zero_vals = np.zeros([32768])

    extra_peds = []

    C1 = [0,1,8,9,16,17,24,25] #Vpols
    C2 = [3,2,11,10,19,18,27,26] #Hpols

    if(float(channel) in C1):
        max_range = 32768
    else:
        max_range = 16384 #Using only odd samples fro Hpols


    for i in range(0,max_range):
        if(float(channel) in C1):
            i = i
        else:
            i=i*2+1 #Only odd samples

        if(i%1000==1):
            print(i)

        this_ADC = np.asarray(ADC_list[i])
        #print('average ADC is', np.mean(this_ADC))
        ADC_mean = np.mean(this_ADC)
        this_volt = np.asarray(volts_list[i])


        lin_ADC = []
        lin_volt = []
        for j in range(0,len(this_ADC)):
            if (np.abs(this_ADC[j]<100) and np.abs(this_volt[j]<100)):
                lin_ADC.append(this_ADC[j])
                lin_volt.append(this_volt[j])

        # print(colored(lin_ADC, 'red'))

        myslope,intercept,r,p,stderr = stats.linregress(np.asarray(lin_ADC),np.asarray(lin_volt))
        zero_vals[i] = intercept/myslope

        if(channel in C2):
            p_pos[i-1,:]=p_pos[i,:]
            p_neg[i-1,:]=p_neg[i,:]

        meas_volt = []
        mean_ADC = []
        var_volt = []
        tmin = np.min(this_ADC)
        tmax = tmin+5.0

        sorted = np.argsort(this_ADC)
        sort_ADC = this_ADC[sorted]
        sort_volt = this_volt[sorted]

        #print('')
        #print(this_ADC)
        my_spacing = 5
        for k in range(0,int(len(this_ADC)/my_spacing)):
            # print("k is %i"%k)
            meas_volt.append(np.mean(sort_volt[k*my_spacing:k*my_spacing+my_spacing]))
            mean_ADC.append(np.mean(sort_ADC[k*my_spacing:k*my_spacing+my_spacing]))
            var_volt.append(np.var(sort_volt[k*my_spacing:k*my_spacing+my_spacing]))
            #var_volt.append(1)
        #print(meas_volt,mean_ADC,var_volt)
        #meas_volt = sort_volt
        #mean_ADC = sort_ADC
        #var_volt = np.linspace(1,1,len(mean_ADC))

        """
        while(tmax<np.max(this_ADC)):

            meas_volt.append(np.mean(this_volt[(this_ADC>tmin) & (this_ADC<tmax)]))
            mean_ADC.append(np.mean(this_ADC[(this_ADC>tmin) & (this_ADC<tmax)]))
            var_volt.append(np.var(this_volt[(this_ADC>tmin) & (this_ADC<tmax)]))

            tmin=tmin+5
            tmax=tmax+5
        """

        mean_ADC = np.asarray(mean_ADC)
        meas_volt = np.asarray(meas_volt)
        var_volt = np.asarray(var_volt)

        #print('')
        #print(var_volt)
        #print(mean_ADC)
        #print(meas_volt)

        mean_ADC=np.append(mean_ADC,np.zeros(100))
        #this_ADC=np.append(this_ADC,np.zeros(100)+np.max(this_ADC))
        meas_volt=np.append(meas_volt,np.zeros(100)+intercept)

        #print('mean volts is', np.mean(this_volt))
        #plt.figure()
        #plt.scatter(mean_ADC,meas_volt)
        #plt.show()
        # print(np.count_nonzero(mean_ADC[mean_ADC>=0]))
        # if(np.count_nonzero(mean_ADC[mean_ADC>=0])==0):
        #     continue
        try:
            if(np.count_nonzero(mean_ADC[mean_ADC>=0])==0 or np.count_nonzero(mean_ADC[mean_ADC<=0])==0 or
            np.count_nonzero(meas_volt[mean_ADC>=0])==0 or np.count_nonzero(meas_volt[mean_ADC<=0])==0):
                print(colored(np.count_nonzero(mean_ADC[mean_ADC>=0]), 'red'))
                p_pos[i,:] = p_pos[i-2,:]
                p_neg[i,:] = p_neg[i-2,:]
            else:
            # print(meas_volt[mean_ADC>=0])
                p_pos[i,:] = np.polyfit(mean_ADC[mean_ADC>=0],meas_volt[mean_ADC>=0],degree)
                p_neg[i,:] = np.polyfit(mean_ADC[mean_ADC<=0],meas_volt[mean_ADC<=0],degree)
        except ValueError:
            print("Failed!!!")
            p_pos[i,:] = p_pos[i-2,:]
            p_neg[i,:] = p_neg[i-2,:]

        mean_ADC = mean_ADC[:-100]
        meas_volt = meas_volt[:-100]

        mean_ADC = mean_ADC[~np.isnan(var_volt)]
        meas_volt = meas_volt[~np.isnan(var_volt)]
        var_volt= var_volt[~np.isnan(var_volt)]

        mean_ADC = mean_ADC[np.nonzero(var_volt)]
        meas_volt= meas_volt[np.nonzero(var_volt)]
        var_volt = var_volt[np.nonzero(var_volt)]

        pred_v = Cubic(mean_ADC[mean_ADC<=0],p_neg[i,0],p_neg[i,1],p_neg[i,2],p_neg[i,3])
        chi2_n[i]=np.sum(np.divide((meas_volt[mean_ADC<=0]-pred_v)**2,var_volt[mean_ADC<=0]))/(len(var_volt[mean_ADC<=0]))
        pred_v = Cubic(mean_ADC[mean_ADC>=0],p_pos[i,0],p_pos[i,1],p_pos[i,2],p_pos[i,3])
        chi2_p[i]=np.sum(np.divide((meas_volt[mean_ADC>=0]-pred_v)**2,var_volt[mean_ADC>=0]))/(len(var_volt[mean_ADC>=0]))
        #print(chi2_p[i])
        #plot_cubic(mean_ADC,meas_volt,p_pos,p_neg,i,0.0)

        """
        this_ADC = this_ADC[:-100]
        this_volt = this_volt[:-100]
        meas_volt = []
        mean_ADC = []
        var_volt = []
        tmin = np.min(this_volt)
        tmax = tmin+30.0
        while(tmax<np.max(this_volt)):
            #good_indices = np.where((this_ADC>tmin) & (this_ADC<tmax))
            if(len(this_volt[(this_volt>tmin) & (this_volt<tmax)])>1):
                meas_volt.append(np.mean(this_volt[(this_volt>tmin) & (this_volt<tmax)]))
                mean_ADC.append(np.mean(this_ADC[(this_volt>tmin) & (this_volt<tmax)]))
                var_volt.append(np.var(this_volt[(this_volt>tmin) & (this_volt<tmax)]))
                if(var_volt[-1]<1):
                    print('variance is ',var_volt[-1])
                    var_volt[-1]=30.0
            tmin=tmin+30
            tmax=tmax+30

        meas_volt = np.asarray(meas_volt)
        var_volt = np.asarray(var_volt)
        mean_ADC = np.asarray(mean_ADC)
        v_expn = Cubic(this_ADC[this_ADC<=0],p_neg[i,0],p_neg[i,1],p_neg[i,2],p_neg[i,3])
        v_expp = Cubic(this_ADC[this_ADC>=0],p_pos[i,0],p_pos[i,1],p_pos[i,2],p_pos[i,3])
        mean_voltp = Cubic(mean_ADC[meas_volt>=0],p_pos[i,0],p_pos[i,1],p_pos[i,2],p_pos[i,3])
        mean_voltn = Cubic(mean_ADC[meas_volt<=0],p_neg[i,0],p_neg[i,1],p_neg[i,2],p_neg[i,3])

        #print('volts : ',meas_volt,mean_voltp,mean_voltn)
        #p_poly = np.polynomial.polynomial.Polynomial.fit(this_ADC,this_volt)
        #coef = p_poly.convert().coef
        #p_pos[i,:]=[coef[-1],coef[-2],coef[-3],coef[-4]]
        #p_neg[i,:]=p_pos[i,:]
        #p_neg[i,:] = np.polynomial.polynomial.Polynomial.fit(this_ADC,this_volt,3)
        #if zero_val<-100.0:
        #    plot_cubic(this_ADC,this_volt,p_pos,p_neg,i,0.0)


        #print(len(meas_volt[meas_volt>=0]))
        #print(len(mean_voltp))
        chi2_p[i]=np.sum(((meas_volt[meas_volt>=0]-mean_voltp)**2)/var_volt[meas_volt>=0])/float(len(meas_volt[meas_volt>=0]))
        chi2_n[i]=np.sum(((meas_volt[meas_volt<=0]-mean_voltn)**2)/var_volt[meas_volt<=0])/float(len(meas_volt[meas_volt<=0]))
        """
        #hi2_n[i]=np.sum(((this_ADC[this_ADC<=0]-v_expn)**2)/v_expn)/len(this_ADC)
        """
        print(chi2_p[i],chi2_n[i])
        if(chi2_p[i]>2.0):
            #plot_cubic(this_ADC,this_volt,p_pos,p_neg,i)
            p_pos[i,:]=p_pos[i-1,:]
        if(chi2_n[i]>2.0):
            #plot_cubic(this_ADC,this_volt,p_pos,p_neg,i)
            p_neg[i,:]=p_neg[i-1,:]
        """
        #print(var_volt)
        #print(mean_ADC)
        #print(meas_volt)
        #print('chi2 is', chi2_p[i])
        #plot_cubic(this_ADC,this_volt,p_pos,p_neg,i,0.0)


        #if(i%2==0):
        #plot_cubic(this_ADC,this_volt,p_pos,p_neg,i,0.0)
        #plot_cubic(mean_ADC,meas_volt,p_pos,p_neg,i,0.0)
        """
        if(i==570):
            print(i,chi2_p[i])
            plot_cubic(this_ADC,this_volt,mean_ADC,meas_volt,p_pos,p_neg,i,0.0)
        """
        #if(i>500 and chi2_p[i]>5.0 and chi2_p[i-2]<1.0 and i%2==1):
        #if(i==909):
        #    print(i,chi2_p[i])
        #    plot_cubic(this_ADC,this_volt,mean_ADC,meas_volt,p_pos,p_neg,i,0.0)
        """
        if((i%898==0 and i>0)):
            print(i)
            print('chi2 is ', chi2_p[i])
            plot_chip = chi2_p[:i]
            plot_chin = chi2_n[:i]
            plt.figure(0)
            plt.hist(plot_chip[plot_chip<100],bins=100,alpha=0.5)
            plt.hist(plot_chin[plot_chin<100],bins=100,alpha=0.5)
            plt.show()

            plot_cubic(this_ADC,this_volt,mean_ADC,meas_volt,p_pos,p_neg,i,0.0)

            my_ind = np.where(block_nums==counts)
            print('this index is', my_ind)
            if(my_ind[0][0]!=None):
                plot_voltage(t_cal,ADC[my_ind[0][0],:],p_pos,p_neg,block_nums[my_ind[0][0]],freq,A)
            counts = counts +14

        """
        plot_blocks.append((i/64))

        #values = np.linspace(0,32768,32768)


    binwidth = 0.1
    #plt.figure(1,facecolor='w')
    #plt.hist(chi2_p[1::2],bins=np.arange(min(chi2_p[1::2]), max(chi2_p[1::2]) + binwidth, binwidth),alpha=0.5,label='Positive Fit')
    #plt.hist(chi2_n[1::2],bins=np.arange(min(chi2_n[1::2]), max(chi2_n[1::2]) + binwidth, binwidth),alpha=0.5,label='Negative Fit')
    #plt.xlabel('Chi^2 Values')
    #plt.xlim([0,5])
    #plt.legend()
    #plt.show()

    np.save('/users/PCON0003/cond0068/ARA/ARA_Calibration/ARA'+str(station)+'_cal_files/p_pos_'+channel+'.npy',p_pos)
    np.save('/users/PCON0003/cond0068/ARA/ARA_Calibration/ARA'+str(station)+'_cal_files/p_neg_'+channel+'.npy',p_neg)
    np.save('/users/PCON0003/cond0068/ARA/ARA_Calibration/ARA'+str(station)+'_cal_files/chi2_pos_'+channel+'.npy',chi2_p)
    np.save('/users/PCON0003/cond0068/ARA/ARA_Calibration/ARA'+str(station)+'_cal_files/chi2_neg_'+channel+'.npy',chi2_n)
    # np.save('/users/PCON0003/cond0068/ARA/ARA_Calibration/ARA'+str(station)+'_cal_files/zerovals_'+channel+'.npy',zero_vals)
    #p_pos = np.load('p_pos_'+channel+'.npy')
    #p_neg = np.load('p_neg_'+channel+'.npy')

    """
    plt.figure(0,facecolor='w')
    plt.subplot(2,2,1)
    plt.hist2d(plot_blocks,p_pos[:,0],bins=(512,50),cmap=plt.cm.jet)
    plt.title('Cubic Coefficient')
    plt.subplot(2,2,2)
    plt.hist2d(plot_blocks,p_pos[:,1],bins=(512,50),cmap=plt.cm.jet)
    plt.title('Quadratic Coefficient')
    plt.subplot(2,2,3)
    plt.hist2d(plot_blocks,p_pos[:,2],bins=(512,50),cmap=plt.cm.jet)
    plt.title('Linear Coefficient')
    plt.subplot(2,2,4)
    plt.hist2d(plot_blocks,p_pos[:,3],bins=(512,50),cmap=plt.cm.jet)
    plt.title('Offset')
    """

    #plt.subplot(2,1,1)
    #plt.hist(ADC_list[15][:],bins = 50)
    #plt.subplot(2,1,2)
    #plt.hist(volts_list[15][:],bins = 50)


    #plt.show()


def main():
    #All the channels here are under the electric chain mapping
    channel = str(sys.argv[1])#'0'
    station = str(sys.argv[2])
    freqs = [0.218,0.353,0.521,0.702]


    N1 = [0,3,8,11,16,19,24,27]
    N2 = [1,2,9,10,17,18,25,26]

    if(int(channel) in N1):
        rootfiles = ['1402', '1403','1404','1405']
    if(int(channel) in N2):
        rootfiles = ['1411','1412','1413','1414']


    N1 = [0,3,8,11,16,19,24,27]
    N2 = [1,2,9,10,17,18,25,26]
    N_special = [9,16,24,25]
    rootfiles=[]
    if(station=='5'):
        if(int(channel) in N1):
            rootfile='1402'
            rootfiles = ['1402', '1403','1404','1405']
        if(int(channel) in N2):
            rootfile='1411'
            rootfiles = ['1411','1412','1413','1414']
    if(station=='4'):
        if(int(channel)==1 or int(channel)== 2):
            rootfiles = ['2840','2841','2842','2843']
        if(int(channel)==0 or int(channel)== 3):
            rootfiles = ['2829', '2830','2831','2832']


    for a in range(0,4):
        CorrectVoltage(station,rootfiles[a],channel,freqs[a])



if __name__=="__main__":
   main()
