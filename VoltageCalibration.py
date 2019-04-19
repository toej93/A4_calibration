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


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

   
def FileReader(MyFile,graphname,length,block_type):
    gr1 = MyFile.Get(graphname)
    
    t_buff = gr1.GetX()
    v_buff = gr1.GetY()
    n = gr1.GetN()

    t_buff.SetSize(n)
    v_buff.SetSize(n)

    v = np.array(v_buff,copy=True)
    t = np.array(t_buff,copy=True)
    #print(t[-1])
    cut = len(v)-length
    
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

def SineFit(t,v,freq):
    params, params_covariance = optimize.curve_fit(SineFunc,t,v,p0=[freq,np.pi/2.0,600.0])#,bounds=([0.216,-np.inf,200],[0.220,np.inf,np.inf]))#freq,offset,amplitude,voff
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
    plt.show()

def SinePlotter(t,v,params,sample,col): 
    plt.figure(10,facecolor='w')
    plt.plot(t,v[sample,:],color=col)
    #print(t)
    #print(v[sample,:])
    #print(params[0])
    t_up = np.linspace(t[0],t[-1],5000)
    #plt.plot(t_up,SineFunc(t_up,params[sample,0],params[sample,1],params[sample,2]),color=col,lw=1.0)
    plt.xlabel('Time (ns)')
    plt.ylabel('ADC Counts')
    #plt.show()
"""
def load_tcal(channel):
    t_cal = np.zeros(896)
    minv = 0
    for i in range(0,896/128):
        t_cal[minv:minv+128]=np.load('cal_files/t_cal_1402_'+channel+'.npy')+minv*40/128
        minv=minv+128
    print('tcal is', t_cal)
    return(t_cal)
"""  
def CorrectVoltage(files, channel,freq):

    num_blocks = 6000
    total_samples = 896#896
    
    ADC = np.zeros([num_blocks,total_samples])
    time = np.zeros(total_samples)
   
    odds = np.linspace(1,total_samples-1,total_samples/2,dtype=int)
    evens = np.linspace(0,total_samples-2,total_samples/2,dtype=int)

    MyFile = ROOT.TFile.Open('data/processed/calibration_data_fulltest_elChan'+channel+'_run'+files+'.root')
    block_nums = np.loadtxt("data/processed/block_data_elChan"+channel+"_run"+files+'.txt')

    pedestals =np.load('best_pedestals/ch_'+channel+'_ped.npy')
    odd_params=np.zeros([num_blocks,3])
    bad_params=np.zeros([num_blocks,3])
    
    #load calibrated time and volts:
    t_cal = np.load('cal_files2/t_cal_'+files+'_'+channel+'.npy')

    ADC_list = [[] for i in range(32768)]#each entry = block_number*64 +sample%64
    volts_list = [[] for i in range(32768)]
    colors =["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    plt.figure(0,facecolor='w')
    
    for i in range(0,num_blocks):
        time[evens],ADC[i,evens]=FileReader(MyFile,'gr_E_'+str(i),total_samples/2,block_nums[i])
        time[odds],ADC[i,odds]=FileReader(MyFile,'gr_O_'+str(i),total_samples/2,block_nums[i])
        if(i==0):
            plt.plot(time,ADC[i,:],color=colors[0],label='Raw Data')
        ADC[i,:]=PedestalFix(ADC[i,:],channel,block_nums[i],pedestals)
        if(i==0):
            plt.plot(time,ADC[i,:],color=colors[1],label='Pedestal Corrected')
        odd_params[i,:] = SineFit(t_cal[odds],ADC[i,odds],freq)
        bad_params[i,:] = SineFit(time[odds],ADC[i,odds],freq)

    plt.legend()
    plt.show()
    #SinePlotter(time,ADC,bad_params,5,colors[0])
    #SinePlotter(t_cal,ADC,odd_params,5,colors[1])
    #plt.show()
    spacing_e = t_cal[evens[1:]]-t_cal[evens[:-1]]
    spacing_o = t_cal[odds[1:]]-t_cal[odds[:-1]]
    binwidth = 0.02
    plt.figure(0,facecolor='w')
    plt.hist(spacing_o,bins=np.arange(min(spacing_o), max(spacing_o) + binwidth, binwidth),edgecolor='none',color=colors[0],label='Odd Spacing')
    plt.hist(spacing_e,bins=np.arange(min(spacing_e), max(spacing_e) + binwidth, binwidth),edgecolor='none',color=colors[1],label='Even Spacing',alpha=0.7)
    plt.axvline(x=0.625,color='black')
    plt.legend()
    plt.show()

    v_amp = 445.0
    
        
    for i in range(0,num_blocks):
        my_block = int((block_nums[i]+1)%512)
                       
        for j in range(0,total_samples):#896
            #slope = partial_derivative(SineFunc,var=0,point=[t_cal[j],odd_params[i,0],odd_params[i,1],v_amp])
            #if(np.abs(slope)<0.45*v_amp):
            if(ADC[i,j]*SineFunc(t_cal[j],odd_params[i,0],odd_params[i,1],v_amp)>0.0):
                ADC_list[my_block*64+j%64].append(ADC[i,j])
                volts_list[my_block*64+j%64].append(SineFunc(t_cal[j],odd_params[i,0],odd_params[i,1],v_amp))
            #print('slope is', slope)

    tot = 32768
    p_pos=np.zeros([32768,4])
    p_neg=np.zeros([32768,4])
    plot_blocks = []
    for i in range(0,32768):
        this_ADC = np.asarray(ADC_list[i])
        this_volt = np.asarray(volts_list[i])

        p_pos[i,:] = np.polyfit(this_ADC[this_ADC>0],this_volt[this_volt>0],3)
        p_neg[i,:] = np.polyfit(this_ADC[this_ADC<0],this_volt[this_volt<0],3)
        plot_blocks.append((i/64))
        #values = np.linspace(0,32768,32768)
        """
        x = np.sort(this_ADC[this_ADC>0])
        x=np.append(x,0)
        x = np.sort(x)
        xn = np.sort(this_ADC[this_ADC<0])
        xn=np.append(xn,0)
        plt.figure(1,facecolor='w')
        plt.scatter(this_ADC,this_volt,s=2,color='black')
        plt.plot(x,np.multiply(p_pos[i,0],x**3)+np.multiply(p_pos[i,1],x**2)+np.multiply(p_pos[i,2],x)+p_pos[i,3],color=colors[0],lw=2.0)
        plt.plot(xn,np.multiply(p_neg[i,0],xn**3)+np.multiply(p_neg[i,1],xn**2)+np.multiply(p_neg[i,2],xn)+p_neg[i,3],color=colors[1],lw=2.0)
        plt.xlabel('ADC Counts')
        plt.ylabel('Voltage (mV)')
        plt.show()
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

    #plt.subplot(2,1,1)
    #plt.hist(ADC_list[15][:],bins = 50)
    #plt.subplot(2,1,2)
    #plt.hist(volts_list[15][:],bins = 50)
    
    
    plt.show()
    

def main():
    
    channel = str(sys.argv[1])#'0'
    freqs = [0.218,0.353,0.521,0.702]


    N1 = [0,3,8,11,16,19,24,27]
    N2 = [1,2,9,10,17,18,25,26]
    
    if(int(channel) in N1):
        rootfiles = ['1402', '1403','1404','1405']
    if(int(channel) in N2):
        rootfiles = ['1411','1412','1413','1414']

    for a in range(0,4):
        CorrectVoltage(rootfiles[a],channel,freqs[a])
    

   
if __name__=="__main__":
   main()
