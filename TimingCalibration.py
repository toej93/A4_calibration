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

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

def CountNumFiles(directory,channel,files):
    num_data=0
    for i in range(0,len(files)):
        exists = os.path.isfile(directory+'t_cal_'+files[i]+'_'+channel+'.npy')
        if exists:
            num_data=num_data+1
    return(num_data)

   
def FileReader2(MyFile,graphname,samples):
    gr1 = MyFile.Get(graphname)
    
    t_buff = gr1.GetX()
    v_buff = gr1.GetY()
    n = gr1.GetN()

    t_buff.SetSize(n)
    v_buff.SetSize(n)

    v = np.array(v_buff,copy=True)
    t = np.array(t_buff,copy=True)
    num = len(v)-samples
    v=v[num:]
    t=t[num:]
    #print(v[-1])
    return(t,v)


def PedestalFix2(v,channel):
    ped_vals = np.load('pedestal_vals/ARA5_ch'+str(channel)+'_pedestal.npy')
    num_samples = len(v)
    for b in range(0,num_samples):
        v[b]=v[b]-ped_vals[b%128]
    return(v)

def update_tcal(tcal,wrap,total_samples):
    
    tcal_all = tcal
    for i in range(1,total_samples/128):
        tcal_all = np.append(tcal_all,tcal+wrap+tcal_all[-1])
    return(tcal_all)


def FindWrapAround(rootfile,channel,freq):
   
    file_name = ['data/processed/calibration_data_fulltest_elChan'+channel+'_run'+rootfile+'.root']
    #file_name = ['../araroot_trunk/calibration_data_fulltest_elChan'+channel+'_run'+rootfile+'.root']
    num_blocks = 2000
    total_samples = 1024#896
    
    volt = np.zeros([num_blocks,total_samples])
    time = np.zeros(total_samples)
   
    odds = np.linspace(1,total_samples-1,total_samples/2,dtype=int)
    evens = np.linspace(0,total_samples-2,total_samples/2,dtype=int)

    MyFile = ROOT.TFile.Open(file_name[0])
    block_nums = np.loadtxt("data/processed/block_data_elChan"+channel+"_run"+rootfile+'.txt')
    #block_nums = np.loadtxt("../araroot_trunk/block_data_elChan"+channel+"_run"+rootfile+'.txt')
    all_offsets = []
    t_cal = np.load('calibrated_times/ch_'+channel+'.npy')
    for i in range(0,num_blocks):
        time[evens],volt[i,evens]=FileReader2(MyFile,'gr_E_'+str(i),total_samples/2)
        time[odds],volt[i,odds]=FileReader2(MyFile,'gr_O_'+str(i),total_samples/2)
        volt[i,:]=PedestalFix2(volt[i,:],channel)
        if block_nums[i]%2==1:
            osets  = singleWrapAround(t_cal,volt[i,:],total_samples,freq)
            all_offsets.extend(osets)
        mean_wrap = np.mean(np.abs(all_offsets))/(2.0*np.pi*freq)

    print('wrap around time is:', mean_wrap)
    t_cal = update_tcal(t_cal,mean_wrap,total_samples)
    print('new tcal is', t_cal)
    np.save('final_times/ch_'+channel+'.npy',t_cal)



def singleWrapAround(t,v,samples,freq):
    minval = 0
    maxval = 32
    size =  32
    odds = np.linspace(1,samples-1,samples/2,dtype=int)
    offsets = []
    #print(samples/(size*2))
    for i in range(0,samples/(size*4)-1):
        #print(t[odds[size:size+size]])
        params1 = SineFit(t[odds[0:size]],v[odds[minval:maxval]],freq)
        #print(odds[maxval:maxval+size])
        try:
            params2 = SineFit(t[odds[size:size+size]],v[odds[maxval:maxval+size]],freq)

        except RuntimeError:
            plt.figure(0)
            plt.plot(t[odds[size:size+size]],v[odds[maxval:maxval+size]])
            plt.show()
        this_offset = params2[1]-params1[1]
        
        while(this_offset >np.pi):
            this_offset=this_offset-2*np.pi
        while(this_offset < -1*np.pi):
            this_offset = this_offset+2*np.pi
        
        offsets.append(this_offset%(2*np.pi))
        minval = minval+size*2
        maxval = maxval+size*2

    return(offsets)

def average_tcals(directory,channel,files):
    num_data = CountNumFiles(directory,channel,files)
    t_cals = np.zeros([num_data,128])

    for i in range(0,num_data):
        t_cals[i,:] = np.load(directory+'t_cal_'+files[i]+'_'+str(channel)+'.npy')

    #find spacings
    spacings = np.zeros(127)
    new_tcal = np.zeros(128)
    for i in range(0,len(spacings)):
        #spacings[i]=np.mean(t_cals[:,i+1]-t_cals[:,i])
        new_tcal[i]=np.mean(t_cals[:,i])
    print(spacings)
    print('new t_cal is', new_tcal)
    np.save('calibrated_times/ch_'+channel+'.npy',new_tcal)


def invertedFit2(params,t_o,v):
    k = params[0]
    phi=params[1]
    A = params[2]
    
    #print('begin')
    T= 1/(k)
    #print('period is', T)
   
    t_diff = 100.0
    #print('original t is',t_o)
    for j in range(-2,11):
        t_on = (1.0/(2.0*np.pi*k))*(np.arcsin(v/A)+phi+2.0*np.pi*j)
        t_off = (1.0/(2.0*np.pi*k))*(np.pi-np.arcsin(v/A)+phi+2.0*np.pi*j)
        t_on_diff = np.abs(t_on-t_o)
        t_off_diff = np.abs(t_off-t_o)
        #print(t_on_diff,t_off_diff)
        if(t_on_diff<=t_off_diff and t_on_diff<t_diff):
            t_diff=t_on_diff
            t_best=t_on
        if(t_off_diff<t_on_diff and t_off_diff<t_diff):
            t_diff=t_off_diff
            t_best=t_off
        #print(t_best-t_o)
    return(t_best-t_o)



        
def invertedFit(params,tval,vval):
    k = params[0]
    phi = params[1]
    A = params[2]
    T = 1/(k)
    #print(T,v_off,phi,k,A)
   
   
    sine = (lambda t: A*np.sin(2*np.pi*k*t-phi))

    t_list = np.linspace(tval-T/2,tval+T/2,100)
    sine_vals = sine(t_list)
    a_max = np.argmax(sine_vals)
    a_min = np.argmin(sine_vals)

   
    t_max=t_list[a_max]
    t_min=t_list[a_min]

    
    if(t_min>t_max):
        minval=t_max
        t_max=t_min
        t_min=minval

    #while(t_min>tval):
    #    t_min=tval-0.05
    #while(t_max<tval):
    #    #   print('hello!')
    #    t_max=tval+0.025
    #    #   print(t_max)
   
    try:   
        t_close = inversefunc(sine,y_values=vval,domain=[t_min,t_max])
    except ValueError:
        print(t_max,t_min,vval,tval)
        print(params)
        #plt.figure(1)
        #plt.plot(np.linspace(0,40,150),sine(np.linspace(0,40,150)))
        #plt.show()

    #jitter= t_close-tval
    #if(jitter>
        
    return(t_close-tval)#used to be t_close-tval
   
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

def PedestalFix(v,channel,block_num,ped_vals):
    block_num = int(block_num)
    #ped_vals = np.load('pedestal_vals/ARA5_ch'+str(channel)+'_pedestal.npy')
    my_ped_vals = np.zeros(len(v))
    counter = 0
    if(block_num%2==0):
        block_num=block_num+2
    if(block_num%2==1):
        block_num=block_num+1
    for i in range(0,len(v)/64):
        #print(ped_vals[(block_num+2)%512,:])
        my_ped_vals[counter:counter+64]=ped_vals[(block_num)%512,:]
        counter = counter+64
        block_num=(block_num+1)%512
    #print('my pedestal values are:',my_ped_vals)
    
    return(v-my_ped_vals)

def SineFunc(t,k,phi,A): #time, freq, offset, amplitude
    return A*np.sin(2*np.pi*k*t-phi)

def SineFunc2(t,phi,A): #time, freq, offset, amplitude
    return A*np.sin(2*np.pi*0.353*t-phi)

def SineFit(t,v,freq):
    params, params_covariance = optimize.curve_fit(SineFunc,t,v,p0=[freq,np.pi/2.0,350])#,bounds=([-np.inf,-np.inf,200],[np.inf,np.inf,np.inf]))#freq,offset,amplitude,voff
    if(params[2]<0):
        params[2]=np.abs(params[2])
        params[1]=params[1]+np.pi
    params[1]=params[1]%(np.pi*2)
    while(params[1]<0):
        params[1]=params[1]+np.pi*2.0
    return(params)

def SineFit2(t,v,freq,amp):
    params, params_covariance = optimize.curve_fit(lambda t,phi: SineFunc(t,freq,phi,amp),t,v,p0=[np.pi/2.0])
    #print(params)
    
    while(params[0]<0):
        params[0]=params[0]+np.pi*2.0
    
    return(params)


def HistPlotter2D(sample,jitter):
    sample_even=[]
    jitter_even = []
    sample_odd=[]
    jitter_odd = []

    for j in range(0,len(sample)):
        if(sample[j]%2==0):#if even
            sample_even.append(sample[j])
            jitter_even.append(jitter[j])
        else:
            sample_odd.append(sample[j])
            jitter_odd.append(jitter[j])
         
    plt.figure(4,facecolor='w')
    plt.hist2d(sample_even,jitter_even,bins=(128,128),cmap=plt.cm.jet,range=np.array([(0.0,128.0),(-1.0,1.0)]))
    plt.title('Even Samples')
   
    plt.figure(5,facecolor='w')
    plt.hist2d(sample_odd,jitter_odd,bins=(128,128),cmap=plt.cm.jet,range=np.array([(0.0,128.0),(-1.0,1.0)]))
    plt.title('Odd Samples')

   
    plt.show()
    return()

def AddOffsets(t,v,freq,odds):
    print(len(v[0,:]))
    #time_1b = np.linspace(0.0,19.375,32)
    #time_1b = t[odds[:32]]
    #time_1b_all = np.linspace(0.0,19.6875,64)
    #time_1b_all = t[:64]
    
    #print(time_1b)
    e_freq = []
    o_freq = []
    for j in range(0,len(v[:,0])):
        val=0
        for i in range(0,14):
            params = SineFit(t[odds[:32]],v[j,odds[val:val+32]],freq)
            if i%2==0:#even block
                e_freq.append(params[0])
            else:
                o_freq.append(params[0])
            val = val+32

    #Scale even and odd blocks based on frequencies
    o_freq = reject_outliers(np.asarray(o_freq))
    e_freq = reject_outliers(np.asarray(e_freq))
    #histogram([e_freq,o_freq],'freqs')
    #even_time = time_1b_all*np.mean(e_freq)/freq
    #odd_time = time_1b_all*np.mean(o_freq)/freq
    #eo_time = np.concatenate((even_time,odd_time+even_time[-1]))
    #oe_time = np.concatenate((odd_time,even_time+odd_time[-1]))
    #Fit Again and find offset
    #print(eo_time)
    #print(oe_time)

    t_scaled_e1 = t[:64]*np.mean(e_freq)/freq
    t_scaled_o = t[64:128]*np.mean(o_freq)/freq
    t_scaled_e2 = t[128:192]*np.mean(e_freq)/freq
    
    e2o_diff = []
    o2e_diff = []
    #plt.figure(0)
    #plt.plot(even_time[odds[:32]],v[0,odds[0:32]])
    #plt.show()

    for j in range(0,len(v[:,0])):
        val=0
        for i in range(0,13):
            #print(i)
            if(i%2==0):
                params = SineFit(t_scaled_e1[odds[:32]],v[j,odds[val:val+32]],freq)
                #print(params[0])
                e_offset=(params[1])
                params = SineFit(t_scaled_o[odds[:32]],v[j,odds[val+32:val+64]],freq)
                o_offset=(params[1])
                #print(e_offset,o_offset)
                e2o_diff.append(((e_offset-o_offset)%(2*np.pi))/(2*np.pi*freq))
            else:
                #print(oe_time[:32])
                #print(len(oe_time[:32]))
                params = SineFit(t_scaled_o[odds[:32]],v[j,odds[val:val+32]],freq)
                o_offset=(params[1])
                params = SineFit(t_scaled_e2[odds[:32]],v[j,odds[val+32:val+64]],freq)
                e_offset=(params[1])
                o2e_diff.append(((e_offset-o_offset)%(2*np.pi))/(2*np.pi*freq))
            val = val+32
    #Scale time so that offset is taken into account
    e2o_mean = np.mean(e2o_diff)
    o2e_mean = np.mean(o2e_diff)
    print('the offsets are:', e2o_mean, o2e_mean)
    
    #histogram([e2o_diff,o2e_diff],'time offset')
    t_updated = np.zeros(896)
    val = 0
    for i in range(0,7):
        if i==0:
            t_updated[val:val+64]=t_scaled_e1
            t_updated[val+64:val+128]=t_scaled_o+e2o_mean
        else:
            t_updated[val:val+64]=t_scaled_e1+o2e_mean+t_updated[val-1]
            t_updated[val+64:val+128]=t_scaled_o+e2o_mean+t_updated[val-1]
        #else:
        #    t_updated[val:val+64]=even_time+(o2e_mean+t_updated[val-1])
        #t_updated[val+64:val+128]=odd_time+(e2o_mean+t_updated[val+64-1])
        val = val+128
    print(t_updated)
    return(t_updated)
    
    #print(t)
def histogram(vals,string):
    fig, axs = plt.subplots(2,1,facecolor='w')
    counter=0
    for ax in axs.reshape(-1):
        print('success')
        ax.hist(vals[counter],color='navy',edgecolor='none')
        ax.axvline(x=np.mean(vals[counter]),color='red',ls='-',linewidth=2.0)
        #ax.text(230,250,"mean (MHz):  "+str(round(np.mean(freq_array[counter]*1000),2)))
        #ax.set_xlim(200,250)
        #ax.set_ylim(0,300)
        ax.set_xlabel(string)
        ax.set_ylabel('Counts')
        counter = counter +1
    plt.show()

def SinePlotter(t,v,params,sample): 
    plt.figure(10)
    plt.scatter(t,v[sample,:])
    #print(t)
    #print(v[sample,:])
    t_up = np.linspace(t[0],t[-1],100)
    plt.plot(t_up,SineFunc(t_up,params[sample,0],params[sample,1],params[sample,2]))
    plt.show()

def SlopeFinder(t,v,sample):
    if sample==0:
        slope = (v[sample+2]-v[sample])/(t[sample+2]-t[sample])
    else:
        slope=(v[sample]-v[sample-2])/(t[sample]-t[sample-2])
    
    return slope
    
def CorrectTimingSample(rootfile,channel,freq,t_cal):
   
    file_name = ['data/processed/calibration_data_fulltest_elChan'+channel+'_run'+rootfile+'.root']
    block_list =np.loadtxt('data/processed/block_data_elChan'+channel+'_run'+rootfile+'.txt')
    num_blocks = 2000

    wf_len = 896
    
    volt = np.zeros([num_blocks,wf_len])
    time = np.zeros(wf_len)
    best_params = np.zeros([num_blocks,4])
   
    #odds = np.linspace(1,127,64,dtype=int)
    #evens = np.linspace(0,126,64,dtype=int)

    odds = np.linspace(1,wf_len-1,wf_len/2,dtype=int)
    evens = np.linspace(0,wf_len-2,wf_len/2,dtype=int)
    #print('length of odds and evens are', odds,evens)
    
    odd_params=np.zeros([num_blocks,3])
    even_params=np.zeros([num_blocks,3])
    odd_params2 = np.zeros([num_blocks,3])
   
    odd_half = np.linspace(1,63,32,dtype=int)
    even_half= np.linspace(0,62,32,dtype=int)
   
    odd_half2 = np.linspace(65,127,32,dtype=int)
   
    jitter_avg = np.zeros(128)
    t_cal = np.zeros(128)
    
    #load all 128 samples into arrays
    best_blocks = []
    best_freqs = []
    jitter_total = []
    odd_diffs = []
    line_diffs = []
   
    #print(odd_half)
    #print(odd_half2)

    print('Loading ROOT File...')
    MyFile = ROOT.TFile.Open(file_name[0])
    ped_vals = np.load('best_pedestals/ch_'+channel+'_ped.npy')
    for i in range(0,num_blocks):
        time[evens],volt[i,evens]=FileReader(MyFile,'gr_E_'+str(i),wf_len/2,block_list[i])
        time[odds],volt[i,odds]=FileReader(MyFile,'gr_O_'+str(i),wf_len/2,block_list[i])
        
        volt[i,:]=PedestalFix(volt[i,:],channel,block_list[i],ped_vals)#remove for actual data, this is just for noise data
        #plt.figure(0)
        #plt.plot(time,volt[i,:])
        #plt.show()

    
    spacing = 0.625

    if(t_cal[5]==0.0):
        print('clearing out old t_cal')
        t_cal=time[:128]
        print(t_cal)

    t_cal_full = time
    #print('t_cal before is', t_cal_full)

    
    for l in range(0,5):
        print('loop number ', l)

        if l==0:
            t_cal_full=AddOffsets(t_cal_full,volt,freq,odds)
        for i in range(0,num_blocks):
            odd_params[i,:]=SineFit(t_cal_full[odds],volt[i,odds],freq)

        #histogram([odd_params[:,0],odd_params[:,0]],'')
            
        """
        for i in range(0,num_blocks):

            odd_params[i,:]=SineFit(t_cal_full[odds],volt[i,odds],freq)
            even_params[i,:]=SineFit(t_cal_full[evens],volt[i,evens],freq)


        freq_no_outliers = reject_outliers(np.asarray(odd_params[:,0]))
        mean_freq = np.mean(freq_no_outliers)
        print('mean frequency is', mean_freq)
        #histogram([odd_params[:,1],even_params[:,1]],'')
        t_cal_full=t_cal_full*mean_freq/freq
        """
        t_cal = t_cal_full[:128]
        
        
        """
        if(l<4):
            odd_diffs = []
            print(t_cal,volt[0,:])
            for i in range(0,num_blocks):
                try:
                    odd_params[i,:]=SineFit(t_cal[odd_half],volt[i,odd_half],freq)
                except RuntimeError:
                    SinePlotter(time[odd_half],volt[:,odd_half],odd_params,i-1)
                    print('1st half error')
                    odd_params[i,:]=[0,0,0]
                try:
                    even_params[i,:]=SineFit(t_cal[odd_half2],volt[i,odd_half2],freq)
                except RuntimeError:
                    print('2nd half error')
                    even_params[i,:]=[0,0,0]
            freq_no_outliers = reject_outliers(np.asarray(odd_params[:,0]))
            mean_freq = np.mean(freq_no_outliers)
            print('mean freq in first have is', mean_freq)
            t_cal[:64]=t_cal[:64]*mean_freq/freq

            freq_no_outliers = reject_outliers(np.asarray(even_params[:,0]))
            mean_freq = np.mean(freq_no_outliers)
            print('mean freq in second half is', mean_freq)
            t_cal[64:]=t_cal[64:]*mean_freq/freq

            #histogram([odd_params[:,0],even_params[:,0]],'')
            #SinePlotter(time[odds],volt[:,odds],odd_params,5)

            for i in range(0,num_blocks):
                odd_params[i,:]=SineFit(t_cal[odd_half],volt[i,odd_half],freq)
                even_params[i,:]=SineFit(t_cal[odd_half2],volt[i,odd_half2],freq)
                #print('even freq is: ', even_params[i,0])
                odd_diff=odd_params[i,1]-even_params[i,1]
                while(odd_diff<-np.pi):
                    odd_diff=odd_diff+2*np.pi
                while(odd_diff>np.pi):
                    odd_diff=odd_diff-2*np.pi
                odd_diffs.append(odd_diff)

            #histogram([odd_diffs,odd_diffs],'Phase Difference')
            #histogram([odd_params[:,0],even_params[:,0]],'')

            block_offset = np.mean(reject_outliers(np.asarray(odd_diffs)))/(2.0*np.pi*mean_freq)
            print('original block offset is', block_offset)
            t_cal[64:]=t_cal[64:]+block_offset#-old_block 

            for i in range(0,num_blocks):
                odd_params[i,:]=SineFit(t_cal_full[odds],volt[i,odds],freq)
         
            freq_no_outliers = reject_outliers(np.asarray(odd_params[:,0]))
            mean_freq = np.mean(freq_no_outliers)   
            #fix frequency
            #SinePlotter(time[odds],volt[:,odds],odd_params,5)
            #histogram([odd_params[:,0],even_params[:,0]],'')

            print('mean frequency is:', mean_freq)
            t_cal = t_cal*mean_freq/freq

            #Fit Entire waveform to sine wave
            for i in range(0,num_blocks):
                odd_params[i,:]=SineFit(t_cal_full[odds],volt[i,odds],freq)

            print('better frequency is', np.mean(np.asarray(odd_params[:,0])))
            #histogram([odd_params[:,0],odd_params[:,0]],'')
            #SinePlotter(time[odds],volt[:,odds],odd_params,5)
        """
        jitter_array = []
        sample_array = []
        slope_array = []
        jitter_slope = []
        new_spacing = np.zeros(128) #spacing between 0 and 1, 1 and 2, etc. 
        for k in range(0,896):
            counter = 0
            for i in range(0,num_blocks):
                if(np.abs(volt[i,k])<30.0 and (freq-odd_params[i,0])<0.002):# and np.abs(odd_params[i,2]>200)):
                    
                    invert_fit = invertedFit(odd_params[i,:],t_cal_full[k],volt[i,k])
                    jitter_array.append(invert_fit)
                    sample_array.append(k%128)   
                    counter = counter+1

            t_cal_full[k]=t_cal_full[k]+np.mean(jitter_array[-counter:])
            
            if(k>0):
                new_spacing[k%128]=new_spacing[k%128]+t_cal_full[k]-t_cal_full[k-1]

        
        new_spacing[1:]=new_spacing[1:]/7.0
        new_spacing[0]=new_spacing[0]/6.0
        #print('spacing is', new_spacing)

        for i in range(0,896):
            if(i==0):
                t_cal_full[i]=0.0
            else:
                t_cal_full[i]=t_cal_full[i-1]+new_spacing[(i)%128]
                
            
        #print('final t_cal is',t_cal_full)
                
        t_cal=t_cal_full[:128]
        
                
        #SinePlotter(time[odds],volt[:,odds],odd_params,5)


        """ 
        plt.figure(6,facecolor='w')
        plt.hist2d(slope_array,jitter_slope,bins=(250,128),cmap=plt.cm.jet,range=np.array([(-1100.0,1100.0),(-1.0,1.0)]))
        plt.title('Even Samples vs Slope')
        plt.show()
        """
        
        if(l<1):
            np.save('samples_'+rootfile+'_'+channel+'first.npy',np.asarray(sample_array))
            np.save('jitter_'+rootfile+'_'+channel+'first.npy',np.asarray(jitter_array))
            
        #HistPlotter2D(sample_array,jitter_array)
    print('final t_cal is', t_cal_full)
    np.save('t_cal_'+rootfile+'_'+channel+'.npy',t_cal_full)
    np.save('samples_'+rootfile+'_'+channel+'final.npy',np.asarray(sample_array))
    np.save('jitter_'+rootfile+'_'+channel+'final.npy',np.asarray(jitter_array))
    #HistPlotter2D(sample_array,jitter_array)
    #print('t_cal is', t_cal)                                       
    return(t_cal)

def main():
    #rootfile = str(sys.argv[1])#'1403'
    channel = str(sys.argv[1])#'0'
    #freq = float(sys.argv[3])#0.353

    #rootfiles = ['1422','1421','1420','1419']
    #freqs = [0.218,0.218,0.218,0.218]

    N1 = [0,3,8,11,16,19,24,27]
    N2 = [1,2,9,10,17,18,25,26]
    
    if(int(channel) in N1):
        rootfiles = ['1402', '1403','1404','1405']
    if(int(channel) in N2):
        rootfiles = ['1411','1412','1413','1414']
    #rootfiles = ['1422']
    #rootfiles = ['1410']
    freqs = [0.218,0.353,0.521,0.702]
    
    cal_t= np.zeros(128)

    #jitter = np.load('jitter_1404_3final.npy')
    #samples = np.load('samples_1404_3final.npy')
    #HistPlotter2D(samples,jitter)

    
    for a in range(0,4):
        exists = os.path.isfile('jitter_'+rootfiles[a]+'_'+channel+'final.npy')
        if(exists):
            print('file exists!')
        else:
            CorrectTimingSample(rootfiles[a],channel,freqs[a],cal_t)
    
    #average all results together
    average_tcals('cal_files/',channel,rootfiles)

    #account for wrap around time
    #FindWrapAround(rootfiles[1],channel,freqs[1])
    
if __name__=="__main__":
   main()
