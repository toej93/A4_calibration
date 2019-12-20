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
from AutomaticLoadData import LoadDataFromWeb
from termcolor import colored


def reject_outliers(data, m=2):
    """
    reject data points that are outside 2 standard deviations
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def update_tcal(tcal,wrap,total_samples):

    tcal_all = tcal
    for i in range(1,total_samples/128):
        tcal_all = np.append(tcal_all,tcal+wrap+tcal_all[-1])
    return(tcal_all)

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

def invertedFit(params,tval,vval):
    """
    Inverts the fit (from given parameters), and returns the times used for
    the fit.
    """
    k = params[0]
    phi = params[1] #phase
    A = params[2] #amplitude
    T = 1/(k) #period
    #print(T,v_off,phi,k,A)


    sine = (lambda t: A*np.sin(2*np.pi*k*t-phi)) #define sin as a function of time.

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

    try:
        t_close = inversefunc(sine,y_values=vval,domain=[t_min,t_max])
    except ValueError:
        print(t_max,t_min,vval,tval)
        print(params)

    return(t_close-tval)#used to be t_close-tval

def sort_vals(t,v):#Sort values in increasing order
    args = t.argsort()
    t=t[args]
    v=v[args]
    return(t,v)

def SineFunc(t,k,phi,A): #time, freq, offset, amplitude
    """
    returns the value of a sine function of the form: Asin(2pi*kt-phi)
    """
    return A*np.sin(2*np.pi*k*t-phi)

def SineFit(t,v,freq):
    """
    Given times, voltages and frequencies, this function returns the parameters
    used to fit a sine function
    """
    params, params_covariance = optimize.curve_fit(SineFunc,t,v,p0=[freq,np.pi/2.0,350])#,bounds=([-np.inf,-np.inf,200],[np.inf,np.inf,np.inf]))#freq,offset,amplitude,voff
    if(params[2]<0):#wrap values if they're negative
        params[2]=np.abs(params[2])
        params[1]=params[1]+np.pi
    params[1]=params[1]%(np.pi*2)
    while(params[1]<0):
        params[1]=params[1]+np.pi*2.0
    return(params)

def HistPlotter2D(sample,jitter):
    """
    Returns a 2D histogram with sample number on the x-axis
    and jitter offset on the y-axis
    """
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


    # plt.show()
    return()

def AddOffsets(t,v,freq,odds,old_mean_o2e,old_mean_e2o,chan):
    """
    Odd and even blocks run through different delay lines. This function
    calculates the shift between the two delay lines and applies the offset.
    The shift is calculated by fitting sine functions to odd and even blocks
    and finding a factor between the phases.

    This function returns the shift-corrected times.
    """
    print(len(v[0,:]))

    e_freq = [] #even freq
    o_freq = [] #odd freq
    for j in range(0,len(v[:,0])):#using the first sample of each event.
    #np.shape(v)=(numEvents,numSamples=896)
        val=0
        for i in range(0,14):#14*64=896 samples. Each block has 64 samples.
            params = SineFit(t[odds[:32]],v[j,odds[val:val+32]],freq)
            if i%2==0:#even block
                e_freq.append(params[0])
            else:#odd
                o_freq.append(params[0])#append the fit result params[0]=frequency
            val = val+32
    #correction factor for the sampling speed (mean_fittedfreq/real_freq)
    t_scaled_e1 = t[:64]*np.mean(e_freq)/freq #even
    t_scaled_o = t[64:128]*np.mean(o_freq)/freq #odd
    t_scaled_e2 = t[128:192]*np.mean(e_freq)/freq #even

    e2o_diff = []
    o2e_diff = []

    #Now, let's repeat the same process but with the scaled timings.
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
                diff = e_offset-o_offset
                while(diff>np.pi):
                    diff=diff-2*np.pi
                while(diff<-1*np.pi):
                    diff=diff+2*np.pi
                #print(e_offset,o_offset)
                e2o_diff.append(((diff))/(2*np.pi*freq))
            else:
                #print(oe_time[:32])
                #print(len(oe_time[:32]))
                params = SineFit(t_scaled_o[odds[:32]],v[j,odds[val:val+32]],freq)
                o_offset=(params[1])
                params = SineFit(t_scaled_e2[odds[:32]],v[j,odds[val+32:val+64]],freq)
                e_offset=(params[1])
                diff = e_offset-o_offset
                while(diff>np.pi):
                    diff=diff-2*np.pi
                while(diff<-1*np.pi):
                    diff=diff+2*np.pi
                o2e_diff.append(((diff))/(2*np.pi*freq))
            val = val+32

    o2e_diff = reject_outliers(np.asarray(o2e_diff))
    e2o_diff = reject_outliers(np.asarray(e2o_diff))
    histogram([e2o_diff,o2e_diff],'Wrap Around Time (ns)',chan)
    #Scale time so that offset is taken into account
    e2o_mean = np.abs(np.mean(e2o_diff)+old_mean_e2o)
    o2e_mean = np.abs(np.mean(o2e_diff)+old_mean_o2e)
    if(e2o_mean>1.0):
        e2o_mean=0.31
    if(o2e_mean>1.0):
        o2e_mean=0.31
    print('the offsets are: %0.3f, %0.3f'%(e2o_mean, o2e_mean))

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
    return(t_updated,o2e_mean,e2o_mean)

    #print(t)
def histogram(vals,string,chan):
    fig, axs = plt.subplots(2,1,facecolor='w')
    counter=0
    for ax in axs.reshape(-1):
        print('success')
        ax.hist(vals[counter],color='navy',edgecolor='none',bins=200)
        ax.axvline(x=np.mean(vals[counter]),color='red',ls='-',linewidth=2.0)
        #ax.text(230,250,"mean (MHz):  "+str(round(np.mean(freq_array[counter]*1000),2)))
        #ax.set_xlim(200,250)
        #ax.set_ylim(0,300)
        ax.set_xlabel(string)
        ax.set_ylabel('Counts')
        ax.set_title('Wrap around time, ch %i, mean = %0.2f'%(int(chan),np.mean(vals[counter])))
        counter = counter +1
    plt.tight_layout()
    plt.savefig("./debugPlots/offset_hist_ch%i.png"%int(chan), dpi=200)

def SinePlotter(t,v,params,sample):
    plt.figure(10)
    plt.scatter(t,v[sample,:])
    #print(t)
    #print(v[sample,:])
    t_up = np.linspace(t[0],t[-1],100)
    plt.plot(t_up,SineFunc(t_up,params[sample,0],params[sample,1],params[sample,2]))
    # plt.show()

def SlopeFinder(t,v,sample):
    if sample==0:
        slope = (v[sample+2]-v[sample])/(t[sample+2]-t[sample])
    else:
        slope=(v[sample]-v[sample-2])/(t[sample]-t[sample-2])

    return slope

def CorrectTimingSample(rootfile,channel,freq,t_cal,station):
    wf_len = 896

    #load in data
    all_times, volt,blocks = LoadDataFromWeb(station,rootfile,"0118","2018",int(channel),wf_len,0,1,0,0,1)
    time = all_times[0]-all_times[0][0]
    print('number of events is', np.shape(volt)[0])

    #define all variables
    num_blocks=len(volt[:,0])#num blocks entries = num events

    best_params = np.zeros([num_blocks,4]) #creates a num_blocksx4 array
    odds = np.linspace(1,wf_len-1,wf_len/2,dtype=int) #a list containing only odd numbers
    evens = np.linspace(0,wf_len-2,wf_len/2,dtype=int)

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

    spacing = 0.625#This is the spacing, in ns, between odd/even samples

    if(t_cal[5]==0.0):
        print('clearing out old t_cal')
        t_cal=time[:128]
        print(t_cal)

    t_cal_full = time
    #print('t_cal before is', t_cal_full)

    odd_mean = 0.0
    even_mean = 0.0
    for l in range(0,1): #Perform the fit 5 times, so it converges to the solution
        print('loop number ', l)

        #First fix offsets between blocks, as that can be larger than the channel to channel fixes.
        if l==0:
            t_cal_full,odd_mean,even_mean=AddOffsets(t_cal_full,volt,freq,odds,odd_mean,even_mean, channel)
        #Fit each waveform to a sine wave

        #plt.figure(0)
        #plt.plot(t_cal_full[odds],volt[0,odds])
        #plt.show()

        for i in range(0,num_blocks):
            odd_params[i,:]=SineFit(t_cal_full[odds],volt[i,odds],freq)

        #histogram([odd_params[:,0],odd_params[:,0]],'')
        freq_no_outliers = reject_outliers(np.asarray(odd_params[:,0]))
        mean_freq = np.mean(freq_no_outliers)
        print('mean frequency is', mean_freq)
        #histogram([odd_params[:,1],even_params[:,1]],'')

        #Scale timing to reflect true frequency
        t_cal_full=t_cal_full*mean_freq/freq

        #Re-fit using new time
        for i in range(0,num_blocks):
            odd_params[i,:]=SineFit(t_cal_full[odds],volt[i,odds],freq)


        t_cal = t_cal_full[:128]

        jitter_array = []
        sample_array = []
        slope_array = []
        jitter_slope = []
        new_spacing = np.zeros(128) #spacing between 0 and 1, 1 and 2, etc.
        #Go through each sample and correct based on fit
        for k in range(0,896):
            counter = 0
            for i in range(0,num_blocks):
                if(np.abs(volt[i,k])<30.0 and (freq-odd_params[i,0])<0.002):# and np.abs(odd_params[i,2]>200)):


                    try:
                        invert_fit = invertedFit(odd_params[i,:],t_cal_full[k],volt[i,k])
                        jitter_array.append(invert_fit)
                        sample_array.append(k%128)
                        counter = counter+1
                    except:
                        print('error in finding inverse!')

            t_cal_full[k]=t_cal_full[k]+np.mean(jitter_array[-counter:])
            # print(colored('Sample is %i'%k, 'red'))
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
            np.save('ARA'+str(station)+'_cal_files/samples_'+str(rootfile)+'_'+str(channel)+'first.npy',np.asarray(sample_array))
            np.save('ARA'+str(station)+'_cal_files/jitter_'+str(rootfile)+'_'+str(channel)+'first.npy',np.asarray(jitter_array))

        # HistPlotter2D(sample_array,jitter_array)
    print('final t_cal is', t_cal_full)
    np.save('ARA'+str(station)+'_cal_files/t_cal_'+str(rootfile)+'_'+str(channel)+'.npy',t_cal_full)
    np.save('ARA'+str(station)+'_cal_files/samples_'+str(rootfile)+'_'+str(channel)+'final.npy',np.asarray(sample_array))
    np.save('ARA'+str(station)+'_cal_files/jitter_'+str(rootfile)+'_'+str(channel)+'final.npy',np.asarray(jitter_array))
    #HistPlotter2D(sample_array,jitter_array)
    #print('t_cal is', t_cal)
    return(t_cal_full)

def main():
    #All the channels here are under the electric chain mapping
    #rootfile='1402'
    channel = str(sys.argv[1])#'0'
    station = str(sys.argv[2])


    #channels in first file
    N1 = [0,3,8,11,16,19,24,27]
    #channels in second file
    N2 = [1,2,9,10,17,18,25,26]
    N_special = [9,16,24,25]
    print("Channel is: %i"%int(channel))
    # if(station=='5'):
    #     if(int(channel) in N1):
    #         rootfile='1402'
    #         rootfiles = ['1402', '1403','1404','1405']
    #     if(int(channel) in N2):
    #         rootfile='1411'
    #         rootfiles = ['1411','1412','1413','1414']
    rootfiles=0
    if(station=='4'):
        if(int(channel)==1 or int(channel)== 2):
            rootfiles = ['2840','2841','2842','2843']
        if(int(channel)==0 or int(channel)== 3):
            rootfiles = ['2829', '2830','2831','2832']
    #     if(int(channel) in N1 and int(channel) not in N_special):
    #         rootfile='2829'
    #         rootfiles = ['2829', '2830','2831','2832']
    #     if(int(channel) in N2 and int(channel) not in N_special):
    #         rootfile='2840'
    #         rootfiles = ['2840','2841','2842','2843']
    #     if(int(channel)in N_special):
    #         rootfiles = ['2855','2856']
    # #rootfiles = ['1402']


    freqs = [0.218,0.353,0.521,0.702]

    cal_t= np.zeros(128)

    for a in range(0,4):
        CorrectTimingSample(rootfiles[a],channel,freqs[a],cal_t,station)


if __name__=="__main__":
   main()
