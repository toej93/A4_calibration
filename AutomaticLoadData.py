from __future__ import print_function
from ROOT import TCanvas, TGraph
from ROOT import gROOT
import ROOT
import os
import matplotlib.pyplot as plt
from scipy import signal,stats
from scipy.stats import norm
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import iirfilter,lfilter,butter,hilbert
from scipy import signal
from scipy.fftpack import fft
from scipy import optimize
from scipy.misc import derivative
from scipy.interpolate import interp1d, Akima1DInterpolator
import numpy as np
import sys
import math
from math import sin
from array import array
from pynverse import inversefunc
#from TimingCalibration import partial_derivative
#from VoltageCalibration import BlockCorrector, MeanEachBlock
import matplotlib
import scipy
import itertools
#from CalPulser import loadvalues
import warnings
import requests
from bs4 import BeautifulSoup
#import urllib2
from urllib.request import urlopen
from termcolor import colored

warnings.simplefilter(action='ignore', category=FutureWarning)

def SineFunc(t,k,phi,A): #time, freq, offset, amplitude
    return A*np.sin(2.0*np.pi*k*t-phi)


def SineFit(t,v,freq,A):

    test = lambda t,k,phi: SineFunc(t,k,phi,A)
    #print(test(3,5,6))

    params, params_covariance = optimize.curve_fit(lambda t,k,phi: SineFunc(t,k,phi,A),t,v,p0=[freq,np.pi/2.0],maxfev=1000000)#,bounds=([0.216,-np.inf,200],[0.220,np.inf,np.inf]))#freq,offset,amplitude,voff


    #if(params[2]<0):
        #params[2]=np.abs(params[2])
    #   params[1]=params[1]+np.pi
    params[1]=params[1]%(np.pi*2)
    while(params[1]<0):
        params[1]=params[1]+np.pi*2.0
    params = np.append(params,[A])
    #print(params)
    return(params)


def SinePlotter(t,v,params,sample,col):
    plt.figure(10,facecolor='w')
    plt.scatter(t,v[:],color=col)
    #print(t)
    #print(v[sample,:])
    #print(params[0])
    t_up = np.linspace(t[0],t[-1],5000)
    plt.plot(t_up,SineFunc(t_up,params[0],params[1],params[2]),color=col,lw=1.9)
    plt.xlabel('Time (ns)')
    plt.ylabel('ADC Counts')
    #plt.show()

def MeanEachBlock(v,channel):
    new_v = np.zeros(len(v))
    if(int(channel) not in [0,1,8,9,16,17,24,25]):
        size = 32
    else:
        size = 64
    for k in range(0,int(len(new_v)/size)):
        temp_v = v[k*size:k*size+size]
        temp_v[1::2]=temp_v[1::2]-np.mean(temp_v[1::2])
        temp_v[0::2]=temp_v[0::2]-np.mean(temp_v[0::2])
        new_v[k*size:k*size+size] = temp_v
        #new_v[k*size:k*size+size] = v[k*size:k*size+size]-np.mean(v[k*size:k*size+size])
    return(new_v)

def RemoveFirstBlock(time,volt,b_number):
    blocks_to_cut=1 #at least 1
    if(b_number%2==0):
        volt=volt[blocks_to_cut*64:]#1280
        time=time[blocks_to_cut*64:]
        b_number = (b_number + blocks_to_cut)%512
    else:
        volt=volt[(blocks_to_cut+0)*64:]#1280
        time=time[(blocks_to_cut+0)*64:]
        b_number = (b_number + blocks_to_cut+0)%512
    return(time,volt,b_number)

def SameLengthWaveforms(time,volt,length):
    n = len(volt)
    if(n>length):
        #print('too long')
        volt=volt[:length]
        time=time[:length]
    if(n<length):
        #print('too short')

        volt=np.append(volt,np.zeros(length-n))

        time=np.append(time,np.linspace(time[-1],time[-1]+(time[1]-time[0])*(-1+length-n),(length-n)))


    return(time,volt)


def get_url_paths(url):
    response = requests.get(url, params={})
    if response.ok:
        response_text = response.text
        #print(response_text)
    else:
        return response.raise_for_status()
        #print('not okay')
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a')]
    nums = []
    for i in range(5,len(parent)):
        current = parent[i][-5:]
        nums.append(int(current[:-1]))
    return nums

def DownloadPedFile(year,run,custom):
    print(custom)
    if(custom!=''):
        print('using custom ped value!')
        return(np.loadtxt(custom))
    pedfiles = "http://icecube:skua@convey.icecube.wisc.edu/data/exp/ARA/2013/monitoring/aware/pedestals/ARA05/"+year+"/raw_data/"
    pednumbers = get_url_paths(pedfiles)
    pednumbers = np.asarray([int(i) for i in pednumbers])
    #print(pednumbers[0]+pednumbers[1])
    #print(pednumbers,run)
    if(pednumbers.min()<int(run)):
        this_ped = pednumbers[pednumbers < int(run)].max()
    else:
        print('Warning: Ped files do not go low enough for this run')
        this_ped = pednumbers.min()
    print('using pedestal file', this_ped)

    username = 'icecube'
    password = 'skua'

    #This should be the base url you wanted to access.
    baseurl = 'http://convey.icecube.wisc.edu/data/exp/ARA/'

    #Create a password manager
    manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, baseurl, username, password)

    #Create an authentication handler using the password manager
    auth = urllib2.HTTPBasicAuthHandler(manager)

    #Create an opener that will replace the default urlopen method on further calls
    opener = urllib2.build_opener(auth)
    urllib2.install_opener(opener)

    #Here you should access the full url you wanted to open
    response = urllib2.urlopen(baseurl + "2013/monitoring/aware/pedestals/ARA05/"+year+"/raw_data/run_00"+str(this_ped)+"/pedestalValues.run00"+str(this_ped)+".dat")
    html = response.read()
    ped_numpy = np.array([[float(j) for j in i.split(' ')] for i in html.splitlines()])
    return(ped_numpy)

def fixlength(t,length):#calibrated time will always start on even block

    spacing = t[1:129]-t[:128]
    t_new = np.zeros(length)

    for b in range(1,length):
        t_new[b]=t_new[b-1]+spacing[(b-1)%128]

    return(t_new)

def PedestalFix(v,channel,block_num,ped_values): # this is for using an ARA generated ped file

    #print('ped values are', ped_values)
    my_ped_values = np.zeros(len(v))
    counter = 0
    for i in range(0,int(len(v)/64)):
        v0 =int(int(channel)/8)
        v1 =int(block_num)
        v2 =int(int(channel)%8)
        ind_test = np.where((ped_values[:,0]==v0))
        my_ped_row = np.where((ped_values[:,0]==v0) & (ped_values[:,1]==v1) &(ped_values[:,2]==v2))
        my_ped_row_vals = ped_values[my_ped_row[0],:]
        my_ped_values[counter:counter+64]=my_ped_row_vals[0,3:]
        block_num=(block_num+1)%512
        counter = counter +64
    return(v-my_ped_values)


def VoltageCorrector(v,block,p_pos,p_neg,total_samples,chi2p,chi2n):

    test =Cubic(50.0,p_pos[50])
    new_v = np.zeros(len(v))
    new_b = block

    for t in range(0,total_samples):
        if t%2==0 :
            t_odd = t+1
        else:
            t_odd = t

        best_index= int(new_b)*64+t_odd%64
        while(chi2p[best_index]>1.0):
            best_index=(best_index+2)%32768

        if(t%64==0 and t>0):
            #print('hello')
            new_b=int(new_b+1)%512
        if(v[t]>=0.0):
            val=Cubic(v[t],p_pos[best_index,:])
        if(v[t]<0.0):
            val=Cubic(v[t],p_neg[best_index,:])
        new_v[t]=val

    return(new_v)
def Cubic(a,p):

    deg = len(p)
    p_p=0.0
    for pi in range(0,deg):
        p_p = p_p+p[deg-pi-1]*a**pi

    return(p_p)


def Calibrator(station,time,volt,block_number,channel,length,ped_values,kPed,kTime,kVolt,run):
    #MyFile = ROOT.TFile.Open('data/processed/calibration_data_full_elChan'+channel+'_run'+files+'.root')
    #block_nums0 = np.loadtxt("data/processed/block_data_elChan"+channel+"_run"+files+'.txt')
    if(int(channel) not in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]):
        channel='18'

    cal_file = 'ARA'+station+'_cal_files'
    #cal_file = 'cal_files8'

    if(kPed==1):
        volt = PedestalFix(volt,channel,block_number,ped_values)
        volt = MeanEachBlock(volt,channel)

    if(kTime==1):
        #print('hello! calibrating the time')
        t_cal = np.load(cal_file+'/t_cal_'+run+'_'+channel+'.npy')
        t_cal = fixlength(t_cal,length+64)
        time,volt = SameLengthWaveforms(time,volt,length+64)
        #plt.figure()
        #plt.plot(time,volt)
        if(block_number%2==0):
            """
            if(int(channel) not in [0,1,8,9,16,17,24,25]):
                time=t_cal[:length/2]
                volt=volt[:length]
                volt=volt[1::2]
                length=length/2
            else:
            """
            time=t_cal[:length]
            volt=volt[:length]
            #plt.figure()
            #plt.plot(time,volt)
        else:
            """
            if(int(channel) not in [0,1,8,9,16,17,24,25]):

                time=t_cal[:length/2]
                volt=volt[32:length+32]
                volt=volt[1::2]
                length=length/2

            else:
            """
            time=t_cal[:length]
            volt=volt[64:length+64]
            block_number = block_number+1
            #plt.figure()
            #plt.plot(time,volt)
    else:
        time,volt = SameLengthWaveforms(time,volt,length)
    #plt.figure()
    #plt.plot(time,volt)
    #plt.show()

    if(kVolt==1):

        chi2_pos = np.load(cal_file+'/chi2_pos_'+channel+'.npy')
        chi2_neg = np.load(cal_file+'/chi2_neg_'+channel+'.npy')
        #pedestals0 = np.load('best_pedestals/ch_'+channel+'_ped.npy')
        p_pos = np.load(cal_file+'/p_pos_'+channel+'.npy')
        p_neg = np.load(cal_file+'/p_neg_'+channel+'.npy')

        volt = VoltageCorrector(volt,block_number,p_pos,p_neg,length,chi2_pos,chi2_neg)
        volt = MeanEachBlock(volt,channel)


    return(time,volt,block_number)

def RemoveBackwardsSamples(tcal,v):
    diffs = tcal[0,1:]-tcal[0,:-1]
    backwards_args = np.where(diffs<0)
    t_new = np.zeros([len(tcal[:,0]),len(tcal[0,:])-len(backwards_args[0])])
    #print(np.shape(t_new))
    v_new = np.zeros([len(tcal[:,0]),len(tcal[0,:])-len(backwards_args[0])])
    for z in range(0,len(tcal[:,0])):
        #print(tcal[z,1:])
        #print(tcal[z,:-1])
        diffs = tcal[z,1:]-tcal[z,:-1]
        #print(len(tcal),len(v[0]))
        #print(diffs)
        backwards_args = np.where(diffs<0)
        #print(backwards_args)
        #print('backwards samples are:',backwards_args[0])
        t_new[z,:] = np.delete(tcal[z,:],backwards_args[0])
        v_new[z,:] = np.delete(v[z,:],backwards_args[0])
        #v = np.delete(v,backwards_args[0],axis=1)
        #print(len(tcal),len(v[0]))
        #print(tcal,v)
    return(t_new,v_new)

def LoadDataFromWeb(station,run,date,year,channel,length,kCalib,kPed,kTime,kVolt,kForCalib):
    wantBlocks = 1
    t_file = "SavedCalibData/time_"+str(run)+"_"+str(channel)+str(kCalib)+str(kPed)+str(kTime)+str(kVolt)+".npy" #this file has shape (numEvents, numBlocks)
    v_file = "SavedCalibData/volts_"+str(run)+"_"+str(channel)+str(kCalib)+str(kPed)+str(kTime)+str(kVolt)+".npy" #this file has shape (numEvents, numBlocks)
    b_file = "SavedCalibData/blocks_"+str(run)+"_"+str(channel)+str(kCalib)+str(kPed)+str(kTime)+str(kVolt)+".npy" #this file has shape (numEvents, numBlocks)
    if(os.path.isfile(t_file) and os.path.isfile(v_file)):
        t=np.load(t_file)
        v=np.load(v_file)
        b = np.load(b_file)
        if(wantBlocks==1):
            return(t,v,b)
        else:
            return(t,v)
    else:
        if(kForCalib==0):
            print('here?')
            ped_values = DownloadPedFile(year,run,'')
        else:
            if(station=='5'):
                print('using ARA5 calibration ped file')
                ped_values = DownloadPedFile(year,run,'./pedFiles/pedestalValues.run001431.dat')
            else:
                print('using ARA4 calibration ped file')
                ped_values = DownloadPedFile(year,run,'./pedFiles/pedestalValues.run002818.dat')
        print("Data loaded. Loading libAraEvent.so now ")
        ROOT.gSystem.Load("/users/PAS0654/osu8354/ARA/AraRootBuild/lib/libAraEvent.so")

        # try:
        #     test = ROOT.TFile.Open("/home/kahughes/ARA/data/root/run"+str(run)+"/event"+str(run)+".root")
        #
        #     calibrator = ROOT.AraEventCalibrator.Instance()
        #     eventTree = test.Get("eventTree")
        # except ReferenceError:
        try:
            print("/users/PCON0003/cond0068/ARA/ARA_Calibration/Data/event"+str(run)+".root")
            test = ROOT.TFile.Open("/users/PCON0003/cond0068/ARA/ARA_Calibration/Data/event"+str(run)+".root")
        except ReferenceError:
            print("http://icecube:skua@convey.icecube.wisc.edu/data/exp/ARA/"+year+"/filtered/L0/ARA04/"+date+"/run"+run+"/event"+run+".root")
            test = ROOT.TWebFile.Open("http://icecube:skua@convey.icecube.wisc.edu/data/exp/ARA/"+year+"/filtered/L0/ARA04/"+date+"/run"+run+"/event"+run+".root")
        if(test.IsOpen()):
            print('made it')
        else:
            return -1


        calibrator = ROOT.AraEventCalibrator.Instance()
        eventTree = test.Get("eventTree")

        rawEvent = ROOT.RawAtriStationEvent()
        eventTree.SetBranchAddress("event",ROOT.AddressOf(rawEvent))
        totalEvents = eventTree.GetEntries()
        print('total events:', totalEvents)

        #length = 1792
        print(int(channel) not in [0,1,8,9,16,17,24,25])
        print(kTime)
        print(kVolt)


        if((int(channel) not in [0,1,8,9,16,17,24,25]) and (kTime==1 and kVolt==1)):
            print('changing length')
            flength=int(length/2)
        else:
            flength=length

        all_volts = np.zeros([totalEvents,flength])
        all_t=np.zeros([totalEvents,flength])
        all_blocks=np.zeros([totalEvents])+701

        for i in range(0,totalEvents):#totalEvents):
            eventTree.GetEntry(i)

            if(rawEvent.isCalpulserEvent()==0 and kCalib==1): #if not a cal pulser and also do want to calibrate
                continue
            #if(rawEvent.isCalpulserEvent()==1):
                #print(i)

            usefulEvent = ROOT.UsefulAtriStationEvent(rawEvent,ROOT.AraCalType.kNoCalib)
            gr1 = usefulEvent.getGraphFromElecChan(channel)


            t_buff = gr1.GetX()
            v_buff = gr1.GetY()
            n = gr1.GetN()
            t_buff.SetSize(n)
            v_buff.SetSize(n)
            v = []
            t = []
            for ii in t_buff:
                t.append(ii)
            t=np.array(t)

            for ii in v_buff:
                v.append(ii)
            v=np.array(v)
            # print(colored('Got Here', 'red'))

            block_number = rawEvent.blockVec[0].getBlock()



            #Remove first block which is corrupted
            t,v,block_number = RemoveFirstBlock(t,v,block_number)


            if(kForCalib==1 and block_number%2==1):#if it's for calibrating time and voltage, only return even blocks
                continue


            #Force all waveforms to be the same length.

            #print(block_number)
            #colors =["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            #t_ped,v_ped,block_number_ped = Calibrator(station,t,v,block_number,str(channel),length,ped_values,1,0,0)
            #t_ped = t_ped-t_ped[0]
            #params = SineFit(t_ped[1::2],v_ped[1::2],0.218,450)
            #SinePlotter(t_ped,v_ped,params,0,colors[0])
            #print('')
            #print(t)
            #print(length)
            #plt.figure()
            #plt.plot(t,v)
            #plt.show()
            t,v,block_number = Calibrator(station,t,v,block_number,str(channel),length,ped_values,kPed,kTime,kVolt,run)

            if(int(channel) not in [0,1,8,9,16,17,24,25] and flength<length):
                #print('here!')
                #print(length)
                #print(flength)
                t=t[:int(length/2)]
                v=v[:length]
                #if(flength<length):
                v=v[1::2]
                #length=length/2

            #print(block_number)
            #params = SineFit(t[1::2],v[1::2],0.218,450)
            #SinePlotter(t,v,params,0,colors[1])
            #plt.show()
            #print(t)
            all_volts[i,:]=v
            all_t[i,:]=t
            all_blocks[i]=block_number

            gr1.Delete()
            usefulEvent.Delete()


        all_t = all_t[~np.all(all_volts==0,axis=1)]
        all_volts = all_volts[~np.all(all_volts == 0, axis=1)]
        all_blocks = all_blocks[all_blocks !=701]
        #print(all_blocks)
        #print(all_t)
        if(kTime==1):
            all_t,all_volts= RemoveBackwardsSamples(all_t,all_volts)

        #print(t,all_volts)
        np.save("SavedCalibData/time_"+str(run)+"_"+str(channel)+str(kCalib)+str(kPed)+str(kTime)+str(kVolt)+".npy",all_t)
        np.save("SavedCalibData/volts_"+str(run)+"_"+str(channel)+str(kCalib)+str(kPed)+str(kTime)+str(kVolt)+".npy",all_volts)
        np.save("SavedCalibData/blocks_"+str(run)+"_"+str(channel)+str(kCalib)+str(kPed)+str(kTime)+str(kVolt)+".npy",all_blocks)

        if(wantBlocks==1):
            return(all_t,all_volts,all_blocks)
        else:
            return(all_t,all_volts)


def main():
    LoadDataFromWeb("5","5337","0529","2019",0,1,1,1,1,0)#run,date,year,channel)



if __name__=="__main__":
   main()
