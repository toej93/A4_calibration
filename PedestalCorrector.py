from __future__ import print_function
from ROOT import TCanvas, TGraph
from ROOT import gROOT
import ROOT
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import iirfilter,lfilter,butter
from scipy import signal
from scipy.fftpack import fft
from scipy import optimize
import numpy as np
import sys
import math
from math import sin
from array import array
from pynverse import inversefunc

def FileReader(MyFile,graphname):
   #print('hello!')
   gr1 = MyFile.Get(graphname)

   t_buff = gr1.GetX()
   v_buff = gr1.GetY()
   n = gr1.GetN()

   t_buff.SetSize(n)
   v_buff.SetSize(n)

   v = np.array(v_buff,copy=True)
   t = np.array(t_buff,copy=True)

   #v=PedestalFix(v)#remove for actual data, this is just for noise data
   
   return(t,v)


def PedestalCorrect(channel,rootfile):
 

   file_name = ['/home/kahughes/ARA/data/processed/calibration_data_fulltest_elChan'+channel+'_run'+rootfile+'.root']

   print('Processing root file',rootfile,'channel',channel)

   num_blocks = 6002
   wf_length = 1024
   
   volt = np.zeros([num_blocks,wf_length])
   time = np.zeros(wf_length)
    
   odds = np.linspace(1,wf_length-1,wf_length/2.0,dtype=int)
   evens = np.linspace(0,wf_length-2,wf_length/2.0,dtype=int)

   MyFile = ROOT.TFile.Open(file_name[0])
   
   for i in range(0,num_blocks):
      time[evens],volt[i,evens]=FileReader(MyFile,'gr_E_'+str(i))
      time[odds],volt[i,odds]=FileReader(MyFile,'gr_O_'+str(i))


   #Remove corrupted blocks
   cut = 128
   time=time[cut:]
   #print(time[:])
   volt=volt[:,cut:]

   plt.figure(1,facecolor='w')
   plt.plot(time,volt[0,:])
   
   
   
   #load block numbers
   block_vals = np.loadtxt('/home/kahughes/ARA/data/processed/block_data_elChan0_run1411.txt')
   block_list = [[] for i in range(32768)] #each entry = block_number*64 +sample%64

   block_id = []
   v_val = []
   
   for i in range(0,num_blocks):#i is which block value to pick
      #print('new block!')
      my_block = int(block_vals[i]+2)%512#plus 2 because of corrupted samples
      
      for j in range(0,wf_length-cut):#j is which sample
         #print(my_block,j)
         block_list[my_block*64+j%64].append(volt[i,j])
         block_id.append(my_block*64+j%64)
         v_val.append(volt[i,j])
         if(j%64==0 and j!=0):
            #print(my_block,'is changing to',my_block+1)
            my_block=(my_block+1)%512



   block_avgs = np.zeros([512,64])
   counter = 0
   for i in range(0,512):
      for j in range(0,64):
         block_avgs[i,j]=np.mean(block_list[counter][:])
         counter = counter + 1

   new_v1 = volt[0,:64]-block_avgs[int(block_vals[i])+2]
   new_v2 = volt[0,64:128]-block_avgs[int(block_vals[i])+3]
   plt.plot(time[:128],np.append(new_v1,new_v2))
   plt.show()
         
   block_id=np.asarray(block_id)
   v_val = np.asarray(v_val)

   np.save('/home/kahughes/ARA/best_pedestals/ch_'+channel+'_ped.npy',block_avgs)

   plt.figure(0,facecolor='w')
   plt.hist(block_list[15][:])
   #plt.hist2d(block_id,v_val,bins=(32768,250),cmap=plt.cm.jet,range=np.array([(0.0,32768),(1200,2500)]))
   plt.xlabel('Sample Number')
   plt.ylabel('ADC Counts')
   #plt.imshow(block_list)
   plt.show()
   """
   print('data has been loaded.')

   volt_avgs = np.zeros(128)

   for i in range(0,128):
      volt_avgs[i] = np.mean(volt[:,i])
      #print(volt_avgs)


   plt.figure(0)
   plt.plot(time,volt[4,:])
      
   plt.figure(1)
   plt.hist(volt_avgs,bins=25)
   plt.title('Average Pedestal Corrections for all samples')
   #plt.show()

   plt.figure(2)
   plt.hist(volt[:,3])
   plt.title('Distribution of voltages for 1 sample')
   plt.show()

   np.save('ARA5_ch'+channel+'_pedestal.npy',volt_avgs)
   #plt.show()
   """

def main():
   ch = ['0','3','8','11','16','19','24','27']
   rootfile='1411'

   for i in ch:
      PedestalCorrect(i,rootfile)

   ch = ['1','2','9','10','17','18','25','26']
   rootfile='1402'
   for i in ch:
      PedestalCorrect(i,rootfile)
   #plt.show()
   
   
if __name__=="__main__":
   main()
