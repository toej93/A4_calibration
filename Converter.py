from __future__ import print_function
import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import sys
import math


def load_t2(channel,station):
    cal_dir = 'ARA'+station+'_cal_files/'

    if(station=='5'):
        rootfiles = {0:'1402',1:'1411',2:'1413',3:'1404',8:'1402',9:'1411',10:'1413',11:'1403',16:'1402',17:'1411',18:'1413',19:'1403',
        24:'1402',25:'1411',26:'1413',27:'1404'}
    if(station=='4'):
        rootfiles = {0:'2829',1:'2840',2:'2841',3:'2830',8:'2829',9:'2855',10:'2841',11:'2830',16:'2855',17:'2840',18:'2841',19:'2830',
        24:'2855',25:'2856',26:'2841',27:'2830'}

    tcals = np.load(cal_dir+'t_cal_'+rootfiles[int(channel)]+'_'+channel+'.npy')



    if channel in ['2','3','10','11','18','19','26','27']:
        #print(tcals)
        epsilon = tcals[128]-tcals[127]
        tcal_odd =tcals[1::2]

        diffs = tcal_odd[1:]-tcal_odd[:-1]
        print('even',np.where(diffs<0))
        good_indices = np.where(diffs>0)
        good_indices = np.append([0],good_indices[0]+1)
        good_indices = good_indices*2+1
    else:
        epsilon = tcals[128]-tcals[127]
        diffs = tcals[1:]-tcals[:-1]
        print(np.where(diffs<0))
        good_indices = np.where(diffs>0)
        good_indices = np.append([0],good_indices[0]+1)
    #print(channel,good_indices)
    return(tcals,good_indices,epsilon)


def load_t(channel,station):
    cal_dir = 'ARA'+station+'_cal_files/'
    #cal_dir = 'cal_files8'
    """
    N1 = [0,3,8,11,16,19,24,27]
    N2 = [1,2,9,10,17,18,25,26]

    if(int(channel) in N1):
        rootfiles = ['1402', '1403','1404','1405']
    if(int(channel) in N2):
        rootfiles = ['1411','1412','1413','1414']
    """
    N1 = [0,3,8,11,16,19,24,27]
    N2 = [1,2,9,10,17,18,25,26]
    N_special = [9,16,24,25]
    if(station=='5'):
        if(int(channel) in N1):
            rootfile='1402'
            rootfiles = ['1402', '1403','1404','1405']
        if(int(channel) in N2):
            rootfile='1411'
            rootfiles = ['1411','1412','1413','1414']
    if(station=='4'):
        if(int(channel) in N1 and int(channel) not in N_special):
            rootfile='2829'
            rootfiles = ['2829', '2830','2831','2832']
        if(int(channel) in N2 and int(channel) not in N_special):
            rootfile='2840'
            rootfiles = ['2840','2841','2842','2843']
        if(int(channel)in N_special):
            rootfiles = ['2855','2856']

    if int(channel) in [0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27]:
        num = 0
        success = 0
        while success==0:
            try:
                tcals=np.load(cal_dir+'t_cal_'+rootfiles[num]+'_'+channel+'.npy')
                success = 1
            except:
                num = num+1
        epsilon = tcals[128]-tcals[127]
        diffs = tcals[1:]-tcals[:-1]
        good_indices = np.where(diffs>0)
        good_indices = np.append([0],good_indices[0]+1)
        return(tcals,good_indices,epsilon)
    badnum = 0.0
    tcals = np.zeros([4,896])
    sum = np.zeros(896)
    counter = 4.0
    for z in range(0,4):
        try:
            tcals[z,:] = np.load(cal_dir+'t_cal_'+rootfiles[z]+'_'+channel+'.npy')
            print(len(tcals[z,:]))
            print(z,'tcal is')
            print(tcals[z,:])
            sum = sum +tcals[z,:]
        except:
            #print('no file here')
            counter = counter -1.0
    average = sum/counter
    epsilon=average[128]-average[127]
    print(len(average))
    good_indices = 0
    diffs = average[1:]-average[:-1]
    good_indices = np.where(diffs>0)

    #good_samples = average[good_indices]
    """
    if(int(channel)%8==0 or int(channel)%8==1):
        diffs = average[1:]-average[:-1]
        good_indices = np.where(diffs>0)
        good_samples = average[good_indices]
    else:
        average = average[1::2]
        diffs = average[1:]-average[:-1]
        good_indices = np.where(diffs>0)
        #good_samples = average[good_indices]
    """
    if(len(diffs[diffs<0])>0):
        print('must remove a sample!')
        print(channel)
    #print(diffs)
    #print('good samples',good_samples)
    good_indices = np.append([0],good_indices[0]+1)
    #print('good indices', good_indices)

    return(average,good_indices,epsilon)

def arraytostr2(time,cap,chan,indices):
    #print('channel is', chan)
    total_str = ''
    sam_list = ''

    if(cap==0):
        samples = indices[indices<64]
    if(cap==1):
        samples = indices[(indices>=64) & (indices<128)]

    #print('samples are', samples)
    #print('time is',time)
    #print('good indices are',indices)
    for z in range(0,len(samples)):
        total_str = total_str+str(round(time[samples[z]],5))+' '
        #sam_list = sam_list+str(int((z-extra)*2*mult1+1*mult0))+' '
        sam_list = sam_list +str(samples[z]) +' '
    #print('sample list is' , sam_list)
    #print(sam_list,total_str)
    return(total_str,sam_list,len(samples))

def arraytostr(time,cap,chan,indices):
    total_str = ''
    #print('channel is', chan)
    #print('good indices are', indices)

    if(chan<2):
        minv = 64
        maxv = 128
        mult0 = 0
        mult1 = 0.5
    else:
        minv=32
        maxv = 64
        mult0 = 1
        mult1 = 1

    if(cap==0):
        #samples = np.linspace(0,63,64,dtype=int)
        samples = indices[indices<minv]
        extra = 0
    if(cap==1):
        #samples = np.linspace(64,127,64,dtype=int)
        samples = indices[(indices>=minv) & (indices<maxv)]
        extra = minv
    #if(chan>=2):
    #    samples = samples[1::2]
    #print(samples)
    sam_list = ''
    for z in samples:
        total_str = total_str+str(round(time[z],5))+' '
        sam_list = sam_list+str(int((z-extra)*2*mult1+1*mult0))+' '
    #print('sample list is' , sam_list)

    return(total_str,sam_list,len(samples))

def makefitstring(p_pos,p_neg,zeroval,chi2,block):
    total_str = ''
    if(block%2==0):
        block=block+1
    for k in range(0,64):
        total_str = total_str + str(round(p_pos[block+k,2],5))+' '+str(round(p_pos[block+k,1],5))+' '+str(round(p_pos[block+k,0],5))+' '+str(round(p_neg[block+k,2],5))+' '+str(round(p_neg[block+k,1],5))+' '+str(round(p_neg[block+k,0],5))+' '+str(round(p_neg[block+k,3],5))+' '+str(round(zeroval[block+k],5))+' '+str(round(chi2[block+k],5)) + ' '
    return(total_str)

def makesamplelist(channel,good_samples,cap):
    if(channel==0 or channel==1):
        if(cap==0):
            num_samples = len(good_samples[good_samples<64])
            extra = 0
        else:
            num_samples = len(good_samples[(good_samples>=64) & (good_samples<128)])
            extra = 64

        sample_list = ''
        for k in range(0,64):
            if(k+extra in good_samples):
                sample_list = sample_list+str(k)+' '
    if(channel==2 or channel ==3):
        num_samples = 32 - len(bad_samples)
        sample_list = ''
        for k in range(0,32):
            if(k not in bad_samples):
                sample_list = sample_list+str(k)+' '
    if(channel>=4):
        sample_list = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63'
        num_samples = 64
    return(num_samples,sample_list)

def main():
    station='4'
    outfile_timing = open("araAtriStation"+station+"SampleTiming.txt","w")
    outfile_epsilon = open("araAtriStation"+station+"Epsilon.txt","w")
    outfile_v = open("araAtriStation"+station+"adcToVoltsConv.txt","w")
    nom_block1 = '0 0.3125 0.625 0.9375 1.25 1.5625 1.875 2.1875 2.5 2.8125 3.125 3.4375 3.75 4.0625 4.375 4.6875 5 5.3125 5.625 5.9375 6.25 6.5625 6.875 7.1875 7.5 7.8125 8.125 8.4375 8.75 9.0625 9.375 9.6875 10 10.3125 10.625 10.9375 11.25 11.5625 11.875 12.1875 12.5 12.8125 13.125 13.4375 13.75 14.0625 14.375 14.6875 15 15.3125 15.625 15.9375 16.25 16.5625 16.875 17.1875 17.5 17.8125 18.125 18.4375 18.75 19.0625 19.375 19.6875'
    nom_block2 = '20 20.3125 20.625 20.9375 21.25 21.5625 21.875 22.1875 22.5 22.8125 23.125 23.4375 23.75 24.0625 24.375 24.6875 25 25.3125 25.625 25.9375 26.25 26.5625 26.875 27.1875 27.5 27.8125 28.125 28.4375 28.75 29.0625 29.375 29.6875 30 30.3125 30.625 30.9375 31.25 31.5625 31.875 32.1875 32.5 32.8125 33.125 33.4375 33.75 34.0625 34.375 34.6875 35 35.3125 35.625 35.9375 36.25 36.5625 36.875 37.1875 37.5 37.8125 38.125 38.4375 38.75 39.0625 39.375 39.6875'
    sample_list_nom = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63'


    for dda in range(0,4):
        for chan in range(0,8):
            for cap in range(0,2):


                """
                if(chan<2 or chan>=4):
                    sample_list = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63'
                    num_samples = 64
                if(chan==2 or chan==3):
                    sample_list = '1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63'
                    num_samples = 32
                """
                if(chan<4):
                    myavg,goodindices,epsilon = load_t2(str(dda*8+chan),station)
                    np.save('ARA'+station+'_cal_files/t_cal_'+str(dda*8+chan)+'.npy',myavg)
                    #epsilon = myavg[128]-myavg[127]
                    avg_str,sample_list,num_samples = arraytostr2(myavg,cap,chan,goodindices)

                if(chan>=4):
                    epsilon = 0.3125
                    num_samples = 64
                    sample_list = sample_list_nom
                    if(cap==0):
                        avg_str = nom_block1
                    else:
                        avg_str = nom_block2

                dataline1 = str(dda) + '\t' + str(chan) + '\t' + str(cap) + '\t' + str(num_samples) + ' ' +sample_list
                dataline2 = str(dda) + '\t' + str(chan) + '\t' + str(cap) + '\t' + str(num_samples) + ' ' +avg_str
                dataline_ep = str(dda) + '\t' + str(chan) + '\t' + str(cap) + '\t' + str(epsilon)
                outfile_timing.write(dataline1 + '\n')
                outfile_timing.write(dataline2 + '\n')
                outfile_epsilon.write(dataline_ep + '\n')

    outfile_timing.close()
    outfile_epsilon.close()

    #oldfile = np.loadtxt('/home/kahughes/AraRoot/share/araCalib/ATRI/araAtriStation2adcToVoltsConv.txt')
    #print(len(oldfile[0,:]))

    cal_dir = 'ARA'+station+'_cal_files/'

    for dda in range(0,4):
        for chan in range(0,8):
            if(chan<4):
                p_pos = np.load(cal_dir+'p_pos_'+str(dda*8+chan)+'.npy')
                p_neg = np.load(cal_dir+'p_neg_'+str(dda*8+chan)+'.npy')
                chi2 = np.load(cal_dir+'chi2_pos_'+str(dda*8+chan)+'.npy')
                zeroval = np.load(cal_dir+'zerovals_'+str(dda*8+chan)+'.npy')
                p_pos = np.around(p_pos,decimals=7)
                p_neg = np.around(p_neg,decimals=7)
                chi2 = np.around(chi2,decimals=4)
                zeroval = np.around(zeroval,decimals=4)
                for block in range(0,512):
                    fitvals = makefitstring(p_pos,p_neg,zeroval,chi2,block)
                    dataline_v = str(dda) + ' '+str(chan) +' '+ str(block) +' '+ fitvals
                    outfile_v.write(dataline_v + '\n')
            if(chan>=4):
                p_pos = np.load(cal_dir+'p_pos_'+str(dda*8)+'.npy')
                p_neg = np.load(cal_dir+'p_neg_'+str(dda*8)+'.npy')
                chi2 = np.load(cal_dir+'chi2_pos_'+str(dda*8)+'.npy')
                zeroval = np.load(cal_dir+'zerovals_'+str(dda*8)+'.npy')
                for block in range(0,512):
                    fitvals = makefitstring(p_pos,p_neg,zeroval,chi2,block)
                    dataline_v = str(dda) + ' '+str(chan) +' '+ str(block) +' '+ fitvals
                    outfile_v.write(dataline_v + '\n')



    outfile_v.close()

    my_ch0_block = np.load('best_pedestals/ch_0_ped.npy')
    #text_file = open('/home/kahughes/run_004804/pedestalValues.run004804.dat','r')
    #lines = text_file.read().split(' ')
    lines = np.loadtxt('data/pedestalValues.run001431.dat')
    lines = lines[::8,3:]
    lines = lines[:512]

    diff = lines-my_ch0_block
    plt.figure(0,facecolor='w')
    plt.subplot(1,3,1)
    plt.imshow(my_ch0_block)
    plt.colorbar()
    plt.title('My Ped Values')
    plt.subplot(1,3,2)
    plt.imshow(lines)
    plt.colorbar()
    plt.title('Automatic Ped Values')
    plt.subplot(1,3,3)
    plt.imshow(diff)
    plt.colorbar()
    plt.title('Difference')
    #plt.show()

    plt.figure(1,facecolor='w')
    my_ch0_block=my_ch0_block.flatten()
    lines=lines.flatten()
    h=plt.hist2d(my_ch0_block,lines,bins=100)
    plt.hist2d(my_ch0_block,lines,bins=100)
    plt.colorbar(h[3])
    plt.xlabel('My Pedestal Values')
    plt.ylabel('Automatically Generated Pedestal Values')

    plt.figure(2,facecolor='w')
    plt.hist(diff.flatten(),bins=30)
    plt.xlabel('Differences in Pedestal Files for Each Sample')
    plt.show()

    plt.show()


    #plist = np.linspace(0,127,128)
    #plt.figure(1)
    #plt.plot(plist[1::2],myavg[1:128:2])
    #plt.show()


if __name__=="__main__":
    main()
