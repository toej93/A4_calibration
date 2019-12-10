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
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT.gSystem.Load("/users/PAS0654/osu8354/ARA/AraRootBuild/lib/libAraEvent.so")
