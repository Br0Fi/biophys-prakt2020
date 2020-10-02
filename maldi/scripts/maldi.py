#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importanweisungen

import sys
import numpy as np
import statistics as stat
import scipy as sci
import scipy.fftpack
#import sympy as sym
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.axes as axes
from matplotlib import colors as mcolors
import math
from scipy import optimize
import uncertainties as unc
import uncertainties.unumpy as unp
import uncertainties.umath as umath
unv=unp.nominal_values
usd=unp.std_devs

import os

# mathe Funktionen
def find_maxima(xarray, yarray, eps=0.):
    #find the positions of the local maxima of yarry array
    #every value with 2 neighbours of lower values gets counted
    #except if it's less than eps away in yarray from the last maximum
    #never uses the outermost values
    #assumes sorted xarray
    result = []
    lastRel = False # true if last value was higher than second to last value
    for i in range(len(xarray)-1):
        if(yarray[i]>=yarray[i+1]):
            if(lastRel):
                if(result==[] or xarray[i]>xarray[result[-1]]+eps): result.append(i)
            lastRel = False
        else: lastRel = True
    return np.array(result)
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def find_nearest(array, value):
    array[find_nearest_index(array,value)]
def normalize(ydata):
    return (ydata-np.amin(ydata))/(np.amax(ydata)-np.amin(ydata))
def mean(n):
    # find the mean value and add uncertainties
    k = np.mean(n)
    err = stat.variance(unv(n))
    return unc.ufloat(unv(k), math.sqrt(usd(k)**2 + err))

def fft(y):
    N = len(y)
    fft = scipy.fftpack.fft(y)
    return 2 * abs(fft[:N//2]) / N

    # allgemeine Fitfunktionen

def linear(x,m): # lineare Funktion mit f(x) = m * x
    return(m*x)

def gerade(x, m, b): # gerade mit = f(x) = m * x + b
    return (m*x + b)

def cyclic(x, a, f, phi):
    return a * np.sin(x * f - phi)

def cyclicOff(x, a, f, phi, offset):
    return cyclic(x, a, f, phi) + offset

def gauss(x, x0, A, d, y0):
    return A * np.exp(-(x - x0)**2 / 2 / d**2) + y0

def exponential(x, c, y0):
    return np.exp(c * x) * y0

def custom(x,n):
    m = x
    l = 650.4*10**-9#unc.ufloat(630,10)*10**-9
    #l =unp.uarray([630],[10])*10**-9
    #t = unp.uarray([5],[0.1])*10**-3
    t = 5.05*10**-3#unc.ufloat(5,0.1)*10**-3
    return (n*m*l+m*m*l*l/(4*t))/(m*l+2*t*(n-1))

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion, sowie dessen unsicherheiten zurueck
#
# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i#
# Updated on 4/6/2016
# User: https://stackoverflow.com/users/1476240/pedro-m-duarte
def fit_curvefit(datax, datay, function, p0=None, yerr=None, **kwargs):
    pfit, pcov = optimize.curve_fit(function,datax,datay,p0=p0, sigma=yerr, **kwargs)
    #pfit, pcov = optimize.curve_fit(function,datax,datay,p0=p0, sigma=yerr, epsfcn=0.0001, **kwargs)
    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return pfit_curvefit, perr_curvefit

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion, sowie dessen unsicherheiten zurueck
#
# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i#
# Updated on 4/6/2016
# User: https://stackoverflow.com/users/1476240/pedro-m-duarte
def fit_curvefit2(datax, datay, function, p0=None, yerr=None, **kwargs):
    pfit, pcov = optimize.curve_fit(function,datax,datay,p0=p0, sigma=yerr, **kwargs)
    #pfit, pcov = optimize.curve_fit(function,datax,datay,p0=p0, sigma=yerr, epsfcn=0.0001, **kwargs)
    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return unp.uarray(pfit_curvefit, perr_curvefit)
# usage zB:
# pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
# fuer eine gerade mit anfangswerten m = 1, b = 0


# Konstanten fuer einheitliche Darstellung

fig_size = (10, 6)
fig_legendsize = 22
fig_labelsize = 23
fig_ticksize = 23
matplotlib.rcParams.update({'font.size': fig_labelsize})

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#colors


# weitere Werte, Konstanten
# Werte von https://physics.nist.gov/cuu/Constants/index.html[0]

c = 299792458 # m/s
k_B = unc.ufloat_fromstr("1.38064852(79)e-23") # J K-1 [0]
h = unc.ufloat_fromstr("4.135667662(25)e-15") # eV s [0]
r_e = unc.ufloat_fromstr("2.8179403227(19)e-15") # m [0]
R = unc.ufloat_fromstr("8.3144598(48)") # J mol-1 K-1 [0]
K = 273.15 # kelvin
g = 9.81 # m/s^2
rad = 360 / 2 / math.pi
grad = 1/rad

def fwhm(x):
    #fwhm = 2.355 * sigma
    return 2*math.sqrt(2*math.log(2)) * x

visual = False

# plots
pathto = "../raw/maldi/"
files = os.listdir(pathto)
files.sort()
if(visual):
    for i,f in enumerate(files):
        print(str(i)+"\t"+f)

#fit1 734.56+-0.2
wfit1L = 734.36
wfit1R = 734.76
#fit2 760.58+-0.2
wfit2L = 760.38
wfit2R = 760.78

#cpos = 13
#for sfile in files[cpos:cpos+1]:
for sfile in files:
    data = np.loadtxt(pathto+sfile, unpack=True, skiprows = 0)
    xdata = unp.uarray(data[0], 0.01) #TODO errors bedenken oder ins Protokoll nehmen
    ydata = unp.uarray(data[1], 300) # TODO für HR vermutlich anders.
    fit1L = find_nearest_index(unv(xdata), wfit1L)
    fit1R = find_nearest_index(unv(xdata), wfit1R)
    fit2L = find_nearest_index(unv(xdata), wfit2L)
    fit2R = find_nearest_index(unv(xdata), wfit2R)
    if(visual):
        print(len(files))
        print(sfile)
        fig=plt.figure(figsize=fig_size)
        plt.plot(unv(xdata), unv(ydata), label='Messung',linewidth='1', color="blue")
        plt.grid()
        plt.tick_params(labelsize=fig_ticksize)
        plt.legend(fontsize = fig_legendsize)
        plt.xlabel(r'm/z)', size = fig_labelsize)
        plt.ylabel(r'Counts', size = fig_labelsize)
        plt.tight_layout(pad=0.3)
        plt.show()
    #          0   1   2     3
    #gauss: x, x0, A, sigma, y0
    params1 = fit_curvefit2(unv(xdata)[fit1L:fit1R], unv(ydata)[fit1L:fit1R], gauss, yerr = usd(ydata)[fit1L:fit1R], p0 = [734.56,60000,0.1,400])
    params2 = fit_curvefit2(unv(xdata)[fit2L:fit2R], unv(ydata)[fit2L:fit2R], gauss, yerr = usd(ydata)[fit2L:fit2R], p0 = [760.58,45000,0.04,1000])
    params1[2] = abs(params1[2]) # fix negative stdev
    params2[2] = abs(params2[2])
    for params in [params1,params2]:
        vfwhm = fwhm(params[2])
        auflR = params[0]/vfwhm # Massenaufl R
        snr = (params[3]+params[1])/params[3]
        #print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(sfile, params[0],params[3]+params[1],params[2], vfwhm,params[3], params[0]/vfwhm))
        #print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(sfile, unv(params[0]),unv(params[3]+params[1]),unv(params[2]), unv(vfwhm),unv(params[3]), unv(auflR)))
        #print("{0} & \\SI{{{1} \\pm {2} }} & \\SI{{{3} \\pm {4} }} & \\SI{{{5} \\pm {6} }} & \\SI{{{7} \\pm {8} }}\\\\".format(
        #    sfile, unv(params[0]), usd(params[0]), unv(vfwhm), usd(vfwhm), unv(auflR), usd(auflR), unv(snr),usd(snr))) # print for latex table
        print("{0} & \\SI{{ {1} }} & \\SI{{ {2} }} & \\SI{{ {3} }} & \\SI{{ {4} }}\\\\".format(
            sfile, params[0], vfwhm, auflR, snr)) # print for latex table


##print one specific graph:
#TODO so machen, dass der nicht alle neu rechnen muss
#currently: print the last one generated
#sfile = files[int(sys.argv[1])]
#print("chosen: {0}:  {1}".format(sys.argv[1], sfile))
#data = np.loadtxt(pathto+sfile, unpack=True, skiprows = 0)

if(visual):
    for showP in [0,1]:
        fitL = [fit1L,fit2L][showP]
        fitR = [fit1R,fit2R][showP]
        xfit = np.linspace(xdata[fitL], xdata[fitR], 1000)
        paramsP = [params1,params2][showP]
        yfit = gauss(unv(xfit), *unv(paramsP)) #TODO Achtung:params1/2
        
        fig=plt.figure(figsize=fig_size)
        plt.plot(unv(xdata[fitL:fitR]), unv(ydata[fitL:fitR]), label='Messung',linewidth='1', color="blue")
        plt.plot(unv(xfit), unv(yfit), label='Fit',linewidth='1', color="red")
        plt.grid()
        plt.tick_params(labelsize=fig_ticksize)
        plt.legend(fontsize = fig_legendsize)
        plt.xlabel(r'm/z)', size = fig_labelsize)
        plt.ylabel(r'Counts', size = fig_labelsize)
        plt.tight_layout(pad=0.3)
        #plt.savefig("../img/all_magn.png")
        plt.show()


