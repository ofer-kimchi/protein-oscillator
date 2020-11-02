#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import random 
import copy
from itertools import groupby
import time
import matplotlib
import pickle

def save(obj, filename):
# =============================================================================
#     Save an object to a file
# =============================================================================
    if not filename[-7:] == '.pickle':
        filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=4) #protocol 4 came out with Python version 3.4
    
def load(filename):
# =============================================================================
#     Load an object from a file
# =============================================================================
    if not filename[-7:] == '.pickle':
        filename = filename + '.pickle'
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


# function that returns dz/dt
def system1_ODEs(t,z,params):
    k, p = z
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
        
    dkdt = kba * ((Atot- n * k) / (1 + gammaA * k / (p + F)))**n - kua * k
    dpdt = kbb * ((Btot- m * p) / (1 + gammaB * k / (p + F)))**m - kub * p
    
    dzdt = [dkdt, dpdt]
    return(dzdt)
    
def system1_Jac(t,z,params):
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
    k, p = z
    
    jac = np.array([[
            -kua + (kba*n*(n*(F + p) + Atot*gammaA)*
                    (((Atot - k*n)*(F + p))/(F + p + k*gammaA))**n)/
                    ((-Atot + k*n)*(F + p + k*gammaA)),
                    (k*kba*n*gammaA*(((Atot - k*n)*(F + p))/(F + p + k*gammaA))**n)/
                    ((F + p)*(F + p + k*gammaA))],
    [-((kbb*m*gammaB*(((F + p)*(Btot - m*p))/(F + p + k*gammaB))**m)/(F + p + k*gammaB)),
     (-(kub*(F + p + k*gammaB)**2) + kbb*m*(((F + p)*(Btot - m*p))/(F + p + k*gammaB))**(-1 + m)*
      (k*(Btot - m*p)*gammaB - m*(F + p)*(F + p + k*gammaB)))/(F + p + k*gammaB)**2]])
        
    return jac

def system1_Sol(tSpan, params, initialConditions, toPlot, plotLog=False, atol=1e-12, rtol=1e-6):
    odeSol = scipy.integrate.solve_ivp(lambda tSpan, z: system1_ODEs(tSpan, z, params),
                                        tSpan,initialConditions,method = 'Radau', vectorized=False,
                                        jac = lambda tSpan,z: system1_Jac(tSpan,z,params),
                                        atol=atol, rtol=rtol)
    z = np.transpose(odeSol.y)
    t = odeSol.t
    
    if toPlot:
        system1_Plot(t, z, params, initialConditions, plotLog=plotLog)
    return z,t

def system1_Plot(t, z, params, initialConditions, plotLog=False):
    
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, f = params
    k = z[:,0]
    p = z[:,1]
        
#     A_err = max(abs(Atot - (a + n*e + ap + a*k/Km3 + ap*f/Km4 + n*b*e/Km5 + ap*p/Km7)))/Atot
#     B_err = max(abs(Btot - (b + m*f + bp + b*e/Km5 + m*ap*f/Km4 + (m + 1)*bp*f/Km6 + bp*p/Km8)))/Btot
#     if A_err+B_err > 1e-9:
#         print('A_err = '+str(A_err))
#         print('B_err = '+str(B_err))

    # plot results
    plt.plot(t,k,'b',label='k')
    
    if not plotLog:
        plt.plot(t,p,'k',label='p')
    else:
        plt.semilogy(t,p,'k',label='p')

    plt.ylabel('values')
    plt.xlabel('time')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize = 12)
    #plt.ylim(top=Atot)
    plt.show()


def system1_Plot_for_Fig(t, z, params, initialConditions, plotLog=False):
    
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, f = params
    k = z[:,0]
    p = z[:,1]
    lenTStart = int(len(z[:,0])/10)

    plt.plot(t[lenTStart:], k[lenTStart:], 'b', label='k')
    
    if not plotLog:
        plt.plot(t[lenTStart:], p[lenTStart:], 'k', label='p')
    else:
        plt.semilogy(t[lenTStart:], p[lenTStart:], 'k', label='p')

    plt.ylabel(r'concentration ($\mu$M)')
    plt.xlabel('time (s)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize = 12)
    #plt.ylim(top=Atot)
    plt.show()



# =============================================================================
# Solve a sample equation to check
# =============================================================================

tSpan = [0, 1000]
n = 2
m = 2 
kba = 1
kbb = 1
kua = 0.01
kub = 1 
gammaA = 10000 
gammaB = 10000
Atot = 100
Btot = 3000 
F = 1

params = [n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F]
initialConditions = [0, 0]
# z,t = system1_Sol(tSpan, params, initialConditions, toPlot=True, plotLog=True)




# =============================================================================
# Find oscillating parameters from random
# =============================================================================


def makeRandomParams_and_checkForApproximations(numToSearch, maxParams, minParams, n, m, kd=False):
    
    paramsAgreeingWithApprox = []
        
    if not kd:
        maxSecondOrderParam, maxFirstOrderParam, maxGamma, maxConc, maxF = maxParams
        minSecondOrderParam, minFirstOrderParam, minGamma, minConc, minF = minParams
    else:
        maxSecondOrderParam, maxKdParam, maxGamma, maxConc, maxF = maxParams
        minSecondOrderParam, minKdParam, minGamma, minConc, minF = minParams
    
    for i in range(numToSearch):        
        
        kba = np.exp(random.random() * (np.log(maxSecondOrderParam)-np.log(minSecondOrderParam)) + np.log(minSecondOrderParam))
        kbb = np.exp(random.random() * (np.log(maxSecondOrderParam)-np.log(minSecondOrderParam)) + np.log(minSecondOrderParam))
        
        if not kd:
            kua = np.exp(random.random() * (np.log(maxFirstOrderParam)-np.log(minFirstOrderParam)) + np.log(minFirstOrderParam))
            kub = np.exp(random.random() * (np.log(maxFirstOrderParam)-np.log(minFirstOrderParam)) + np.log(minFirstOrderParam))
        else:
            kda = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
            kdb = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
            kua = kda * kba
            kub = kdb * kbb
        
        gammaA = np.exp(random.random() * (np.log(maxGamma)-np.log(minGamma)) + np.log(minGamma))
        gammaB = gammaA #np.exp(random.random() * (np.log(maxGamma)-np.log(minGamma)) + np.log(minGamma))
    
        Atot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        Btot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        F = np.exp(random.random() * (np.log(maxF)-np.log(minF)) + np.log(minF))
                 
        params = [n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F]
        
        expectedFP = scipy.optimize.root(lambda x: system1_ODEs(0, x, params), 
                [Atot/10, Btot/10], 
                method='hybr', 
                jac=lambda x: system1_Jac(0, x, params), 
                tol=1e-9, 
                callback=None, 
                options=None)
        kfp, pfp = expectedFP.x
        
        ggConst = 5 # how much greater than is much greater than?
        if (gammaA * kfp > ggConst * (pfp + F) and gammaB * kfp > ggConst * (pfp + F) and 
            Atot > ggConst * n * kfp and Btot > ggConst * m * pfp and expectedFP.success and
            kfp < Atot/n and pfp < Btot/m and kfp > 0 and pfp > 0):
    
            paramsAgreeingWithApprox.append(params)
    
    print('Found ' + str(len(paramsAgreeingWithApprox)) + 
          ' parameters satisfying our approximations')
    return(paramsAgreeingWithApprox)
   
    
def makeRandomParams(numToSearch, maxParams, minParams, n, m, kd=False):  
    allParams = []
        
    if not kd:
        maxSecondOrderParam, maxFirstOrderParam, maxGamma, maxConc, maxF = maxParams
        minSecondOrderParam, minFirstOrderParam, minGamma, minConc, minF = minParams
    else:
        maxSecondOrderParam, maxKdParam, maxGamma, maxConc, maxF = maxParams
        minSecondOrderParam, minKdParam, minGamma, minConc, minF = minParams
    
    for i in range(numToSearch):        
        
        kba = np.exp(random.random() * (np.log(maxSecondOrderParam)-np.log(minSecondOrderParam)) + np.log(minSecondOrderParam))
        kbb = np.exp(random.random() * (np.log(maxSecondOrderParam)-np.log(minSecondOrderParam)) + np.log(minSecondOrderParam))
        
        if not kd:
            kua = np.exp(random.random() * (np.log(maxFirstOrderParam)-np.log(minFirstOrderParam)) + np.log(minFirstOrderParam))
            kub = np.exp(random.random() * (np.log(maxFirstOrderParam)-np.log(minFirstOrderParam)) + np.log(minFirstOrderParam))
        else:
            kda = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
            kdb = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
            kua = kda * kba
            kub = kdb * kbb
        
        gammaA = np.exp(random.random() * (np.log(maxGamma)-np.log(minGamma)) + np.log(minGamma))
        gammaB = gammaA #np.exp(random.random() * (np.log(maxGamma)-np.log(minGamma)) + np.log(minGamma))
    
        Atot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        Btot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        F = np.exp(random.random() * (np.log(maxF)-np.log(minF)) + np.log(minF))
          
        params = [n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F]
        allParams.append(params)
    
    return(allParams)


def findOscParameters(tmaxMultiplier, paramsToSearch, printOutTime=10000):
    
    start = time.time()
    
    oscParams = []
    oscAlpha1s = []
    oscAlpha3s = []
    oscAlpha4s = []
    oscPeriods = []
    
    nonOscParams = []
    nonOscAlpha1s = []
    nonOscAlpha3s = []
    nonOscAlpha4s = []
    
    for i, params in enumerate(paramsToSearch):
        if i % printOutTime == 0:
            print('i = ' + str(i) +' and time elapsed = ' + str(time.time()-start))
        
        n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
        initialConditions = [0, 0]

        #maximum time we'll integrate to        
        tmax = tmaxMultiplier/min([kua, kub, kba * Atot**(n-1), kbb * Btot**(m-1)]) 
        tSpan = [0, tmax]
        toPlot = False
    
        startPlot = time.time()
        z,t = system1_Sol(tSpan, params, initialConditions, toPlot)
        K = z[:, 0]
    
        dKdt = (K[1:] - K[0:-1])/(t[1:]-t[0:-1])
    
        groups = []
        uniquekeys = []
        counter = 0
        prevCounter = 0
        prevPrevCounter = 0
        prevCounterArray = []
        xChange = [] #how much does the value of Ao change in between each inflection point?
        for k, g in groupby(np.sign(dKdt)):
            groups.append(list(g))
            c = len(uniquekeys) #counter that just increases by one in each for loop
            xChange.append(np.sum(dKdt[counter:counter+len(groups[c])]*
                                  (t[counter+1:counter+len(groups[c])+1]-t[counter:counter+len(groups[c])])))
#            prevPrevPrevCounter = prevPrevCounter
            prevPrevCounter = prevCounter
            prevCounter = counter
            prevCounterArray += [t[counter]]
            counter += len(groups[c])
            uniquekeys.append(k)
        nIP = len(uniquekeys)
        
        alpha3 = ((m * kua * kbb * gammaA**(m-1) * Btot**m) / 
                           (kba * Atot**(m-1) * gammaB**m * (m*kua + kub)))
        alpha1 = (kub/kbb)**(n+1) / (kua / kba)**m * (Atot / gammaA)**(n*m) * (gammaB / Btot)**(n*m + m) * F**(n+1-m)
        alpha4 = (m-1)*kub / ((n+1)* kua)

        oscillating = False
        if (nIP > 10 and 
            min(np.abs(xChange[:-1] / xChange[5])) > 0.5 and 
            t[-3]/t[-2] > 0.95 and abs((abs(xChange[3*len(xChange)//4]) - abs(xChange[-2]))/xChange[-2]) < 1 and 
                not (z<-10000).any()): 
            
            period = prevCounterArray[-2] - prevCounterArray[-4]
            period2 = prevCounterArray[-4] - prevCounterArray[-6]
            
            if np.abs(period - period2) / period < 1e-2:  # periods must be consistent
                oscillating = True
            
            else:  # perhaps the oscillations just haven't stabilized yet
                z,t = system1_Sol([0, tmax + 10 * period], params, initialConditions, toPlot)
                K = z[:, 0]
        
                dKdt = (K[1:] - K[0:-1])/(t[1:]-t[0:-1])
            
                groups = []
                uniquekeys = []
                counter = 0
                prevCounter = 0
                prevPrevCounter = 0
                
                prevCounterArray = []
                xChange = [] #how much does the value of Ao change in between each inflection point?
                for k, g in groupby(np.sign(dKdt)):
                    groups.append(list(g))
                    c = len(uniquekeys) #counter that just increases by one in each for loop
                    xChange.append(np.sum(dKdt[counter:counter+len(groups[c])]*
                                          (t[counter+1:counter+len(groups[c])+1]-t[counter:counter+len(groups[c])])))
        #            prevPrevPrevCounter = prevPrevCounter
                    prevPrevCounter = prevCounter
                    prevCounter = counter
                    prevCounterArray += [t[counter]]
                    counter += len(groups[c])
                    uniquekeys.append(k)
                nIP = len(uniquekeys)
                                
                if (nIP > 10 and 
                        min(np.abs(xChange[:-1] / xChange[5])) > 0.5 and 
                        t[-3]/t[-2] > 0.95 and abs((abs(xChange[3*len(xChange)//4]) - abs(xChange[-2]))/xChange[-2]) < 1 and 
                        not (z<-10000).any()):  # periods must be consistent
                    
                    period = prevCounterArray[-2] - prevCounterArray[-4]
                    period2 = prevCounterArray[-4] - prevCounterArray[-6]
                    if np.abs(period - period2) / period < 1e-2:
                        oscillating = True
                    
        if oscillating:
            print(params)
            print(initialConditions)
            print(tSpan)
            print(nIP)

            z,t = system1_Sol([t[max(0, prevCounter - (prevCounter - prevPrevCounter) * 50)],t[prevCounter]], 
                                 params, initialConditions, toPlot=True, plotLog=True)
            print('time taken for this parameter set = ' + str(time.time()-startPlot))
            oscParams.append(params)
            oscAlpha1s.append(alpha1)
            oscAlpha3s.append(alpha3)
            oscAlpha4s.append(alpha4)
            oscPeriods.append(
                    prevCounterArray[len(prevCounterArray)//2] - 
                    prevCounterArray[len(prevCounterArray)//2 - 2])
        else:
            nonOscParams.append(params)
            nonOscAlpha1s.append(alpha1)
            nonOscAlpha3s.append(alpha3)
            nonOscAlpha4s.append(alpha4)
            
    timeElapsed = time.time()-start
    print('timeElapsed = ' + str(timeElapsed))
    
    return(oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods, nonOscParams, 
           nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s)


def combineOscParamsFromDifferentTMax(tmaxMultipliers, paramsToStart):
    
    oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods, nonOscParams, \
            nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s = findOscParameters(
                tmaxMultipliers[0], paramsToStart, printOutTime=1000)

    # =============================================================================
    # Find which of the parameters we searched through we had misclassified as nonoscillating
    # We assume we didn't misclassify anything as oscillating, and that any misclassification
    # was just because we didn't integrate for enough time.
    # =============================================================================
    
    for tmaxMultiplier in tmaxMultipliers[1:]:
        prevOscParams = copy.copy(oscParams)
        prevOscAlpha1s = copy.copy(oscAlpha1s)
        prevOscAlpha3s = copy.copy(oscAlpha3s)
        prevOscAlpha4s = copy.copy(oscAlpha4s)
        prevNonOscParams = copy.copy(nonOscParams)
        prevOscPeriods = copy.copy(oscPeriods)
        
        oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods, nonOscParams, \
                nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s = findOscParameters(
                        tmaxMultiplier, 
                        prevNonOscParams, 
                        printOutTime=min(2000, int(np.ceil(len(prevNonOscParams)/20))))
        
        oscParams = copy.copy(prevOscParams + oscParams)
        oscAlpha1s = copy.copy(prevOscAlpha1s + oscAlpha1s)
        oscAlpha3s = copy.copy(prevOscAlpha3s + oscAlpha3s)
        oscAlpha4s = copy.copy(prevOscAlpha4s + oscAlpha4s)
        oscPeriods = copy.copy(prevOscPeriods + oscPeriods)

    return(oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods,
           nonOscParams, nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s)
    
    
    
    
# =============================================================================
# m = n+1 --- did not make it into paper
# =============================================================================

# =============================================================================
# Choose hyperparameters:
# =============================================================================
#maxSecondOrderParam = 10**2
#minSecondOrderParam = 10**-2
#maxFirstOrderParam = 10**2
#minFirstOrderParam = 10**-2
#maxGamma = 10**2
#minGamma = 10**-2
#maxConc = 10**1
#minConc = 10**-1
#maxF = 10**-3
#minF = 10**-6
#
#maxParams = [maxSecondOrderParam, maxFirstOrderParam, maxGamma, maxConc, maxF]
#minParams = [minSecondOrderParam, minFirstOrderParam, minGamma, minConc, minF]
#
#
## =============================================================================
## Find oscillating parameters for the case m=n+1
## =============================================================================
#
#randomParamsn2m3 = makeRandomParams_and_checkForApproximations(
#        200000, maxParams, minParams, n=2, m=3)
#
#oscParamsn2m3, oscAlpha1sn2m3, oscAlpha3sn2m3, oscAlpha4sn2m3, oscPeriodsn2m3, nonOscParamsn2m3, \
#        nonOscAlpha1sn2m3, nonOscAlpha3sn2m3, nonOscAlpha4sn2m3 = combineOscParamsFromDifferentTMax(
#                [1, 100], randomParamsn2m3)
#        
#randomParamsn3m4 = makeRandomParams_and_checkForApproximations(
#        200000, maxParams, minParams, n=3, m=4)
#
#oscParamsn3m4, oscAlpha1sn3m4, oscAlpha3sn3m4, oscAlpha4sn3m4, oscPeriodsn3m4, nonOscParamsn3m4, \
#        nonOscAlpha1sn3m4, nonOscAlpha3sn3m4, nonOscAlpha4sn3m4 = combineOscParamsFromDifferentTMax(
#                [1, 100], randomParamsn3m4)
#        
#randomParamsn4m5 = makeRandomParams_and_checkForApproximations(
#        200000, maxParams, minParams, n=4, m=5)
#
#oscParamsn4m5, oscAlpha1sn4m5, oscAlpha3sn4m5, oscAlpha4sn4m5, oscPeriodsn4m5, nonOscParamsn4m5, \
#        nonOscAlpha1sn4m5, nonOscAlpha3sn4m5, nonOscAlpha4sn4m5 = combineOscParamsFromDifferentTMax(
#                [1, 100], randomParamsn4m5)
#        
##randomParamsn5m6 = makeRandomParams_and_checkForApproximations(
##        200000, maxParams, minParams, n=5, m=6)
##
##oscParamsn5m6, oscAlpha1sn5m6, oscAlpha3sn5m6, oscAlpha4sn5m6, oscPeriodsn5m6, nonOscParamsn5m6, \
##        nonOscAlpha1sn5m6, nonOscAlpha3sn5m6, nonOscAlpha4sn5m6 = combineOscParamsFromDifferentTMax(
##                [1, 100], randomParamsn5m6)
#
## =============================================================================
## Plot and checking
## =============================================================================
#
#params = [4, 5, 1.0374333479068407, 0.03199172002654881, 20.188116406191902, 0.6841366920962954, 19.53265037933001, 69.8330343008613, 3.0266602416796435, 0.16902607198676656, 5.313078069607042e-06]
#z,t = system1_Sol([0, 1000], params, [0, 0], toPlot=True, plotLog=True)
#        
#
#        
#params = nonOscParamsn3m4[np.argmax(nonOscAlpha3sn3m4)]
#n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
#z,t = system1_Sol([0, 1000], params, [0, 0], toPlot=True, plotLog=True)
#
#
#
#plt.figure()
#plt.plot(np.linspace(0, 0.7, num=len(nonOscAlpha3sn2m3)), 
#         nonOscAlpha3sn2m3, 'bo', alpha=0.1, label='non-oscillating')
#plt.plot(np.linspace(0, 0.7, num=len(oscAlpha3sn2m3)), 
#         oscAlpha3sn2m3, 'ro', alpha=0.1, label='oscillating')
#
#plt.plot(np.linspace(1, 1.7, num=len(nonOscAlpha3sn3m4)), 
#         nonOscAlpha3sn3m4, 'bo', alpha=0.1)
#plt.plot(np.linspace(1, 1.7, num=len(oscAlpha3sn3m4)), 
#         oscAlpha3sn3m4, 'ro', alpha=0.1)
#
#plt.plot(np.linspace(2, 2.7, num=len(nonOscAlpha3sn4m5)), 
#         nonOscAlpha3sn4m5, 'bo', alpha=0.1)
#plt.plot(np.linspace(2, 2.7, num=len(oscAlpha3sn4m5)), 
#         oscAlpha3sn4m5, 'ro', alpha=0.1)
#
#plt.plot(np.linspace(-0.1, 2.8, num=4), np.ones(4), '--', color='k')
#
#plt.ylabel(r'$\frac{m k_{uA} k_{bB} \gamma_A^{m-1} B_{tot}^{m}}{ k_{bA} A_{tot}^{m-1} \gamma_B^{m} ( m k_{uA} + k_{uB} )}$')
#
#plt.xticks([0.35, 1.35, 2.34], [r'$n=2$', r'$n=3$', r'$n=4$'])
##plt.yticks(range(int(np.ceil(max(oscAlpha3sn4m5)))))
##plt.xlabel(r'$n=2$; $m=3$')
##plt.yscale('log')
#
#leg = plt.legend(markerscale=3)
#for lh in leg.legendHandles: 
#    lh._legmarker.set_alpha(0.8)
#
##plt.ylim([0, 8])
#plt.title(r'$m=n+1$')
#plt.show()
#
#
#
#
#plt.figure()
#plt.plot(oscAlpha1sn2m3, oscAlpha4sn2m3, 'ro', alpha=0.1, label='oscillating')
#plt.plot(nonOscAlpha1sn2m3, nonOscAlpha4sn2m3, 'bo', alpha=0.1, label='non-oscillating')
#plt.plot(np.linspace(min(nonOscAlpha1sn2m3) * 0.1, max(nonOscAlpha1sn2m3) * 10, num=4), 
#         np.ones(4), '--', color='k')
#plt.plot(np.ones(4), 
#         np.linspace(min(nonOscAlpha4sn2m3) * 0.1, max(nonOscAlpha4sn2m3) * 10, num=4), 
#         '--', color='k')
#plt.ylabel(r'$\frac{(m - 1) k_{uB}}{(n + 1) k_{uA}}$')
#plt.xlabel(r'$\alpha_1$')
##plt.yticks(range(int(np.ceil(max(oscAlpha3sn2m3)))))
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim([10**-1.4, 10**5])
#plt.title(r'$m=n+1=3$')
##leg = plt.legend(markerscale=2.5)
##for lh in leg.legendHandles: 
##    lh._legmarker.set_alpha(0.5)
#plt.show()

# =============================================================================
# end of m = n+1 --- did not make it into paper
# =============================================================================

#
## =============================================================================
## Repeat for the case n=m=2
## =============================================================================
#
#randomParamsn2m2 = makeRandomParams_and_checkForApproximations(
#        50000, maxParams, minParams, n=2, m=2)
#
#oscParamsn2m2, oscAlpha1sn2m2, oscAlpha3sn2m2, oscAlpha4sn2m2, oscPeriodsn2m2, nonOscParamsn2m2, \
#        nonOscAlpha1sn2m2, nonOscAlpha3sn2m2, nonOscAlpha4sn2m2 = combineOscParamsFromDifferentTMax(
#                [1, 100], randomParamsn2m2)
#
#
##Params to check:
##z,t = system1_Sol([0, 500], params, [0, 0], toPlot=True, plotLog=True)
#
#
#
#fig, ax = plt.subplots()
#ax.plot(nonOscAlpha1sn2m2, nonOscAlpha4sn2m2, 'bo', alpha=0.05, label='non-oscillating')
#ax.plot(oscAlpha1sn2m2, oscAlpha4sn2m2, 'ro', alpha=0.05, label='oscillating')
#ax.plot(np.linspace(min(nonOscAlpha1sn2m2) * 0.1, max(nonOscAlpha1sn2m2) * 10, num=4), 
#         np.ones(4), '--', color='k')
#ax.plot(np.ones(4), 
#         np.linspace(min(nonOscAlpha4sn2m2) * 0.1, max(nonOscAlpha4sn2m2) * 10, num=4), 
#         '--', color='k')
#plt.ylabel(r'$\frac{(m - 1) k_{uB}}{(n + 1) k_{uA}}$')
#plt.xlabel(r'$\alpha_1$')
##plt.yticks(range(int(np.ceil(max(oscAlpha3sn2m3)))))
#plt.xscale('log')
#plt.yscale('log')
#plt.title(r'$n=m=2$')
##leg = plt.legend(markerscale=2.5)
##for lh in leg.legendHandles: 
##    lh._legmarker.set_alpha(0.5)
#
## Add inset that's just a zoom of the oscillating region
#axins = ax.inset_axes([0.5, 0.03, 0.47, 0.37], xscale='log', yscale='log', xticks=[], yticks=[])
#axins.plot(nonOscAlpha1sn2m2, nonOscAlpha4sn2m2, 'bo', alpha=0.1, label='non-oscillating')
#axins.plot(oscAlpha1sn2m2, oscAlpha4sn2m2, 'ro', alpha=0.1, label='oscillating')
#axins.plot(np.linspace(min(nonOscAlpha1sn2m2) * 0.1, max(nonOscAlpha1sn2m2) * 10, num=4), 
#         np.ones(4), '--', color='k')
#axins.plot(np.ones(4), 
#         np.linspace(min(nonOscAlpha4sn2m2) * 0.1, max(nonOscAlpha4sn2m2) * 10, num=4), 
#         '--', color='k')
#axins.set_xlim([10**-6, 10**1.5])
#axins.set_ylim([10**-0.5, 10**4])
#
#ax.indicate_inset_zoom(axins)
#
#plt.show()
#
#
#
#
#
#params = nonOscParamsn2m2[np.argmax(np.array(nonOscAlpha4sn2m2)**2 / np.array(nonOscAlpha1sn2m2))]







# =============================================================================
# Repeat for the case n=m=2 for experimental constraints
# =============================================================================

maxSecondOrderParam = 10**0
minSecondOrderParam = 10**-2
maxKdParam = 10**3
minKdParam = 10**-3
maxGamma = (3.1/14 #Identification of a Major Determinant for Serine-Threonine Kinase Phosphoacceptor Specificity
            ) / (2006 / 10300) #Mutational Analysis of a Ser/Thr Phosphatase  
minGamma = maxGamma
maxConc = 10**2
minConc = 10**-3
maxF = 10**-2
minF = 10**-8

maxParamsExp = [maxSecondOrderParam, maxKdParam, maxGamma, maxConc, maxF]
minParamsExp = [minSecondOrderParam, minKdParam, minGamma, minConc, minF]

randomParamsn2m2Exp = makeRandomParams_and_checkForApproximations(
        50000, maxParamsExp, minParamsExp, n=2, m=2, kd=True)

oscParamsn2m2Exp, oscAlpha1sn2m2Exp, oscAlpha3sn2m2Exp, oscAlpha4sn2m2Exp, oscPeriodsn2m2Exp, nonOscParamsn2m2Exp, \
        nonOscAlpha1sn2m2Exp, nonOscAlpha3sn2m2Exp, nonOscAlpha4sn2m2Exp = combineOscParamsFromDifferentTMax(
                [1, 100], randomParamsn2m2Exp)



fig, ax = plt.subplots()
ax.plot(oscAlpha1sn2m2Exp, oscAlpha4sn2m2Exp, '.', color=[i/256 for i in [12, 123, 220]], 
        alpha=0.05, label='oscillating')
ax.plot(nonOscAlpha1sn2m2Exp, nonOscAlpha4sn2m2Exp, '.', color=[i/256 for i in [255, 194, 10]], 
        alpha=0.05, label='non-oscillating')
ax.plot(np.linspace(min(nonOscAlpha1sn2m2Exp) * 0.25, max(nonOscAlpha1sn2m2Exp) * 4, num=4), 
         np.ones(4), '--', color='k')
ax.plot(np.ones(4), 
         np.linspace(min(nonOscAlpha4sn2m2Exp) * 0.25, max(nonOscAlpha4sn2m2Exp) * 4, num=4), 
         '--', color='k')
plt.ylabel(r'$\frac{(m - 1) k_{uB}}{(n + 1) k_{uA}}$')
plt.xlabel(r'$\alpha_1$')
#plt.yticks(range(int(np.ceil(max(oscAlpha3sn2m3)))))
plt.xscale('log')
plt.yscale('log')
#plt.title(r'$n=m=2$')
leg = plt.legend(markerscale=3.5)
for lh in leg.legendHandles: 
    lh._legmarker.set_alpha(0.5)

# Add inset that's just a zoom of the oscillating region
#axins = ax.inset_axes([0.5, 0.03, 0.47, 0.37], xscale='log', yscale='log', xticks=[], yticks=[])
#axins.plot(nonOscAlpha1sn2m2Exp, nonOscAlpha4sn2m2Exp, '.', color=[i/256 for i in [255, 194, 10]],
#     alpha=0.1, label='non-oscillating')
#axins.plot(oscAlpha1sn2m2Exp, oscAlpha4sn2m2Exp, '.', color=[i/256 for i in [12, 123, 220]],
#     alpha=0.1, label='oscillating')
#axins.plot(np.linspace(min(nonOscAlpha1sn2m2Exp) * 0.1, max(nonOscAlpha1sn2m2Exp) * 10, num=4), 
#         np.ones(4), '--', color='k')
#axins.plot(np.ones(4), 
#         np.linspace(min(nonOscAlpha4sn2m2Exp) * 0.1, max(nonOscAlpha4sn2m2Exp) * 10, num=4), 
#         '--', color='k')
#axins.set_xlim([10**-8, 10**1.5])
#axins.set_ylim([10**-1, 10**5])
#
#ax.indicate_inset_zoom(axins)

plt.show()


# =============================================================================
# Now with the redefined gamma = 1/alpha1
# =============================================================================

oscGammasn2m2Exp = [1/i for i in oscAlpha1sn2m2Exp]
nonOscGammasn2m2Exp = [1/i for i in nonOscAlpha1sn2m2Exp]

fig, ax = plt.subplots()
ax.plot(oscGammasn2m2Exp, oscAlpha4sn2m2Exp, '.', color=[i/256 for i in [12, 123, 220]], 
        alpha=0.05, label='oscillating')
ax.plot(nonOscGammasn2m2Exp, nonOscAlpha4sn2m2Exp, '.', color=[i/256 for i in [255, 194, 10]], 
        alpha=0.05, label='non-oscillating')
ax.plot(np.linspace(min(nonOscGammasn2m2Exp) * 0.25, max(nonOscGammasn2m2Exp) * 4, num=4), 
         np.ones(4), '--', color='k')
ax.plot(np.ones(4), 
         np.linspace(min(nonOscAlpha4sn2m2Exp) * 0.25, max(nonOscAlpha4sn2m2Exp) * 4, num=4), 
         '--', color='k')
plt.ylabel(r'$\nu$')
plt.xlabel(r'$\gamma$')
plt.xscale('log')
plt.yscale('log')
plt.xticks([10**-28, 10**-14, 10**0])
plt.yticks([10**-5, 10**0, 10**5])
#plt.title(r'$n=m=2$')
leg = plt.legend(markerscale=3.5)
for lh in leg.legendHandles: 
    lh._legmarker.set_alpha(0.5)
plt.show()





# =============================================================================
# osc frequency plot -- only including points satisfying the assumptions made in the previous plot
# =============================================================================

print('minimum timescale for oscillation = ' + str(min(oscPeriodsn2m2Exp)))
print('maximum timescale for oscillation = ' + str(max(oscPeriodsn2m2Exp)))

oscExpectedOmegaSq = []
for e, params in enumerate(oscParamsn2m2Exp):
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
    pstar = oscAlpha1sn2m2Exp[e]**(-1 / (n+1-m))
    pstarOverpstarPlus1 = 1  # pstar / (pstar + 1)
    omegaSq = n*m*kua*kub * pstarOverpstarPlus1 - (
            (n+1)*kua + (-1 + m*pstarOverpstarPlus1)*kub)**2 / 4
    
    oscExpectedOmegaSq += [copy.copy(omegaSq)]


plt.figure()
plt.plot(oscExpectedOmegaSq, [(2*np.pi/x)**2 for x in oscPeriodsn2m2Exp], '.', alpha=0.5,
                              color=[i/256 for i in [12, 123, 220]])
plt.plot(np.logspace(-9,np.log10(max(oscExpectedOmegaSq)),100000, base=10), 
         np.logspace(-9,np.log10(max(oscExpectedOmegaSq)),100000, base=10), '--k')
plt.xscale('symlog', linthreshx=1e-8)
plt.xticks([-10**4, -10**-4, 0, 10**-4, 10**4])
plt.yscale('log')
plt.yticks([10**-8, 10**-4, 10**0, 10**4])
plt.xlabel(r'$\omega^2_{pred}$ (s$^{-2}$)')
plt.ylabel(r'$\omega^2$ (s$^{-2}$)')
plt.show()




## =============================================================================
## Oscillation frequency histogram without paying attention to approximations
## =============================================================================
#maxSecondOrderParam = 10**0
#minSecondOrderParam = 10**-2
#maxKdParam = 10**3
#minKdParam = 10**-3
#maxGamma = (3.1/14 #from paper Identification of a Major Determinant for Serine-Threonine Kinase Phosphoacceptor Specificity
#            ) / (2006 / 10300) #from paper Mutational Analysis of a Ser/Thr Phosphatase  
#minGamma = maxGamma
#maxConc = 10**2
#minConc = 10**-3
#maxF = 10**-2
#minF = 10**-8
#
#maxParamsExp = [maxSecondOrderParam, maxKdParam, maxGamma, maxConc, maxF]
#minParamsExp = [minSecondOrderParam, minKdParam, minGamma, minConc, minF]
#
#randomParamsn2m2NoApprox = makeRandomParams(
#        50000, maxParamsExp, minParamsExp, n=2, m=2, kd=True)
#
#oscParamsn2m2NoApprox, oscAlpha1sn2m2NoApprox, oscAlpha3sn2m2NoApprox, oscAlpha4sn2m2NoApprox, \
#        oscPeriodsn2m2NoApprox, nonOscParamsn2m2NoApprox, nonOscAlpha1sn2m2NoApprox, \
#        nonOscAlpha3sn2m2NoApprox, nonOscAlpha4sn2m2NoApprox = combineOscParamsFromDifferentTMax(
#                [1, 100], randomParamsn2m2NoApprox)
#
#
#
#
## =============================================================================
## Alpha2 calculation
## =============================================================================
#oscGammasn2m2NoApprox = [1/i for i in oscAlpha1sn2m2NoApprox]
#nonOscGammasn2m2NoApprox = [1/i for i in nonOscAlpha1sn2m2NoApprox]
#
#oscAlpha2sn2m2NoApprox = []
#for params in oscParamsn2m2NoApprox:
#    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
#    oscAlpha2sn2m2NoApprox += [(kub / kbb)**(n/m) / (kua / kba) * (
#            (gammaB * Atot / (gammaA * Btot)))**n]
#    
#nonOscAlpha2sn2m2NoApprox = []
#for params in nonOscParamsn2m2NoApprox:
#    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
#    nonOscAlpha2sn2m2NoApprox += [(kub / kbb)**(n/m) / (kua / kba) * (
#            (gammaB * Atot / (gammaA * Btot)))**n]
#
#
#
#fig, ax = plt.subplots()
#ax.plot(nonOscGammasn2m2NoApprox, 
#         nonOscAlpha2sn2m2NoApprox, '.', color=[i/256 for i in [255, 194, 10]], 
#         alpha=0.01, label='non-oscillating')
#ax.plot(oscGammasn2m2NoApprox, 
#         oscAlpha2sn2m2NoApprox, '.', color=[i/256 for i in [12, 123, 220]], 
#        alpha=0.01, label='oscillating')
#ax.plot(np.linspace(min(nonOscGammasn2m2NoApprox) * 0.25, max(nonOscGammasn2m2NoApprox) * 4, num=4), 
#         np.ones(4), '--', color='k')
#ax.plot(np.ones(4), 
#         np.linspace(min(nonOscAlpha2sn2m2NoApprox) * 0.25, max(nonOscAlpha2sn2m2NoApprox) * 4, num=4), 
#         '--', color='k')
#plt.ylabel(r'$\gamma$')
#plt.xlabel(r'$\alpha$')
#plt.xscale('log')
#plt.yscale('log')
#plt.yticks([10**-5, 10**0, 10**5])
##plt.title(r'$n=m=2$')
#leg = plt.legend(markerscale=3.5)
#for lh in leg.legendHandles: 
#    lh._legmarker.set_alpha(0.5)
#plt.show()
#
#


# =============================================================================
# Phase space diagram of an oscillator
# =============================================================================

params = [2,
  2,
  0.017008672411578697,
  0.31511894635426035,
  0.1476559558825341,
  63.53111175046402,
  1.1369463039453072,
  1.1369463039453072,
  58.32477960431114,
  20.53440667917614,
  4.769308906434382e-08] 
#params = oscParamsn2m2Exp[0]
initialConditions = [0, 0]
z,t = system1_Sol([0, 200], params, initialConditions, toPlot=False, plotLog=False, rtol=1e-10, atol=1e-16)
system1_Plot_for_Fig(t, z, params, initialConditions, plotLog=True)
plt.figure(); plt.plot(t[350:550], z[350:550]); plt.yscale('log'); plt.show()

plt.set_cmap('twilight')
kSmall = z[14000:19000, 0]
pSmall = z[14000:19000, 1]
tSmall = t[14000:19000]
plt.figure()
plt.scatter(kSmall, pSmall, c=tSmall)
plt.yscale('log')
plt.xscale('log')
plt.xlim(min(kSmall)*0.5, max(kSmall)*2)
plt.ylim(min(pSmall)*0.01, max(pSmall)*100)
plt.xlabel(r'K ($\mu$M)')
plt.ylabel(r'P ($\mu$M)')
plt.colorbar()
plt.savefig('sys1_phase_diag.png', format='png', dpi=500)
plt.show()
