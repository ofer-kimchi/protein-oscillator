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
def system2_ODEs(t,z,params):
    a, b = z
    kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
    
    sa = np.sqrt(1 + 2*a)
    sb = np.sqrt(1 + 2*b)
    
    k = a**2 / ((1 + sa) * (1 + a + sa))
    p = b**2 / ((1 + sb) * (1 + b + sb))
    f_p = (F + kdB * p)
    
    dadt = -2 * gammaAK * a / (1 + sa) * kdA * k + gammaAP * (2 * Atot / kdA - a) * f_p
    dbdt = -2 * gammaBK * b / (1 + sb) * kdA * k + gammaBP * (2 * Btot / kdB - b) * f_p
    
    dzdt = [dadt, dbdt]
    return(dzdt)
    
def system2_Jac(t,z,params):
    kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
    a, b = z
    
    sa = np.sqrt(1 + 2*a)
    sb = np.sqrt(1 + 2*b)
    
    jac = np.array([[(a**2*(-1 - 2/sa)*kdA*gammaAK)/(1 + a + sa)**2 - F*gammaAP - 
                     (b**2*kdB*gammaAP)/((1 + sb)*(1 + b + sb)),
                     (b*(4*(1 + sb) + b*(7 + b + 3*sb))*(2*Atot - a*kdA)*kdB*gammaAP)/
                     (sb*(1 + sb)**2*(1 + b + sb)**2*kdA)],
    [(-2*a*(4*(1 + sa) + a*(7 + a + 3*sa))*b*kdA*gammaBK)/
     (sa*(1 + sa)**2*(1 + a + sa)**2*(1 + sb)),
     (-4*a*b*Btot*gammaBP + 4*a*(-1 + sb)*Btot*gammaBP - 3*a*b**3*kdB*gammaBP + 
      b**2*((2 + 3*a - 2*sa - a*sa)*kdA*gammaBK + a*(2*Btot - 3*kdB + sb*(-2*F + 3*kdB))*gammaBP))/
      (2.*a*b**2*sb)]])
        
    return jac

def system2_Sol(tSpan, params, initialConditions, toPlot, plotLog=False, atol=1e-12, rtol=1e-6):
    odeSol = scipy.integrate.solve_ivp(lambda tSpan, z: system2_ODEs(tSpan, z, params),
                                        tSpan,initialConditions,method = 'Radau', vectorized=False,
                                        jac = lambda tSpan,z: system2_Jac(tSpan,z,params),
                                        atol=atol, rtol=rtol)
    z = np.transpose(odeSol.y)
    t = odeSol.t
    
    if toPlot:
        system2_Plot(t, z, params, initialConditions, plotLog=plotLog)
    return z,t

def system2_Plot(t, z, params, initialConditions, plotLog=False):
    
    kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
    a = z[:,0]
    b = z[:,1]
        
#     A_err = max(abs(Atot - (a + n*e + ap + a*k/Km3 + ap*f/Km4 + n*b*e/Km5 + ap*p/Km7)))/Atot
#     B_err = max(abs(Btot - (b + m*f + bp + b*e/Km5 + m*ap*f/Km4 + (m + 1)*bp*f/Km6 + bp*p/Km8)))/Btot
#     if A_err+B_err > 1e-9:
#         print('A_err = '+str(A_err))
#         print('B_err = '+str(B_err))

    # plot results
    plt.plot(t,a,'b',label='a')
    
    if not plotLog:
        plt.plot(t,b,'k',label='b')
    else:
        plt.semilogy(t,b,'k',label='b')

    plt.ylabel('Values')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize = 12)
    #plt.ylim(top=Atot)
    plt.show()


def system2_Plot_for_fig(t, z, params, initialConditions, plotLog=False):
    
    tHr = t / (60*60)  # so time is measured in hours
    lenTStart = 0 #int(len(z[:,0])/10)

    kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
    a = z[:,0]
    b = z[:,1]
    
    Ap = Atot - (kdA / 2) * a
    Bp = Btot - (kdB / 2) * b
    
    plt.plot(tHr[lenTStart:], Ap[lenTStart:] + Bp[lenTStart:], 'b', label=r'$A^p$')
    plt.ylabel(r'Total phosphorylation ($\mu$M)')
    plt.xlabel('Time (h)')
    if plotLog:
        plt.yscale('log')
    if kdA == 0.0017365224716919757:
        plt.ylim([1, 5])
        plt.yticks([1, 3, 5])
    plt.show()


# =============================================================================
# Solve a sample equation to check
# =============================================================================

tSpan = [0, 1000]
kdA = 0.03
kdB = 0.06 
gammaAK = 3
gammaBK = 3
gammaAP = 1
gammaBP = 1 
Atot = 6
Btot = 1 
F = 10**-4

params = [kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F]
initialConditions = [2 * Atot / kdA, 2 * Btot / kdB]
#z,t = system2_Sol(tSpan, params, initialConditions, toPlot=True, plotLog=True)




# =============================================================================
# Find oscillating parameters from random
# =============================================================================

def makeRandomParams_and_checkForApproximations_sys2(numToSearch, maxParams, minParams):
    
    paramsAgreeingWithApprox = []
    
    maxKdParam, maxGammaK, maxGammaP, maxConc, maxF = maxParams
    minKdParam, minGammaK, minGammaP, minConc, minF = minParams
    
    for i in range(numToSearch):        
        
        kdA = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
        kdB = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
        
        gammaAK = np.exp(random.random() * (np.log(maxGammaK)-np.log(minGammaK)) + np.log(minGammaK))
        gammaBK = gammaAK
        
        gammaAP = np.exp(random.random() * (np.log(maxGammaP)-np.log(minGammaP)) + np.log(minGammaP))
        gammaBP = gammaAP
        
        Atot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        Btot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        F = np.exp(random.random() * (np.log(maxF)-np.log(minF)) + np.log(minF))
                 
        params = [kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F]
        
        expectedFP = scipy.optimize.root(lambda x: system2_ODEs(0, x, params), 
                [2 * Atot / kdA, 2 * Btot / kdB], 
                method='hybr', 
                jac=lambda x: system2_Jac(0, x, params), 
                tol=1e-9, 
                callback=None, 
                options=None)
        afp, bfp = expectedFP.x
                
        ggConst = 5 # how much greater than is much greater than?
        if (afp > ggConst and bfp * ggConst < 1 and
            Atot * kdB * gammaAP * gammaBK**2 > Btot**2 * gammaAK * gammaBP**2 and 
            expectedFP.success and afp > 0 and bfp > 0):
                
            paramsAgreeingWithApprox.append(params)
    
    print('Found ' + str(len(paramsAgreeingWithApprox)) + 
          ' parameters satisfying our approximations')
    return(paramsAgreeingWithApprox)


def makeRandomParams_and_checkForApproximations_2_sys2(numToSearch, maxParams, minParams):
    
    paramsAgreeingWithApprox = []
    
    maxKdParam, maxGammaK, maxGammaP, maxConc, maxF = maxParams
    minKdParam, minGammaK, minGammaP, minConc, minF = minParams
    
    for i in range(numToSearch):        
        
        kdA = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
        kdB = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
        
        gammaAK = np.exp(random.random() * (np.log(maxGammaK)-np.log(minGammaK)) + np.log(minGammaK))
        gammaBK = np.exp(random.random() * (np.log(maxGammaK)-np.log(minGammaK)) + np.log(minGammaK))
        
        gammaAP = np.exp(random.random() * (np.log(maxGammaP)-np.log(minGammaP)) + np.log(minGammaP))
        gammaBP = np.exp(random.random() * (np.log(maxGammaP)-np.log(minGammaP)) + np.log(minGammaP))
        
        Atot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        Btot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        F = np.exp(random.random() * (np.log(maxF)-np.log(minF)) + np.log(minF))
        
        params = [kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F]
        
        expectedFP = scipy.optimize.root(lambda x: system2_ODEs(0, x, params), 
                [2 * Atot / kdA, 2 * Btot / kdB], 
                method='hybr', 
                jac=lambda x: system2_Jac(0, x, params), 
                tol=1e-9, 
                callback=None, 
                options=None)
        afp, bfp = expectedFP.x
                
        ggConst = 5 # how much greater than is much greater than?
        if (afp * ggConst < 1 and bfp * ggConst < 1 and
            afp * kdA * ggConst < 2 * Atot and bfp * kdB * ggConst < 2 * Btot and
            expectedFP.success and afp > 0 and bfp > 0):
            
            paramsAgreeingWithApprox.append(params)
    
    print('Found ' + str(len(paramsAgreeingWithApprox)) + 
          ' parameters satisfying our approximations')
    return(paramsAgreeingWithApprox)


def makeRandomParams_sys2(numToSearch, maxKd, minKd, Atot, Btot, F, gammaAK, gammaBK, gammaAP, gammaBP):
    
    params = []
    
    for i in range(numToSearch):        
        kdA = np.exp(random.random() * (np.log(maxKd)-np.log(minKd)) + np.log(minKd))
        kdB = np.exp(random.random() * (np.log(maxKd)-np.log(minKd)) + np.log(minKd))
                         
        params.append([kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F])
        
    return(params)
    
    
def makeRandomParams_sys2_2(numToSearch, maxKd, minKd, maxConc, minConc, maxF, minF, gammaAK, gammaBK, gammaAP, gammaBP):
    
    params = []
    
    for i in range(numToSearch):        
        kdA = np.exp(random.random() * (np.log(maxKd)-np.log(minKd)) + np.log(minKd))
        kdB = np.exp(random.random() * (np.log(maxKd)-np.log(minKd)) + np.log(minKd))
        Atot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        Btot = np.exp(random.random() * (np.log(maxConc)-np.log(minConc)) + np.log(minConc))
        F = np.exp(random.random() * (np.log(maxF)-np.log(minF)) + np.log(minF))

        params.append([kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F])
        
    return(params)



def findOscParameters_sys2(tmaxMultiplier, paramsToSearch, printOutTime=10000):
    
    start = time.time()
    
    oscParams = []
    oscMaxF = []
    oscKdA = []
    oscKdB = []
    oscPeriods = []
    
    nonOscParams = []
    nonOscMaxF = []
    nonOscKdA = []
    nonOscKdB = []
    
    for i, params in enumerate(paramsToSearch):
        if i % printOutTime == 0:
            print('i = ' + str(i) +' and time elapsed = ' + str(time.time()-start))
        
        kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
        initialConditions = [2 * Atot / kdA, 2 * Btot / kdB]

        #maximum time we'll integrate to
        tmax = tmaxMultiplier/min([gammaAK * Atot, #gammaAK * kdA, 
                                   gammaAP * Atot, #gammaAP * kdA,
                                   gammaBK * Btot, #gammaBK * kdB,
                                   gammaBP * Btot, #gammaBP * kdB
                                   ]) 
        tSpan = [0, tmax]
        toPlot = False
        
        startPlot = time.time()
        z,t = system2_Sol(tSpan, params, initialConditions, toPlot)
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
        
        maxF = 4 * Btot**6 * kdA**2 * gammaAK**4 * (gammaBK - 3 * gammaAK) * gammaBP**6 / (
                Atot**4 * kdB**3 * gammaAP**4 * gammaBK **4 * (gammaBK + 3 * gammaAK)**3 * F)

        oscillating = False
        
        if (nIP > 10 and 
            min(np.abs(xChange[5:-1] / xChange[5])) > 0.5 and 
            t[-3]/t[-2] > 0.95 and abs((abs(xChange[3*len(xChange)//4]) - abs(xChange[-2]))/xChange[-2]) < 1 and 
                not (z<-10000).any()):
            
            period = prevCounterArray[-2] - prevCounterArray[-4]
            period2 = prevCounterArray[-4] - prevCounterArray[-6]
            
            if np.abs(period - period2) / period < 1e-2:  # periods must be consistent
                oscillating = True
            else:  # perhaps the oscillations just haven't stabilized yet
                z,t = system2_Sol([0, tmax + 10 * period], params, initialConditions, toPlot)
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

            z,t = system2_Sol([t[max(0, prevCounter - (prevCounter - prevPrevCounter) * 50)],t[prevCounter]], 
                                 params, initialConditions, toPlot=True, plotLog=True)
            print('time taken for this parameter set = ' + str(time.time()-startPlot))
            oscParams.append(params)
            oscMaxF.append(maxF)
            oscKdA.append(kdA)
            oscKdB.append(kdB)
            oscPeriods.append(period)
        else:
            nonOscParams.append(params)
            nonOscMaxF.append(maxF)
            nonOscKdA.append(kdA)
            nonOscKdB.append(kdB)
            
    timeElapsed = time.time()-start
    print('timeElapsed = ' + str(timeElapsed))
    
    return(oscParams, oscMaxF, oscKdA, oscKdB, oscPeriods, nonOscParams, nonOscMaxF, nonOscKdA, nonOscKdB)


def combineOscParamsFromDifferentTMax_sys2(tmaxMultipliers, paramsToStart,
                                           prevOscParams=[], prevOscMaxF=[],
                                           prevOscKdA=[], prevOscKdB=[], prevOscPeriods=[]):
    
    oscParams, oscMaxF, oscKdA, oscKdB, oscPeriods, nonOscParams, nonOscMaxF, nonOscKdA, \
            nonOscKdB = findOscParameters_sys2(
                tmaxMultipliers[0], 
                paramsToStart, 
                printOutTime=min(1000, int(np.ceil(len(paramsToStart)/20))))

    oscParams = copy.copy(prevOscParams + oscParams)
    oscMaxF = copy.copy(prevOscMaxF + oscMaxF)
    oscKdA = copy.copy(prevOscKdA + oscKdA)
    oscKdB = copy.copy(prevOscKdB + oscKdB)
    oscPeriods = copy.copy(prevOscPeriods + oscPeriods)
    
    # =============================================================================
    # Find which of the parameters we searched through we had misclassified as nonoscillating
    # We assume we didn't misclassify anything as oscillating, and that any misclassification
    # was just because we didn't integrate for enough time.
    # =============================================================================
    
    for tmaxMultiplier in tmaxMultipliers[1:]:
        print('tmaxMultiplier = ' + str(tmaxMultiplier))
        prevOscParams = copy.copy(oscParams)
        prevOscMaxF = copy.copy(oscMaxF)
        prevOscKdA = copy.copy(oscKdA)
        prevOscKdB = copy.copy(oscKdB)
        prevOscPeriods = copy.copy(oscPeriods)
        
        prevNonOscParams = copy.copy(nonOscParams)
        
        oscParams, oscMaxF, oscKdA, oscKdB, oscPeriods, nonOscParams, nonOscMaxF, nonOscKdA, \
                nonOscKdB = findOscParameters_sys2(
                        tmaxMultiplier, 
                        prevNonOscParams, 
                        printOutTime=min(2000, int(np.ceil(len(prevNonOscParams)/20))))
        
        oscParams = copy.copy(prevOscParams + oscParams)
        oscMaxF = copy.copy(prevOscMaxF + oscMaxF)
        oscKdA = copy.copy(prevOscKdA + oscKdA)
        oscKdB = copy.copy(prevOscKdB + oscKdB)
        oscPeriods = copy.copy(prevOscPeriods + oscPeriods)

    return(oscParams, oscMaxF, oscKdA, oscKdB, oscPeriods, nonOscParams, nonOscMaxF, nonOscKdA, nonOscKdB)



# =============================================================================
# Find oscillating parameters from random search
# =============================================================================

#results = dict()

#AtotVec = [1, 3, 10, 30]
#gammaKVec = [(3.1/14) / (2006/10300)]#[5, 50, 500]
#FVec = [10**-3, 10**-4, 10**-5]
#
#for F in FVec:
#    for Atot in AtotVec:
#        for gammaK in gammaKVec:
#            key = 'Atot_' + str(Atot) + '_gammaK_' + str(gammaK) + '_F_' + str(F) + '_2'
#            print(key)
#            
#            randomParams = makeRandomParams_sys2(
#                    numToSearch=10000, maxKd=10**1, minKd=10**-3, Atot=Atot, Btot=1, F=F, 
#                    gammaAK=gammaK, gammaBK=gammaK, gammaAP=1, gammaBP=1)
#            
#            oscParams, oscMaxF, oscKdA, oscKdB, nonOscParams, nonOscMaxF, nonOscKdA, nonOscKdB = \
#                    combineOscParamsFromDifferentTMax_sys2([100, 10000, 1000000], randomParams)
#            
#            results[key] = [oscParams, oscMaxF, oscKdA, oscKdB, nonOscParams, nonOscMaxF, nonOscKdA, nonOscKdB]


results = load('results_sys_2_11_7.pickle')

AtotVec = [3, 10, 30]  #Atot = 1 doesn't give any oscillations
gammaK = (3.1/14) / (2006/10300)
FVec = [10**-3, 10**-4, 10**-5] #Ftot = 10**-2 doesn't give any oscillations
fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=[10,9])
for i, Atot in enumerate(AtotVec[::-1]):
    for j, F in enumerate(FVec):
        key = 'Atot_' + str(Atot) + '_gammaK_' + str(gammaK) + '_F_' + str(F) + '_2'
        oscParams, oscMaxF, oscKdA, oscKdB, nonOscParams, nonOscMaxF, nonOscKdA, nonOscKdB = \
                results[key]
        axs[i, j].plot(nonOscKdA, nonOscKdB, '.', color=[i/256 for i in [255, 194, 10]], alpha=0.1)
        axs[i, j].plot(oscKdA, oscKdB, '.', color=[i/256 for i in [12, 123, 220]], alpha=0.2)

        axs[i, j].set_xscale('log')
        axs[i, j].set_yscale('log')
        
        if i == 0 and j == len(FVec) - 1:
            axs[i, j].set_xlabel(r'$k_{d\kappa}/\rho_{tot}$')# + '\n' + str(gammaK), fontsize=24)
            axs[i, j].set_ylabel(#str(Atot) + '\n' + 
                    r'$k_{d\rho}/\rho_{tot}$')#, fontsize=24)
#            axs[i, j].set_xticks([10**-4, 10**0, 10**4])
#            axs[i, j].set_yticks([10**-4, 10**0, 10**4])
            
            plt.sca(axs[i, j])
            axs[i, j].yaxis.tick_right()
            axs[i, j].xaxis.tick_top()
#            plt.xticks([10**-4, 10**0, 10**4], fontsize=16)
#            plt.yticks([10**-4, 10**0, 10**4], fontsize=16)
            plt.xticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0], fontsize=16,
                       #labels=[r'$10^{-5}$','',r'$10^{-3}$','',r'$10^{-1}$','']
                       )
            plt.yticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0], fontsize=16,
                       #labels=[r'$10^{-5}$','',r'$10^{-3}$','',r'$10^{-1}$','']
                       )

        else:
#            axs[i, j].set_xticks([])
#            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0])
            axs[i, j].set_yticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0])
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])

            if j == 0:
#                if Atot == 10:
#                    axs[i, j].set_ylabel(r'$10^{1}$', fontsize=24)
#                elif Atot == 100:
#                    axs[i, j].set_ylabel(r'$10^{2}$', fontsize=24)
#                elif Atot == 1000:
#                    axs[i, j].set_ylabel(r'$10^{3}$', fontsize=24)
                if False:
                    pass
                else:
                    axs[i, j].set_ylabel(str(Atot), fontsize=24)
                    
            if i == len(AtotVec) - 1:
                if F == 10**-3:
                    axs[i, j].set_xlabel(r'$10^{-3}$', fontsize=24)
                elif F == 10**-4:
                    axs[i, j].set_xlabel(r'$10^{-4}$', fontsize=24)
                elif F == 10**-5:
                    axs[i, j].set_xlabel(r'$10^{-5}$', fontsize=24)
                elif F == 10**-6:
                    axs[i, j].set_xlabel(r'$10^{-6}$', fontsize=24)
                else:
                    axs[i, j].set_xlabel(str(F), fontsize=24)
        axs[i, j].yaxis.set_ticks_position('both')
        axs[i, j].xaxis.set_ticks_position('both')
        axs[i, j].set_xlim([10**-2, 10**0])
        axs[i, j].set_ylim([10**-2, 10**0])
fig.text(0.5, -0.02, r'$\tilde{P}_{tot}/\rho_{tot}$', ha='center', fontsize=30)
fig.text(-0.02, 0.5, r'$\kappa_{tot}/\rho_{tot}$', va='center', rotation='vertical', fontsize=30)
plt.tight_layout(h_pad=0, w_pad=0.15)
plt.show()



# Make legend
plt.figure()
plt.plot([0,10],[0,10], '.', color=[i/256 for i in [12, 123, 220]], 
        alpha=0.05, label='oscillating')
plt.plot([0,10],[0,10], '.', color=[i/256 for i in [255, 194, 10]], 
        alpha=0.05, label='non-oscillating')

leg = plt.legend(markerscale=3.5, loc='center')
for lh in leg.legendHandles: 
    lh._legmarker.set_alpha(0.5)
plt.show()

# =============================================================================
# Examples
# =============================================================================
#oscillating parameters
#oscParamsExp = [[0.0027405151012453396, 0.00042070671804512685, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 4.316115832243102, 0.06454517431406613, 4.772068846909474e-06],
#                [0.006909190815090066, 0.0015200045424874859, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 3.299278565594285, 0.08464921622391819, 1.7376472514666925e-05],
#                [0.000109197390078942, 0.0003812235109324108, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 1.7606376524578338, 0.02004851047264581, 5.5459380717282704e-05],
#                [0.00026286522990849907, 1.0137465697310194e-05, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 5.470947546326025, 0.012201833188502074, 1.7237522507533572e-06],
#                [0.00692900568664362, 0.012601525263139613, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 27.418297720230658, 0.5345286266535181, 3.450210865633003e-05],
#                [0.0001417398670526383, 2.5755503480485496e-05, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 93.47750017166443, 0.09376892988822558, 3.036890180731937e-06],
#                [0.033456717683060494, 0.3712169025829943, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 15.696816789919467, 1.8857844107969015, 2.691472125371366e-05],
#                [0.0011688990375140764, 0.00010408395974765446, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 9.70406194661247, 0.047876392127070504, 9.720602850292035e-06],
#                [0.0013273145496809754, 0.0002187862452340915, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 51.55116827514682, 0.18284229993845405, 2.8250216912796216e-05],
#                [0.07726970642853082, 0.6921073524829503, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 18.382607534073376, 2.8664522191918755, 0.0008111848174770369],
#                [0.009136084151318662, 0.0013139550650433532, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 87.68448854729938, 0.6758659934091337, 2.059496245631704e-06],
#                [0.00035274175150784494, 2.69729602966115e-05, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 78.89222253900617, 0.08176967123770265, 2.011653319949742e-06],
#                [0.0006384060721402517, 0.00036727309694523037, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 49.066368859121184, 0.14844666896397754, 1.2033407709744227e-06],
#                [0.0011731460856892024, 0.0012671070252422729, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 7.621862673805529, 0.08680910417786271, 3.936033208346747e-05],
#                [0.0004057008262946171, 0.00892291758207391, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 0.01968530581800862, 0.010707840477929643, 1.4632117420024677e-06],
#                [0.03461607068031029, 0.09704499689122016, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 17.104958893666833, 1.0495587487139044, 4.5225633758687665e-06],
#                [0.009820489391844456, 0.007342907767068722, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 5.70720974675152, 0.25291975181955717, 0.00025475670487605143],
#                [0.0017365224716919757, 0.007253916167248211, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 4.702544862423312, 0.16278590555321057, 2.6804303924361173e-06],
#                [0.0013977915398441939, 7.943795223336854e-05, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 32.82360515989102, 0.07764514442294919, 2.7554987860550544e-06],
#                [0.05049609691353438, 0.14266667876956235, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 6.878033226933746, 1.003060669735341, 1.0514069013163502e-05],
#                [0.0910204420612403, 0.36745807191862473, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 6.498894759145005, 1.458665514857553, 3.8441064733365595e-06],
#                [0.010914691184349208, 0.00335042200130895, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 64.71195337055956, 0.5548994562537045, 1.6673704666665162e-05],
#                [0.02838619227295303, 0.012321113724379435, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 70.586513854834, 1.2445966083588624, 0.00012847100374410974],
#                [0.016310903376361395, 0.005186657834772619, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 51.362568580560236, 0.714077170745761, 1.6349233582902662e-06],
#                ]
#
#for params in oscParamsExp:
#    print(params)
#    kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
#    initialConditions = [2 * Atot / kdA, 2 * Btot / kdB]
#    z,t = system2_Sol([0, 1000], params, initialConditions, toPlot=False, plotLog=True)
#    system2_Plot_for_fig(t, z, params, initialConditions, plotLog=True)
#
#params = [0.0017365224716919757, 0.007253916167248211, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 4.702544862423312, 0.16278590555321057, 2.6804303924361173e-06]
##params = [0.07726970642853082, 0.6921073524829503, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 18.382607534073376, 2.8664522191918755, 0.0008111848174770369]
#kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
#initialConditions = [2 * Atot / kdA, 2 * Btot / kdB]
#z,t = system2_Sol([0, 200000], params, initialConditions, toPlot=False, plotLog=True)
#system2_Plot_for_fig(t, z, params, initialConditions, plotLog=False)
#




# =============================================================================
# Look at the phase space plot for an arbitrary oscillator
# =============================================================================

params = [0.0017365224716919757, 0.007253916167248211, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 4.702544862423312, 0.16278590555321057, 2.6804303924361173e-06]
#params = [0.033456717683060494, 0.3712169025829943, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 15.696816789919467, 1.8857844107969015, 2.691472125371366e-05]
#params = [0.07726970642853082, 0.6921073524829503, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 18.382607534073376, 2.8664522191918755, 0.0008111848174770369]
kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params
initialConditions = [2 * Atot / kdA, 2 * Btot / kdB]
z,t = system2_Sol([0, 200000], params, initialConditions, toPlot=False, plotLog=True)
system2_Plot_for_fig(t, z, params, initialConditions, plotLog=False)

aSmall = z[200:480,0]
bSmall = z[200:480,1]
tSmall = t[200:480]

ApSmall = Atot - (kdA / 2) * aSmall
BpSmall = Btot - (kdB / 2) * bSmall

A_unphosSmall = Atot - ApSmall
B_unphosSmall = Btot - BpSmall

xASmall = 2 * A_unphosSmall / kdA
xBSmall = 2 * B_unphosSmall / kdB

kSmall = A_unphosSmall - kdA * xASmall / (1 + xASmall + np.sqrt(1 + 2 * xASmall))
pSmall = B_unphosSmall - kdB * xBSmall / (1 + xBSmall + np.sqrt(1 + 2 * xBSmall))

plt.set_cmap('twilight')
plt.figure()
plt.scatter(kSmall, pSmall, c=tSmall)
plt.yscale('log')
plt.xscale('log')
plt.xlim(min(kSmall)*0.5, max(kSmall)*2)
plt.ylim(min(pSmall)*0.5, max(pSmall)*2)
plt.xlabel(r'K ($\mu$M)')
plt.ylabel(r'P ($\mu$M)')
plt.colorbar()
plt.show()




# =============================================================================
# Oscillation periods for all possible parameters
# =============================================================================

maxKdParam = 10**3
minKdParam = 10**-3
maxConc = 10**2
minConc = 10**-3
maxF = 10**-2
minF = 10**-8

randomParamsAll = makeRandomParams_sys2_2(
        500000, maxKdParam, minKdParam, maxConc, minConc, maxF, minF, 
        3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300)

oscParamsAll, oscMaxFAll, oscKdAAll, oscKdBAll, oscPeriodsAll, nonOscParamsAll, nonOscMaxFAll, nonOscKdAAll, nonOscKdBAll = \
        combineOscParamsFromDifferentTMax_sys2([10000, 1000000], randomParamsAll)


plt.figure()
plt.hist(np.log10(oscPeriodsAll), 20)
plt.show()


exampleOsc = [
        [0.004208647598279732, 0.006853075006250446, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 0.38332143096192983, 0.04569909335589712, 2.7145308216763477e-06],
        [0.023966133215795188, 0.011532742437969156, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 17.85253509735449, 0.6277150552313967, 3.5224757977936643e-06],
        [0.0014583609972353188, 0.0015163404740074502, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 6.780392301511485, 0.10245105795080583, 2.431670891174658e-06],
        [0.002358849746793811, 0.004705481021989813, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 1.56513864448026, 0.09095217780173721, 1.2249283195189822e-06],
        [0.04637043883752713, 0.007416045986659947, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 15.281524722046766, 0.3072332053367441, 3.734497686280399e-07],
        [0.07143959222247113, 0.0353270731830509, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 16.74409292590358, 0.7674972718102512, 3.0458599818661697e-07],
        [0.003915784319400101, 0.35896044833524277, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 0.07635012712345123, 0.13551879393864363, 2.6934871089671505e-08],
        [0.013851637348279714, 0.07830725597065832, 0.22142857142857145, 0.22142857142857145, 0.19475728155339805, 0.19475728155339805, 12.918887515289383, 0.9303179816269448, 2.504781446346332e-08]
        ]



(oscParamsAll, oscMaxFAll, oscKdAAll, oscKdBAll, oscPeriodsAll, nonOscParamsAll, 
 nonOscMaxFAll, nonOscKdAAll, nonOscKdBAll) = load(
         'sys2_no_approx_500000.pickle')

(oscParamsAll2, oscMaxFAll2, oscKdAAll2, oscKdBAll2, oscPeriodsAll2, nonOscParamsAll2, 
 nonOscMaxFAll2, nonOscKdAAll2, nonOscKdBAll2) = load(
         'n2m2_no_approx.pickle')

plt.figure()
plt.hist(np.log10(oscPeriodsAll2), np.linspace(-1, 6, 25), label='bounded',alpha=0.5, density=True)
plt.hist(np.log10(oscPeriodsAll), np.linspace(-1, 6, 25), label='unbounded', alpha=0.5, density=True)
plt.legend()
plt.xlabel('Period (s)')
plt.xticks([0, 2, 4, 6], [r'$10^0$', r'$10^2$', r'$10^4$', r'$10^6$'])
plt.ylabel('Normalized counts')
plt.show()




# =============================================================================
# Show some trajectories for some parameter sets
# =============================================================================

def phaseSpacePlotSys2(params, t_to_integrate, t_index_to_show, markersize=36, atol=1e-12, rtol=1e-6):
    kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F = params

    initialConditions = [2 * Atot / kdA, 2 * Btot / kdB]
    z,t = system2_Sol([0, t_to_integrate], params, initialConditions, 
                      toPlot=False, plotLog=False, atol=atol, rtol=rtol)
    
#    plt.figure(); plt.plot(t[t_index_to_show], z[t_index_to_show]); plt.yscale('log'); plt.show()
    print('kdk/ptot = ' + str(kdA / Btot))
    print('kdp/ptot = ' + str(kdB / Btot))
    print('ktot/ptot = ' + str(Atot / Btot))
    print('ptilde/ptot = ' + str(F / Btot))
    
    a = z[:, 0]
    b = z[:, 1]
    sa = np.sqrt(1 + 2*a)
    sb = np.sqrt(1 + 2*b)
    
    k = kdA * a**2 / ((1 + sa) * (1 + a + sa))
    p = kdB * b**2 / ((1 + sb) * (1 + b + sb))
    
    plt.set_cmap('twilight')
    kSmall = k[t_index_to_show]
    pSmall = p[t_index_to_show]
    tSmall = t[t_index_to_show] - t[t_index_to_show[0]]
    plt.figure()
    plt.scatter(kSmall, pSmall, c=tSmall, s=markersize)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(min(kSmall)*0.5, max(kSmall)*2)
    plt.ylim(min(pSmall)*0.5, max(pSmall)*2)
    plt.xlabel(r'K ($\mu$M)')
    plt.ylabel(r'P ($\mu$M)')
    plt.colorbar(label='Time (s)')
    #plt.savefig('sys2_phase_diag.png', format='png', dpi=500)
    plt.show()


phaseSpacePlotSys2(
        [0.05, 0.05, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 30, 1, 1e-5], 
         7000, range(750, 1120), markersize=16, rtol=1e-8)

phaseSpacePlotSys2(
        [0.01, 0.05, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 30, 1, 1e-5], 
         14000, range(1016, 1900), markersize=16, rtol=1e-8)

phaseSpacePlotSys2(
        [0.03, 0.087, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 10, 1, 1e-4], 
         7000, range(760, 1370), markersize=16, rtol=1e-8)


#phaseSpacePlotSys2(
#        [0.1, 0.1, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 10, 1, 1e-4], 
#         7000, range(750, 1120), markersize=16, rtol=1e-8)

#phaseSpacePlotSys2(
#        [0.027, 0.17, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 10, 1, 1e-4], 
#         10000, range(750, 1120), markersize=16, rtol=1e-8)
#
#phaseSpacePlotSys2(
#        [0.1, 0.12, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 10, 1, 1e-4], 
#         10000, range(750, 1120), markersize=16, rtol=1e-8)