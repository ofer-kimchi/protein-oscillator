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
def system1_ODEs_noF(t,z,params):
    a, b, k, p, ap, bp, AK, BK, ApP, BpP = z
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params
        
#    dadt = -n*(kba*a**n - kua*k) - kbak*a*k + kuak*AK + kcap*(ApP+ApF)
#    dbdt = -m*(kbb*b**m - kub*p) - kbbk*b*k + kubk*BK + kcbp*(BpP+BpF)
#    dkdt = kba*a**n - kua*k - kbak*a*k + (kuak+kcak)*AK - kbbk*b*k + (kubk+kcbk)*BK
#    dpdt = kbb*b**m - kub*p - kbap*ap*p + (kuap+kcap)*(ApP) - kbbp*bp*(p) + (kubp+kcbp)*(BpP)
#    dapdt = -kbap*ap*(p+f) + kuap*(ApP+ApF) + kcak*AK
#    dbpdt = -kbbp*bp*(p+f) + kubp*(BpP+BpF) + kcbk*BK
#    dAKdt = kbak*a*k - (kuak+kcak)*AK
#    dBKdt = kbbk*b*k - (kubk+kcbk)*BK
#    dApPdt = kbap*ap*(p) - (kuap+kcap)*(ApP)
#    dBpPdt = kbbp*bp*(p) - (kubp+kcbp)*(BpP)
#    dApFdt = kbap*ap*(f) - (kuap+kcap)*(ApF)
#    dBpFdt = kbbp*bp*(f) - (kubp+kcbp)*(BpF)
#    dFdt = -kbap*ap*(f) + (kuap+kcap)*(ApF) - kbbp*bp*(f) + (kubp+kcbp)*(BpF)
#    
#    dzdt = [dadt, dbdt, dapdt, dbpdt, dkdt, dpdt, dAKdt, dBKdt, dApPdt, dBpPdt, dApFdt, dBpFdt, dFdt]
#    return(dzdt)
    
    paramsMod = [n, m, kba/n, kbb/m, kua, kub, kbak/2, kuak, kcak, kbbk/2, kubk, 
             kcbk, kbap/2, kuap, kcap, kbbp/2, kubp, kcbp, Atot, Btot]
    jacMod = system1_Jac_noF(t,z,paramsMod)
    return(np.matmul(jacMod,z))
    

def system1_Jac_noF(t,z,params):
    a, b, k, p, ap, bp, AK, BK, ApP, BpP = z
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params

    j = np.array([np.array([-(k*kbak) - a**(-1 + n)*kba*n**2,0,-(a*kbak) + kua*n,0,0,0,kuak,0,kcap,0]),
                  np.array([0,-(k*kbbk) - b**(-1 + m)*kbb*m**2,-(b*kbbk),kub*m,0,0,0,kubk,0,kcbp]),
                  np.array([-(k*kbak) + a**(-1 + n)*kba*n,-(k*kbbk),-(a*kbak) - b*kbbk - kua,0,0,0,kcak + kuak,kcbk + kubk,0,0]),
                  np.array([0,b**(-1 + m)*kbb*m,0,-(ap*kbap) - bp*kbbp - kub,-(kbap*p),-(kbbp*p),0,0,kcap + kuap,kcbp + kubp]),
                  np.array([0,0,0,-(ap*kbap),-(kbap*(p)),0,kcak,0,kuap,0]),
                  np.array([0,0,0,-(bp*kbbp),0,-(kbbp*(p)),0,kcbk,0,kubp]),
                  np.array([k*kbak,0,a*kbak,0,0,0,-kcak - kuak,0,0,0]),
                  np.array([0,k*kbbk,b*kbbk,0,0,0,0,-kcbk - kubk,0,0]),
                  np.array([0,0,0,ap*kbap,kbap*p,0,0,0,-kcap - kuap,0]),
                  np.array([0,0,0,bp*kbbp,0,kbbp*p,0,0,0,-kcbp - kubp])])
    
    return(j)


def system1_Sol_noF(tSpan, params, initialConditions, toPlot, plotLog=False, atol=1e-12, rtol=1e-6):
    odeSol = scipy.integrate.solve_ivp(lambda tSpan, z: system1_ODEs_noF(tSpan, z, params),
                                        tSpan,initialConditions,method = 'Radau', vectorized=False,
                                        jac = lambda tSpan,z: system1_Jac_noF(tSpan,z,params),
                                        atol=atol, rtol=rtol)
    z = np.transpose(odeSol.y)
    t = odeSol.t
    
    if toPlot:
        system1_Plot_noF(t, z, params, initialConditions, plotLog=plotLog)
    return z,t



def system1_Plot_noF(t, z, params, initialConditions, plotLog=False):
    
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params
    lenTStart = int(len(z[:,0])/10)
    a = z[:,0]
    b = z[:,1]
    k = z[:,2]
    p = z[:,3]
    ap = z[:,4]
    bp = z[:,5]
    AK = z[:,6]
    BK = z[:,7]
    ApP = z[:,8]
    BpP = z[:,9]
    
    A_err = max(abs(Atot - (a + ap + n*k + (n + 1)*AK + ApP + n*BK)))/Atot
    B_err = max(abs(Btot - (b + bp + m*p + m*ApP + (m + 1)*BpP + BK)))/Btot
    if A_err+B_err > 1e-7:
        print('A_err = '+str(A_err))
        print('B_err = '+str(B_err))

    if (z<0).any():
        print('NEG_z')
    # plot results
    plt.plot(t[lenTStart:], a[lenTStart:], 'g', label='A')
    plt.plot(t[lenTStart:], b[lenTStart:], 'r', label='B')
    plt.plot(t[lenTStart:], ap[lenTStart:], 'orange', label='ap')
    plt.plot(t[lenTStart:], bp[lenTStart:], 'c', label='bp')
    plt.plot(t[lenTStart:],k[lenTStart:],'b',label='K')

    if not plotLog:
        plt.plot(t[lenTStart:],p[lenTStart:],'k',label='P')
    else:
        plt.semilogy(t[lenTStart:],p[lenTStart:],'k',label='P')

    plt.ylabel(r'Concentration ($\mu$M)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize = 12)
    #plt.ylim(top=Atot)
    plt.show()


def system1_Plot_forFig(t, z, params, initialConditions, plotLog=False):
    #tSec = copy.copy(t)
    tHr = t# / (60*60) # so it is measured in hours
    
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params
    lenTStart = int(len(z[:,0])/10)
    a = z[:,0]
    b = z[:,1]
    k = z[:,2]
    p = z[:,3]
    ap = z[:,4]
    bp = z[:,5]
    AK = z[:,6]
    BK = z[:,7]
    ApP = z[:,8]
    BpP = z[:,9]
    
    A_err = max(abs(Atot - (a + ap + n*k + (n + 1)*AK + ApP + n*BK)))/Atot
    B_err = max(abs(Btot - (b + bp + m*p + m*ApP + (m + 1)*BpP + BK)))/Btot
    if A_err+B_err > 1e-7:
        print('A_err = '+str(A_err))
        print('B_err = '+str(B_err))

    if (z<0).any():
        print('NEG_z')

    plt.plot(tHr[lenTStart:], ap[lenTStart:] + bp[lenTStart:], 'b', label=r'$A^p$')
    plt.ylabel(r'Total phosphorylation ($\mu$M)')
    plt.xlabel('Time (s)')
    if plotLog:
        plt.yscale('log')
#    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize = 24)
#    plt.ylim(bottom=min(min(bp[len(tHr)//2:]),min(ap[len(tHr)//2:])))
    plt.show()


# =============================================================================
# Solve a sample equation to check
# =============================================================================

tSpan = [0, 1]
n = 2
m = 2 
kba = 1
kbb = 1
kua = 0.01
kub = 1 
gammaAK = 1e-4
gammaBK = 1e-4
gammaAP = 1e-8
gammaBP = 1e-8
Atot = 100
Btot = 3000 

KMAK = 10000
KMAP = 10000
KMBK = 10000
KMBP = 10000

kcak = gammaAK * KMAK
kcbk = gammaBK * KMBK
kcap = gammaAP * KMAP
kcbp = gammaBP * KMBP

kbak = 10**2 # So that Michaelis-Menten is approx. satisfied, set this very large
kbbk = 10**2
kbap = 10**2
kbbp = 10**2

kuak = kbak * KMAK - kcak
kubk = kbbk * KMBK - kcbk
kuap = kbap * KMAP - kcap
kubp = kbbp * KMBP - kcbp

params = [n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot]
initialConditions = [Atot, Btot, 0, 0, 0, 0, 0, 0, 0, 0]
z,t = system1_Sol_noF(tSpan, params, initialConditions, toPlot=True, plotLog=True)


# =============================================================================
# Find oscillating parameters from random
# =============================================================================


def makeRandomParams_noF(numToSearch, maxParams, minParams, n, m, kd=False):
    paramsList = []
    
    if not kd:
        maxKba, maxKbb, maxFirstOrderParam, maxkck, maxkcp, maxKMK, maxKMP, maxkuk, maxkup, maxA, maxB = maxParams
        minKba, minKbb, minFirstOrderParam, minkck, minkcp, minKMK, minKMP, minkuk, minkup, minA, minB = minParams
    else:
        maxKba, maxKbb, maxKdParam, maxkck, maxkcp, maxKMK, maxKMP, maxkuk, maxkup, maxA, maxB = maxParams
        minKba, minKbb, minKdParam, minkck, minkcp, minKMK, minKMP, minkuk, minkup, minA, minB = minParams
    
    for i in range(numToSearch):        
        
        kba = np.exp(random.random() * (np.log(maxKba)-np.log(minKba)) + np.log(minKba))
        kbb = np.exp(random.random() * (np.log(maxKbb)-np.log(minKbb)) + np.log(minKbb))
        
        if not kd:
            kua = np.exp(random.random() * (np.log(maxFirstOrderParam)-np.log(minFirstOrderParam)) + np.log(minFirstOrderParam))
            kub = np.exp(random.random() * (np.log(maxFirstOrderParam)-np.log(minFirstOrderParam)) + np.log(minFirstOrderParam))
        else:
            kda = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
            kdb = np.exp(random.random() * (np.log(maxKdParam)-np.log(minKdParam)) + np.log(minKdParam))
            kua = kda * kba
            kub = kdb * kbb
                    
        KMAK = np.exp(random.random() * (np.log(maxKMK)-np.log(minKMK)) + np.log(minKMK))
        KMBK = KMAK 
        KMAP = np.exp(random.random() * (np.log(maxKMP)-np.log(minKMP)) + np.log(minKMP))
        KMBP = KMAP    
        
        kcak = np.exp(random.random() * (np.log(maxkck)-np.log(minkck)) + np.log(minkck))
        kcbk = kcak
        kcap = np.exp(random.random() * (np.log(maxkcp)-np.log(minkcp)) + np.log(minkcp))
        kcbp = kcap
        
        kuak = np.exp(random.random() * (np.log(maxkuk)-np.log(minkuk)) + np.log(minkuk))
        kubk = kuak
        kuap = np.exp(random.random() * (np.log(maxkup)-np.log(minkup)) + np.log(minkup))
        kubp = kuap
        
        kbak = (kuak + kcak) / KMAK
        kbbk = (kubk + kcbk) / KMBK
        kbap = (kuap + kcap) / KMAP
        kbbp = (kubp + kcbp) / KMBP

        Atot = np.exp(random.random() * (np.log(maxA)-np.log(minA)) + np.log(minA))
        Btot = np.exp(random.random() * (np.log(maxB)-np.log(minB)) + np.log(minB))
          
        params = [n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot]
        paramsList.append(params)
    
    return(paramsList)

[2, 2, 1.0, 0.10000000000000002, 0.010851903778645535, 0.42847794176203385, 0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009]
def findOscParameters_noF(tmaxMultiplier, paramsToSearch, printOutTime=10000):
    
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
        
        n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params
        initialConditions = [Atot, Btot, 0, 0, 0, 0, 0, 0, 0, 0]

        #maximum time we'll integrate to
        tmax = tmaxMultiplier/min([kua, kub, kba*Atot**(n-1), kbb*Btot**(m-1), kcak, kcap, kbak*Atot, kbbk*Btot, kuak, kuap]) 
        tSpan = [0, tmax]
        toPlot = False
        startPlot = time.time()
    
        z,t = system1_Sol_noF(tSpan, params, initialConditions, toPlot)
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
        
        KMAK = (kuak + kcak) / kbak
        KMAP = (kuap + kcap) / kbap
        gammaAK = kcak / KMAK
        gammaAP = kcap / KMAP
        gammaA = gammaAK / gammaAP
        gammaB = gammaA
        F = 0
        
        alpha3 = ((m * kua * kbb * gammaA**(m-1) * Btot**m) / 
                           (kba * Atot**(m-1) * gammaB**m * (m*kua + kub)))
        alpha1 = (kub/kbb)**(n+1) / (kua / kba)**m * (Atot / gammaA)**(n*m) * (gammaB / Btot)**(n*m + m) * F**(n+1-m)
        alpha4 = (m-1)*kub / ((n+1)* kua)

        oscillating = False
        if (nIP > 10 and 
            min(np.abs(xChange[5:-1] / xChange[5])) > 0.5 and 
            t[-3]/t[-2] > 0.95 and abs((abs(xChange[3*len(xChange)//4]) - abs(xChange[-2]))/xChange[-2]) < 1 and 
                not (z<-10000).any()): 

            period = prevCounterArray[-2] - prevCounterArray[-4]
            period2 = prevCounterArray[-4] - prevCounterArray[-6]
            
            if np.abs(period - period2) / period < 5e-2:  # periods must be consistent
                oscillating = True  
                
            else:  # perhaps the oscillations just haven't stabilized yet
                z,t = system1_Sol_noF([0, tmax + 10 * period], params, initialConditions, toPlot)
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
                        not (z<-10000).any()):  
                    
                    period = prevCounterArray[-2] - prevCounterArray[-4]
                    period2 = prevCounterArray[-4] - prevCounterArray[-6]
                    if np.abs(period - period2) / period < 5e-2:
                        oscillating = True
                
        if oscillating:
            print(params)
            print(initialConditions)
            print(tSpan)
            print(nIP)
            z,t = system1_Sol_noF([t[max(0, prevCounter - (prevCounter - prevPrevCounter) * 50)],t[prevCounter]], 
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
    
    return(oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods, 
           nonOscParams, nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s)


def combineOscParamsFromDifferentTMax_noF(tmaxMultipliers, paramsToStart):
    
    print('tmaxMultiplier = ' + str(tmaxMultipliers[0]))
    oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods, nonOscParams, \
            nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s = findOscParameters_noF(
                tmaxMultipliers[0], paramsToStart, printOutTime=100)

    # =============================================================================
    # Find which of the parameters we searched through we had misclassified as nonoscillating
    # We assume we didn't misclassify anything as oscillating, and that any misclassification
    # was just because we didn't integrate for enough time.
    # =============================================================================
    
    for tmaxMultiplier in tmaxMultipliers[1:]:
        print('tmaxMultiplier = ' + str(tmaxMultiplier))
        prevOscParams = copy.copy(oscParams)
        prevOscAlpha1s = copy.copy(oscAlpha1s)
        prevOscAlpha3s = copy.copy(oscAlpha3s)
        prevOscAlpha4s = copy.copy(oscAlpha4s)
        prevOscPeriods = copy.copy(oscPeriods)
        prevNonOscParams = copy.copy(nonOscParams)
        
        oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods, nonOscParams, \
                nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s = findOscParameters_noF(
                        tmaxMultiplier, 
                        prevNonOscParams, 
                        printOutTime=min(100, int(np.ceil(len(prevNonOscParams)/100))))
        
        oscParams = copy.copy(prevOscParams + oscParams)
        oscAlpha1s = copy.copy(prevOscAlpha1s + oscAlpha1s)
        oscAlpha3s = copy.copy(prevOscAlpha3s + oscAlpha3s)
        oscAlpha4s = copy.copy(prevOscAlpha4s + oscAlpha4s)
        oscPeriods = copy.copy(prevOscPeriods + oscPeriods)

    return(oscParams, oscAlpha1s, oscAlpha3s, oscAlpha4s, oscPeriods,
           nonOscParams, nonOscAlpha1s, nonOscAlpha3s, nonOscAlpha4s)
    

    
# =============================================================================
# Choose hyperparameters:
# =============================================================================
maxKba = 10**0
minKba = 10**-2
maxKbb = maxKba
minKbb = minKba
maxKdParam = 500
minKdParam = 10**-1
maxkck = 3.1 #from paper Identification of a Major Determinant for Serine-Threonine Kinase Phosphoacceptor Specificity
minkck = 3.1
maxkcp = 2006 #from paper Mutational Analysis of a Ser/Thr Phosphatase 
minkcp = 2006
maxKMK = 14 #from paper Identification of a Major Determinant for Serine-Threonine Kinase Phosphoacceptor Specificity
minKMK = 14
maxKMP = 10.3*1000 #from paper Mutational Analysis of a Ser/Thr Phosphatase 
minKMP = 10.3*1000
maxkuk = maxkck * 10
minkuk = minkck / 10
maxkup = maxkcp * 10
minkup = minkcp / 10
maxA = 10**2
minA = 10**-2
maxB = maxA
minB = minA

maxParams = [maxKba, maxKbb, maxKdParam, maxkck, maxkcp, maxKMK, maxKMP, maxkuk, maxkup, maxA, maxB]
minParams = [minKba, minKbb, minKdParam, minkck, minkcp, minKMK, minKMP, minkuk, minkup, minA, minB]


randomParamsn2m2 = makeRandomParams_noF(
        50000, maxParams, minParams, n=2, m=2, kd=True)
#
oscParamsn2m2, oscAlpha1sn2m2, oscAlpha3sn2m2, oscAlpha4sn2m2, oscPeriods, nonOscParamsn2m2, \
        nonOscAlpha1sn2m2, nonOscAlpha3sn2m2, nonOscAlpha4sn2m2 = combineOscParamsFromDifferentTMax_noF(
                [1, 100, 1000], randomParamsn2m2)



# =============================================================================
# Parameter sweeps to find oscillations
# =============================================================================
results = dict()

maxKba = 10**0
minKba = 10**-2
maxKbb = 10**0
minKbb = 10**-2
maxKdParam = 10**3
minKdParam = 10**-3
maxkck = 3.1 #Identification of a Major Determinant for Serine-Threonine Kinase Phosphoacceptor Specificity
minkck = 3.1
maxkcp = 2006 #Mutational Analysis of a Ser/Thr Phosphatase 
minkcp = 2006
maxKMK = 14 #Identification of a Major Determinant for Serine-Threonine Kinase Phosphoacceptor Specificity
minKMK = 14
maxKMP = 10.3*1000 #Mutational Analysis of a Ser/Thr Phosphatase 
minKMP = 10.3*1000

# Can play around with these
maxkuk = maxkck# * 10
minkuk = minkck# / 10
maxkup = maxkcp# * 10
minkup = minkcp# / 10

# Can play around with these:
maxA = 10
minA = 10
maxB = 10
minB = 10

kbaVec = [0.01, 0.1, 1] #[0.01, 0.03, 0.1]
kbbVec = [0.01, 0.1, 1]

for kba in kbaVec:
    for kbb in kbbVec:
        key = 'kba_' + str(kba) + '_kbb_' + str(kbb)
        print(key)
        
        maxParams = [kba, kbb, maxKdParam, maxkck, maxkcp, maxKMK, 
                     maxKMP, maxkuk, maxkup, maxA, maxB]
        minParams = [kba, kbb, minKdParam, minkck, minkcp, minKMK, 
                     minKMP, minkuk, minkup, minA, minB]
        
        randomParams = makeRandomParams_noF(  # first iteration of this had 500 random parameters
                500, maxParams, minParams, n=2, m=2, kd=True)
        
        oscParamsn2m2, oscAlpha1sn2m2, oscAlpha3sn2m2, oscAlpha4sn2m2, oscPeriodsn2m2, nonOscParamsn2m2, \
                nonOscAlpha1sn2m2, nonOscAlpha3sn2m2, nonOscAlpha4sn2m2 = \
                combineOscParamsFromDifferentTMax_noF([1, 100, 1000], randomParams)
        
        results[key] = [randomParams, oscParamsn2m2, oscAlpha1sn2m2, oscAlpha3sn2m2, \
               oscAlpha4sn2m2, oscPeriodsn2m2, nonOscParamsn2m2, nonOscAlpha1sn2m2, \
               nonOscAlpha3sn2m2, nonOscAlpha4sn2m2]
        
        save(results[key], 'sys1_manyDim_results/results_noF_1_30_20_' + key)

# Load the results we just made so we can just run from here on 
results = dict()
for kba in kbaVec:
    for kbb in kbbVec:
        key = 'kba_' + str(kba) + '_kbb_' + str(kbb)
        start = time.time()
        results[key] = load('sys1_manyDim_results/results_noF_1_30_20_' + key)
        print('key = ' + key + ' and time to load = ' + str(time.time() - start))

# results = load('results_sys_1_11_5.pickle')
fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=[10,9])
for i, kbb in enumerate(kbbVec[::-1]):
    for j, kba in enumerate(kbaVec):        
        key = 'kba_' + str(kba) + '_kbb_' + str(kbb)
        (randomParams,  # not in earlier versions of the code
         oscParamsn2m2, oscAlpha1sn2m2, oscAlpha3sn2m2, oscAlpha4sn2m2, 
         oscPeriodsn2m2,  # not in earlier versions of the code
         nonOscParamsn2m2, nonOscAlpha1sn2m2, nonOscAlpha3sn2m2, nonOscAlpha4sn2m2) = results[key]
        
        oscKdA = []
        nonOscKdA = []
        oscKdB = []
        nonOscKdB = []
        for params in oscParamsn2m2:
            n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, \
                    kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params
            oscKdA += [kua / kba]
            oscKdB += [kub / kbb]
        for params in nonOscParamsn2m2:
            n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, \
                    kuap, kcap, kbbp, kubp, kcbp, Atot, Btot = params
            nonOscKdA += [kua / kba]
            nonOscKdB += [kub / kbb]    
        
        kba = np.round(kba, 2)
        kbb = np.round(kbb, 2)
        if kba == 1:
            kba = 1  # not 1.0
        if kbb == 1:
            kbb = 1

        axs[i, j].plot(nonOscKdA, nonOscKdB, '.', alpha=0.1, color=[i/256 for i in [255, 194, 10]])
        axs[i, j].plot(oscKdA, oscKdB, '.', alpha=0.2, color=[i/256 for i in [12, 123, 220]])
        
        axs[i, j].set_xscale('log')
        axs[i, j].set_yscale('log')
        
        if i == 0 and j == len(kbaVec) - 1:
            axs[i, j].set_xlabel(r'$k_{d\kappa}$ ($\mu$M)')# + '\n' + str(gammaK), fontsize=24)
            axs[i, j].set_ylabel(#str(Atot) + '\n' + 
                    r'$k_{d\rho}$ ($\mu$M)')#, fontsize=24)
#            axs[i, j].set_xticks([10**-4, 10**0, 10**4])
#            axs[i, j].set_yticks([10**-4, 10**0, 10**4])
            
            plt.sca(axs[i, j])
            axs[i, j].yaxis.tick_right()
            axs[i, j].xaxis.tick_top()
#            plt.xticks([10**-4, 10**0, 10**4], fontsize=16)
#            plt.yticks([10**-4, 10**0, 10**4], fontsize=16)
            plt.xticks([10**-3, 10**0, 10**3], fontsize=16) #10**-1, 10**0, 10**1, 10**2
            plt.yticks([10**-3, 10**0, 10**3], fontsize=16) #10**-1, 10**0, 10**1, 10**2
#            axs[i, j].set_xticks([10**-1, 10**0, 10**1, 10**2, 500])

        else:
#            axs[i, j].set_xticks([])
#            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([10**-3, 10**0, 10**3]) #10**-1, 10**0, 10**1, 10**2
            axs[i, j].set_yticks([10**-3, 10**0, 10**3]) #10**-1, 10**0, 10**1, 10**2
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])

            if j == 0:
                axs[i, j].set_ylabel(str(kbb), fontsize=24)
            if i == len(kbbVec) - 1:
                axs[i, j].set_xlabel(str(kba), fontsize=24)
        axs[i, j].yaxis.set_ticks_position('both')
        axs[i, j].xaxis.set_ticks_position('both')
#        axs[i, j].set_xlim([10**-4, 10**-1])
#        axs[i, j].set_ylim([10**-3, 10**0])
fig.text(0.5, -0.02, r'$k_{b\kappa}$ $(\mu$M$^{-1}$s$^{-1})$', ha='center', fontsize=30)
fig.text(-0.04, 0.5, r'$k_{b\rho}$ $(\mu$M$^{-1}$s$^{-1})$', va='center', rotation='vertical', fontsize=30)
plt.tight_layout(h_pad=-0.15, w_pad=-0.15)
plt.show()



# =============================================================================
# Show some trajectories for some parameter sets
# =============================================================================
def phaseSpacePlot(params, t_to_integrate, t_index_to_show, markersize=36, atol=1e-12, rtol=1e-6):
    initialConditions = params[-3:-1] + [0] * 10 + [params[-1]]
    z,t = system1_Sol([0, t_to_integrate], params, initialConditions, 
                      toPlot=False, plotLog=False, atol=atol, rtol=rtol)
    
#    plt.figure(); plt.plot(t[t_index_to_show], z[t_index_to_show]); plt.yscale('log'); plt.show()
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot, Ftot = params
    print('kbk = ' + str(kba))
    print('kbp = ' + str(kbb))
    print('kdk = ' + str(kua / kba))
    print('kdp = ' + str(kub / kbb))
    
    
    plt.set_cmap('twilight')
    kSmall = z[t_index_to_show, 2]
    pSmall = z[t_index_to_show, 3]
    tSmall = t[t_index_to_show] - t[t_index_to_show[0]]
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(kSmall, pSmall, c=tSmall, s=markersize)
    plt.xlim(min(kSmall)*0.5, max(kSmall)*2)
    plt.ylim(min(pSmall)*0.5, max(pSmall)*2)
    plt.xlabel(r'K ($\mu$M)')
    plt.ylabel(r'P ($\mu$M)')
    plt.colorbar(label='Time (s)')
    #plt.savefig('sys1_phase_diag.png', format='png', dpi=500)
    plt.show()


phaseSpacePlot(
        [2, 2, 0.01, 1, 0.001, 3.6,
         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
         7000, range(450, 1560), markersize=16, rtol=1e-8)

phaseSpacePlot(
        [2, 2, 0.03, 0.01, 0.02, 0.1, 
         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
         7000, range(350, 582))

phaseSpacePlot(
        [2, 2, 0.03, 0.01, 0.032, 0.1, 
         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
         7000, range(100, 1300), markersize=16, rtol=1e-8)


#phaseSpacePlot(
#        [2, 2, 0.029999999999999995, 0.10000000000000002, 0.016401921452028297, 2.6910972267587976, 
#         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
#         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
#         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
#         5000, range(520, 605))
#
#phaseSpacePlot(
#        [2, 2, 0.01, 0.1, 0.016401921452028297, 2.6910972267587976, 
#         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
#         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
#         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
#         5000, range(620, 868))
#
#phaseSpacePlot(
#        [2, 2, 0.03, 0.01, 0.005, 0.1, 
#         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
#         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
#         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
#         3000, range(390, 655))
#
#phaseSpacePlot(
#        [2, 2, 0.01, 1, 0.008, 5.4,
#         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
#         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
#         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
#         1000, range(270, 1300), markersize=16, rtol=1e-8)