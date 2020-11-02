#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:19:23 2020

@author: Ofer
"""
# =============================================================================
# Loading data and making figures for oscillator paper
# =============================================================================

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import copy
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



# =============================================================================
# # ===========================================================================
# # # =========================================================================
# # # Bounded self-assembly: simplified (2D) system
# # # only considering parameter sets that satisfy certain assumptions (outlined in paper and given below)
# # # Python file: system_1_osc_2.py
# # # =========================================================================
# # ===========================================================================
# =============================================================================

## Assumptions: (ggConst = 5)
#(gammaA * kfp > ggConst * (pfp + F) and gammaB * kfp > ggConst * (pfp + F) and 
# Atot > ggConst * n * kfp and Btot > ggConst * m * pfp and expectedFP.success and
# kfp < Atot/n and pfp < Btot/m and kfp > 0 and pfp > 0)

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
        system1_Plot_for_Fig(t, z, params, initialConditions, plotLog=plotLog)
    return z,t

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
# Phase diagram 
# =============================================================================

params = [2,2,0.017008672411578697,  0.31511894635426035,  0.1476559558825341,  
          63.53111175046402,  1.1369463039453072,  1.1369463039453072,  
          58.32477960431114,  20.53440667917614,  4.769308906434382e-08] 
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
#plt.savefig('bounded_2d_phase_diagram_square.png', 
#            dpi=800, bbox_inches='tight')
plt.show()


# =============================================================================
# Oscillations vs. non-oscillations split
# =============================================================================

(oscParamsn2m2Exp, oscAlpha1sn2m2Exp, oscAlpha3sn2m2Exp, oscAlpha4sn2m2Exp, 
 oscPeriodsn2m2Exp, nonOscParamsn2m2Exp, nonOscAlpha1sn2m2Exp, nonOscAlpha3sn2m2Exp, 
 nonOscAlpha4sn2m2Exp) = load(
         'n2m2_for_fig_3.pickle')

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
#plt.savefig('osc_vs_nonosc_split_gamma_nu.png', 
#            dpi=800, bbox_inches='tight')
plt.show()


# =============================================================================
# Visualizing p^star in the osc/non-osc split
# =============================================================================
allParamsn2m2Exp = oscParamsn2m2Exp + nonOscParamsn2m2Exp

pstar_pred_array_n2m2 = []
pstar_pred_array_n2m2_nu = []
pstar_pred_array_n2m2_ptilde = []
for q in range(len(allParamsn2m2Exp)):
    params = allParamsn2m2Exp[q]
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
    expectedFP = scipy.optimize.root(lambda x: system1_ODEs(0, x, params), 
            [Atot/10, Btot/10], 
            method='hybr', 
            jac=lambda x: system1_Jac(0, x, params), 
            tol=1e-9, 
            callback=None, 
            options=None)
    kfp, pfp = expectedFP.x
    if (expectedFP.success and kfp < Atot/n and pfp < Btot/m and kfp > 0 and pfp > 0):
        pstar_pred_array_n2m2 += [pfp]
        pstar_pred_array_n2m2_nu += [(pfp/F) / (((n + 1) * kua + kub) / ((m-1) * kub - (n+1) * kua))]
        pstar_pred_array_n2m2_ptilde += [pfp / F]
    else:
        pstar_pred_array_n2m2 += [0]
        pstar_pred_array_n2m2_nu += [0]
        pstar_pred_array_n2m2_ptilde += [0]
pstar_pred_array_n2m2 = np.array(pstar_pred_array_n2m2)
pstar_pred_array_n2m2_nu = np.array(pstar_pred_array_n2m2_nu)
pstar_pred_array_n2m2_ptilde = np.array(pstar_pred_array_n2m2_ptilde)


fig, ax = plt.subplots()
plt.scatter(np.array(oscGammasn2m2Exp + nonOscGammasn2m2Exp), 
            np.array(oscAlpha4sn2m2Exp + nonOscAlpha4sn2m2Exp), 
            c=pstar_pred_array_n2m2_nu, alpha=0.2, s=5,
            norm=matplotlib.colors.SymLogNorm(linthresh=1e-1, linscale=1))
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
cbar = plt.colorbar()
cbar.set_ticks([-1e6, -1e3, -1e0, 0, 1e0, 1e3, 1e6])
cbar.set_label(r'Normalized $P^\star$')
#plt.savefig('normalized_pstar_gamma_nu.png', 
#            dpi=800, bbox_inches='tight')
plt.show()


# =============================================================================
# # ===========================================================================
# # # =========================================================================
# # # Bounded self-assembly: simplified (2D) system with no assumptions on parameter sets
# # # Python file: system_1_osc_2.py
# # # =========================================================================
# # ===========================================================================
# =============================================================================

(oscParamsn2m2NoApprox, oscAlpha1sn2m2NoApprox, oscAlpha3sn2m2NoApprox, oscAlpha4sn2m2NoApprox, 
 oscPeriodsn2m2NoApprox, nonOscParamsn2m2NoApprox, nonOscAlpha1sn2m2NoApprox, nonOscAlpha3sn2m2NoApprox, 
 nonOscAlpha4sn2m2NoApprox) = load(
         'n2m2_no_approx.pickle')

oscGammasn2m2NoApprox = [1/i for i in oscAlpha1sn2m2NoApprox]
nonOscGammasn2m2NoApprox = [1/i for i in nonOscAlpha1sn2m2NoApprox]

# =============================================================================
# # =============================================================================
# # osc frequency plots
# # =============================================================================
# =============================================================================

print('minimum timescale for oscillation = ' + str(min(oscPeriodsn2m2NoApprox)))
print('maximum timescale for oscillation = ' + str(max(oscPeriodsn2m2NoApprox)))

oscExpectedOmegaSq = []
for e, params in enumerate(oscParamsn2m2NoApprox):
    n, m, kba, kbb, kua, kub, gammaA, gammaB, Atot, Btot, F = params
    pstar = oscGammasn2m2NoApprox[e]**(1 / (n+1-m))
    pstarOverpstarPlus1 = 1  # pstar / (pstar + 1)
    omegaSq = n*m*kua*kub * pstarOverpstarPlus1 - (
            (n+1)*kua + (-1 + m*pstarOverpstarPlus1)*kub)**2 / 4
    
    oscExpectedOmegaSq += [copy.copy(omegaSq)]


plt.figure()
plt.plot(oscExpectedOmegaSq, [(2*np.pi/x)**2 for x in oscPeriodsn2m2NoApprox], '.', alpha=0.45,
                              color=[i/256 for i in [12, 123, 220]])
plt.plot(np.logspace(-9,np.log10(max(oscExpectedOmegaSq)),100000, base=10), 
         np.logspace(-9,np.log10(max(oscExpectedOmegaSq)),100000, base=10), '--k')
plt.xscale('symlog', linthreshx=1e-8)
plt.xticks([-10**4, -10**-4, 0, 10**-4, 10**4])
plt.yscale('log')
plt.yticks([10**-8, 10**-4, 10**0, 10**4])
plt.xlabel(r'$\omega^2_{pred}$ (s$^{-2}$)')
plt.ylabel(r'$\omega^2$ (s$^{-2}$)')
#plt.savefig('wpred_sq_no_approx.png', 
#            dpi=800, bbox_inches='tight')
plt.show()



# =============================================================================
# Predict frequency outside linear regime?
# =============================================================================
oscExpectedOmegaSqArr = np.array(oscExpectedOmegaSq)
oscParamsn2m2ExpOutsideLinear = np.array(oscParamsn2m2NoApprox)[oscExpectedOmegaSqArr<0]
oscPeriodsn2m2ExpOutsideLinear = np.array(oscPeriodsn2m2NoApprox)[oscExpectedOmegaSqArr<0]
oscExpectedOmegaSqArrOutsideLinear = oscExpectedOmegaSqArr[oscExpectedOmegaSqArr<0]


plt.figure()
plt.plot([x[4]**1 for x in oscParamsn2m2NoApprox],
         #[-x for x in oscExpectedOmegaSqArrOutsideLinear], 
         [(2*np.pi/x)**1 for x in oscPeriodsn2m2NoApprox], 
         '.', alpha=0.2, color=[i/256 for i in [12, 123, 220]])
plt.plot(np.logspace(-5, 2, 100, base=10), 
         np.logspace(-5, 2, 100, base=10), '--k')
plt.xscale('log')
plt.xticks([10**-4, 10**-2, 10**0, 10**2])
plt.yscale('log')
plt.yticks([10**-4, 10**-2, 10**0, 10**2])
plt.xlabel(r'$k_{u\kappa}$ (s$^{-1}$)')
plt.ylabel(r'$\omega$ (s$^{-1}$)')
#plt.title('Predicting oscillation frequency')
#plt.savefig('kuk_predicts_freq_no_approx.png', 
#            dpi=800, bbox_inches='tight')
plt.show()


relative_squared_error_in_predicting_freq_by_kua = (
        np.array([(2*np.pi/x)**2 for x in oscPeriodsn2m2NoApprox]) - np.array([x[4]**2 for x in oscParamsn2m2NoApprox]))**2 / np.array(
                [(2*np.pi/x)**2 for x in oscPeriodsn2m2NoApprox])


print('Root Mean Squared Relative Error for kua = ' + str(
        np.sqrt(np.mean(relative_squared_error_in_predicting_freq_by_kua))))

from sklearn.metrics import r2_score, mean_squared_error

print('R^2 value for kua predicting freq is ' + str(r2_score(
        [(2*np.pi/x) for x in oscPeriodsn2m2NoApprox], [x[4] for x in oscParamsn2m2NoApprox])))

print('R^2 value in log-space for kua predicting freq is ' + str(r2_score(
        np.log([(2*np.pi/x) for x in oscPeriodsn2m2NoApprox]), np.log([x[4] for x in oscParamsn2m2NoApprox]))))





# =============================================================================
# =============================================================================
# =============================================================================
# # # Bounded self-assembly for unsimplified (many-D) system
# # # Python file: system_1_osc_fuller_for_figs.py
# =============================================================================
# =============================================================================
# =============================================================================

kbaVec = [0.01, 0.1, 1] #[0.01, 0.03, 0.1]
kbbVec = [0.01, 0.1, 1]
FVec = [10**-4]

#results = load('results_sys_1_11_5.pickle')

results = dict()
for F in FVec:
    for kba in kbaVec:
        for kbb in kbbVec:
            key = 'kba_' + str(kba) + '_kbb_' + str(kbb) + '_F_' + str(F)
            start = time.time()
            results[key] = load('sys1_manyDim_results/results_1_30_20_' + key)
            print('key = ' + key + ' and time to load = ' + str(time.time() - start))


F = 10**-4
fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=[10,9])
for i, kbb in enumerate(kbbVec[::-1]):
    for j, kba in enumerate(kbaVec):        
        key = 'kba_' + str(kba) + '_kbb_' + str(kbb) + '_F_' + str(F)
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
                    kuap, kcap, kbbp, kubp, kcbp, Atot, Btot, Ftot = params
            oscKdA += [kua / kba]
            oscKdB += [kub / kbb]
        for params in nonOscParamsn2m2:
            n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, \
                    kuap, kcap, kbbp, kubp, kcbp, Atot, Btot, Ftot = params
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
#plt.savefig('bounded_full_equations_osc_nonosc_split.png', 
#            dpi=900, bbox_inches='tight')
plt.show()

# =============================================================================
# Show some sample trajectories
# =============================================================================

def system1_ODEsFull(t,z,params):
    a, b, k, p, ap, bp, AK, BK, ApP, BpP, ApF, BpF, f = z
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot, Ftot = params
        
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
             kcbk, kbap/2, kuap, kcap, kbbp/2, kubp, kcbp, Atot, Btot, Ftot]
    jacMod = system1_JacFull(t,z,paramsMod)
    return(np.matmul(jacMod,z))
    

def system1_JacFull(t,z,params):
    a, b, k, p, ap, bp, AK, BK, ApP, BpP, ApF, BpF, f = z
    n, m, kba, kbb, kua, kub, kbak, kuak, kcak, kbbk, kubk, kcbk, kbap, kuap, kcap, kbbp, kubp, kcbp, Atot, Btot, Ftot = params

    j = np.array([np.array([-(k*kbak) - a**(-1 + n)*kba*n**2,0,-(a*kbak) + kua*n,0,0,0,kuak,0,kcap,0,kcap,0,0]),
                  np.array([0,-(k*kbbk) - b**(-1 + m)*kbb*m**2,-(b*kbbk),kub*m,0,0,0,kubk,0,kcbp,0,kcbp,0]),
                  np.array([-(k*kbak) + a**(-1 + n)*kba*n,-(k*kbbk),-(a*kbak) - b*kbbk - kua,0,0,0,kcak + kuak,kcbk + kubk,0,0,0,0,0]),
                  np.array([0,b**(-1 + m)*kbb*m,0,-(ap*kbap) - bp*kbbp - kub,-(kbap*p),-(kbbp*p),0,0,kcap + kuap,kcbp + kubp,0,0,0]),
                  np.array([0,0,0,-(ap*kbap),-(kbap*(f + p)),0,kcak,0,kuap,0,kuap,0,-(ap*kbap)]),
                  np.array([0,0,0,-(bp*kbbp),0,-(kbbp*(f + p)),0,kcbk,0,kubp,0,kubp,-(bp*kbbp)]),
                  np.array([k*kbak,0,a*kbak,0,0,0,-kcak - kuak,0,0,0,0,0,0]),
                  np.array([0,k*kbbk,b*kbbk,0,0,0,0,-kcbk - kubk,0,0,0,0,0]),
                  np.array([0,0,0,ap*kbap,kbap*p,0,0,0,-kcap - kuap,0,0,0,0]),
                  np.array([0,0,0,bp*kbbp,0,kbbp*p,0,0,0,-kcbp - kubp,0,0,0]),
                  np.array([0,0,0,0,f*kbap,0,0,0,0,0,-kcap - kuap,0,ap*kbap]),
                  np.array([0,0,0,0,0,f*kbbp,0,0,0,0,0,-kcbp - kubp,bp*kbbp]),
                  np.array([0,0,0,0,-(f*kbap),-(f*kbbp),0,0,0,0,kcap + kuap,kcbp + kubp,-(ap*kbap) - bp*kbbp])])
    
    return(j)


def system1_SolFull(tSpan, params, initialConditions, toPlot, plotLog=False, atol=1e-12, rtol=1e-6):
    odeSol = scipy.integrate.solve_ivp(lambda tSpan, z: system1_ODEsFull(tSpan, z, params),
                                        tSpan,initialConditions,method = 'Radau', vectorized=False,
                                        jac = lambda tSpan,z: system1_JacFull(tSpan,z,params),
                                        atol=atol, rtol=rtol)
    z = np.transpose(odeSol.y)
    t = odeSol.t
    
    return z,t

def phaseSpacePlotSys1Full(params, t_to_integrate, t_index_to_show, markersize=36, 
                           atol=1e-12, rtol=1e-6, savePlot=''):
    initialConditions = params[-3:-1] + [0] * 10 + [params[-1]]
    z,t = system1_SolFull([0, t_to_integrate], params, initialConditions, 
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
#    if savePlot:
#        plt.savefig('savePlot + '.png', 
#            dpi=800, bbox_inches='tight')
    plt.show()


phaseSpacePlotSys1Full(
        [2, 2, 0.01, 1, 0.001, 3.6,
         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
         7000, range(450, 1560), markersize=16, rtol=1e-8, savePlot='bounded_manyd_phase_diagram_square')

#phaseSpacePlotSys1Full(
#        [2, 2, 0.03, 0.01, 0.02, 0.1, 
#         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
#         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
#         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
#         7000, range(350, 582))
##
#phaseSpacePlotSys1Full(
#        [2, 2, 0.03, 0.01, 0.032, 0.1, 
#         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
#         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
#         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
#         7000, range(100, 1300), markersize=16, rtol=1e-8)


phaseSpacePlotSys1Full(
        [2, 2, 0.1, 0.01, 0.02, 0.1, 
         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
         7000, range(960, 1490), rtol=1e-8, savePlot='bounded_manyd_phase_diagram_curve')

phaseSpacePlotSys1Full(
        [2, 2, 0.1, 0.01, 0.058, 0.1, 
         0.442857142857143, 3.1, 3.1, 0.442857142857143, 3.1, 3.1, 0.3895145631067961, 
         2006.000000000001, 2006.000000000001, 0.3895145631067961, 2006.000000000001, 
         2006.000000000001, 10.000000000000002, 10.000000000000002, 0.00010000000000000009], 
         7000, range(650, 1650), markersize=12, rtol=1e-8, savePlot='bounded_manyd_phase_diagram_spiral')


# =============================================================================
# =============================================================================
# =============================================================================
# # # Unbounded oscillator
# # # Python file: system_2_osc_for_figs.py
# =============================================================================
# =============================================================================
# =============================================================================

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
        system2_Plot_for_fig(t, z, params, initialConditions, plotLog=plotLog)
    return z,t

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
# Where are the oscillations?
# =============================================================================

results = load('results_sys_2_11_7.pickle')

AtotVec = [3, 10, 30]  #Atot = 1 doesn't give any oscillations
gammaK = (3.1/14) / (2006/10300)
FVec = [10**-5, 10**-4, 10**-3] #Ftot = 10**-2 doesn't give any oscillations
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
fig.text(0.5, -0.02, r'$\tilde{P}/\rho_{tot}$', ha='center', fontsize=30)
fig.text(-0.02, 0.5, r'$\kappa_{tot}/\rho_{tot}$', va='center', rotation='vertical', fontsize=30)
plt.tight_layout(h_pad=0, w_pad=0.15)
#plt.savefig('unbounded_osc_nonosc_split.png', 
#    dpi=900, bbox_inches='tight')
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
plt.axis('off')
#plt.savefig('osc_nonosc_split_legend.png', 
#    dpi=800, bbox_inches='tight')
plt.show()




# =============================================================================
# Show some trajectories for some parameter sets
# =============================================================================

def phaseSpacePlotSys2(params, t_to_integrate, t_index_to_show, markersize=36, 
                       atol=1e-12, rtol=1e-6, savePlot=''):
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
#    if savePlot:
#        plt.savefig(savePlot + '.png', 
#                    dpi=800, bbox_inches='tight')
    plt.show()


phaseSpacePlotSys2(
        [0.01, 0.05, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 30, 1, 1e-5], 
         14000, range(1016, 1900), markersize=16, rtol=1e-8, savePlot='unbounded_phase_diagram_1')

phaseSpacePlotSys2(
        [0.05, 0.05, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 30, 1, 1e-5], 
         7000, range(750, 1120), markersize=16, rtol=1e-8, savePlot='unbounded_phase_diagram_2')

phaseSpacePlotSys2(
        [0.03, 0.087, 3.1/14, 3.1/14, 2006 / 10300, 2006 / 10300, 10, 1, 1e-4], 
         7000, range(760, 1370), markersize=16, rtol=1e-8, savePlot='unbounded_phase_diagram_3')

# =============================================================================
# =============================================================================
# # # =========================================================================
# # # Periods of oscillation: comparing bounded (with no assumptions) and unbounded self-assembly
# # # =========================================================================
# =============================================================================
# =============================================================================

(oscParamsAll, oscMaxFAll, oscKdAAll, oscKdBAll, oscPeriodsAll, nonOscParamsAll, 
 nonOscMaxFAll, nonOscKdAAll, nonOscKdBAll) = load(
         'sys2_no_approx_500000.pickle')
# element number 631 isn't a real oscillation
# oscParamsAll.pop(631); oscMaxFAll.pop(631); oscKdAAll.pop(631); oscKdBAll.pop(631); oscPeriodsAll.pop(631)


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
#plt.savefig('bound_and_unbound_osc_period_histogram.png', 
#            dpi=800, bbox_inches='tight')
plt.show()



# =============================================================================
# Can we predict the frequency of the unbounded oscillator?
# =============================================================================


plt.figure()
plt.plot([x[0] for x in oscParamsAll],  # for sys2, params are: kdA, kdB, gammaAK, gammaBK, gammaAP, gammaBP, Atot, Btot, F
         [(2*np.pi/x) for x in oscPeriodsAll], 
         '.', alpha=0.25, color=[i/256 for i in [12, 123, 220]])
plt.plot(np.logspace(-3, 1, 100, base=10), 
         np.logspace(-3, 1, 100, base=10), '--k')
plt.xscale('log')
#plt.xticks([10**-3, 10**-1, 10**1])
plt.yscale('log')
#plt.yticks([10**-3, 10**-1, 10**1])
plt.xlabel(r'$k_{d\kappa}$ ($\mu M$)')
plt.ylabel(r'$\omega$ (s$^{-1}$)')
#plt.title('Predicting oscillation frequency')
plt.show()

from sklearn import linear_model

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(np.array([x[0] for x in oscParamsAll]).reshape(-1,1), np.array([(2*np.pi/x) for x in oscPeriodsAll]).reshape(-1,1))

# Make predictions using the testing set
y_pred = regr.predict(np.array([(2*np.pi/x) for x in oscPeriodsAll]).reshape(-1,1))

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error([(2*np.pi/x) for x in oscPeriodsAll], y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score([(2*np.pi/x) for x in oscPeriodsAll], y_pred))
print('Coefficient of determination in log-space: %.2f'
      % r2_score(np.log([(2*np.pi/x) for x in oscPeriodsAll]), np.log(y_pred)))
