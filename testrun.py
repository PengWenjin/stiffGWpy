#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PengWenjin/stiffGWpy/blob/main/testrun.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:

import os, sys, time, yaml
import multiprocessing as mp
# 设置启动方式为 'spawn'，适用于 Windows
# 如果不打算使用多进程，可以注释掉下面这行
# mp.set_start_method('spawn')

from pathlib import Path
from importlib import reload
import numpy as np
import pandas as pd
import math
from scipy import interpolate, integrate, special
from scipy.integrate import odeint, solve_ivp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# In[ ]:

# 引入你的项目模块
import global_param; reload(global_param)
from global_param import *
import functions; reload(functions)
from functions import int_FD, solve_SGWB

import LCDM_stiff_Neff; reload(LCDM_stiff_Neff)
from LCDM_stiff_Neff import LCDM_SN as sn

import stiff_SGWB; reload(stiff_SGWB)
from stiff_SGWB import LCDM_SG as sg

# In[ ]:

if __name__ == '__main__':
    # 1. A single model can be instantiated by a yaml file
    model1 = sg('base_param.yml')

    # 输出模型1的相关参数
    print(model1.cosmo_param, model1.derived_param['nt'], model1.derived_param['N_re'], model1.SGWB_converge, model1.DN_eff_orig)

    # In[ ]:

    # 2. A single model can be instantiated by explicitly inputting parameter values.
    #    For any unspecified parameter at initialization, its default value is taken.
    model2 = sg(r = 1e-2, T_re = 1e11)

    # 输出模型2的相关参数
    print(model2.cosmo_param)
    print(model2.derived_param, model2.SGWB_converge)

    # In[ ]:

    # 计算 SGWB
    t1 = time.time()
    model1.SGWB_iter()
    t2 = time.time()

    # 输出计算时间、收敛状态、相关参数
    print(t2-t1, model1.SGWB_converge, model1.cosmo_param['DN_eff'], model1.DN_eff_orig)
    print(model1.derived_param['kappa_s'], model1.kappa_r, model1.f[0], len(model1.f))

    # In[ ]:

    t1 = time.time()
    model2.SGWB_iter()
    t2 = time.time()

    # 输出计算结果
    print(model2.derived_param['kappa_s'], model2.kappa_r)
    print(t2-t1, model2.hubble[-1], model2.cosmo_param['DN_re'],
          model2.derived_param['N_inf']==model2.derived_param['N_re'] + model2.cosmo_param['DN_re'])
    print(model2.DN_gw[-1], model2.cosmo_param['DN_eff'], model2.SGWB_converge, len(model2.f))

    # In[ ]:

    # 3. A single model can be instantiated by passing a dictionary of parameters
    model3 = sg(model1.cosmo_param)

    # 输出模型3的相关参数
    model3.cosmo_param['DN_eff'] = 0
    model3.cosmo_param['kappa10'] = 1e2
    model3.cosmo_param['T_re'] = 1e1
    model3.cosmo_param['H0'] = 67.5
    model3.cosmo_param['cr'] = 1

    print(model1.cosmo_param)
    print(model3.cosmo_param)
    print(model1.derived_param['nt'], model3.derived_param['nt'])
    print(model1.derived_param['N_re'], model1.derived_param['N_inf'], model3.derived_param['N_re'], model3.derived_param['N_inf'])
    print(model3.SGWB_converge)

    # In[ ]:

    t1 = time.time()
    model3.SGWB_iter()
    t2 = time.time()

    # 输出模型3计算时间和结果
    print(model3.derived_param['kappa_s'], model3.kappa_r)
    print(t2-t1, model3.hubble[-1], model3.DN_gw[-1], model3.cosmo_param['DN_eff'], model3.SGWB_converge, len(model3.f))

    # In[ ]:

    # 重置模型
    model3.reset()
    print(model3.SGWB_converge, model3.DN_eff_orig, model3.cosmo_param['DN_eff'])

    model3.cosmo_param['r'] = 1e-22
    model3.cosmo_param['n_t'] = 4
    model3.cosmo_param['cr'] = 0
    model3.cosmo_param['T_re'] = 1e-2
    model3.cosmo_param['DN_re'] = 30
    model3.cosmo_param['kappa10'] = 0e-10
    print(model3.derived_param)

    # In[ ]:

    t1 = time.time()
    model3.SGWB_iter()
    t2 = time.time()

    # 输出重置后计算结果
    print(t2-t1, model3.hubble[-1], model3.DN_gw[-1], model3.cosmo_param['DN_eff'], model3.SGWB_converge, len(model3.f), model3.kappa_r)

    # In[ ]:

    # 一致性检查
    print(model1.DN_eff_orig, model2.DN_eff_orig, model3.DN_eff_orig, model3.DN_eff_orig + model3.DN_gw[-1] == model3.cosmo_param['DN_eff'])

    # In[ ]:

    # 代表性模型实例
    model4 = sg(r=3.9585109e-05, n_t=1.0116972, cr=0, T_re=0.17453859, DN_re=39.366618, kappa10=110.42477)

    t1 = time.time()
    model4.SGWB_iter()
    t2 = time.time()

    print(model4.derived_param['kappa_s'], model4.kappa_r)
    print(t2-t1, len(model4.f), model4.f[0], model4.hubble[-1], model4.cosmo_param['DN_re'])
    print(model4.DN_gw[-1], model4.cosmo_param['DN_eff'], model4.SGWB_converge)

    # In[ ]:

    # 绘制图形
    colors = plt.get_cmap('rainbow')(np.arange(0,1.1,.1))
    lc = plt.get_cmap('Paired')
    shade = plt.get_cmap('Set3')

    model = model4

    fig1, ((ax1, ax2), (ax3, ax4),) = plt.subplots(2, 2,
                                                  gridspec_kw={
                                                      'width_ratios': [2, 3],
                                                      'height_ratios': [1, 1]},
                                                  figsize=(20,12))
    Ntot = model.Nv[-1]
    #ind = len(model.f[model.f>model.f[0]-3])
    ind = len(model.f[model.Tensor_power(model.f)>=1])
    cond1 = model.N_hc[ind]>=model.Nv[model.find_index_hc(model.f[ind]-3)]
    #cond2 = model.N_hc[ind+20]>=model.Nv[model.find_index_hc(model.f[ind+20]-3)]

    ax1.plot(model.N_hc[0]-Ntot, model.Th[0], 'r:',
         model.N_hc[30]-Ntot, model.Th[30], 'y:',
         model.N_hc[60]-Ntot, model.Th[60], 'g:',
         model.N_hc[90]-Ntot, model.Th[90], 'c:',
         model.N_hc[120]-Ntot, model.Th[120], 'b:',
         model.N_hc[150]-Ntot, model.Th[150], 'm:',
         model.N_hc[180]-Ntot, model.Th[180], 'k:',
        )
    ax1.axhline(0, ls='--', c='k', lw=.5)
    #ax1.axvline(model.N[model.find_index_hc(model.f[ind]-3)], ls='--', c='k', lw=.5)
    #ax1.set_xlim(26.5,28)
    #ax1.set_ylim(500,2e3)
    #ax1.text(model.Nv[model.find_index_hc(model.f[80]-3)]+1,.9, '$kc\,/\,aH \geq 10^3$', fontsize=15)
    ax1.set_xlabel('Number of $e$-folds', fontsize=14)
    ax1.set_ylabel('$h(k,N)\,/\,h(k,0)$', fontsize=14)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))

    ax2.plot(#model.f[model.f<=model.f[0]-3], model.log10OmegaGW[model.f<=model.f[0]-3], 'r*',
         model.f, model.log10OmegaGW, 'r*',
         #model.f, [model.Ogw[i][-1]-model.Oj[i][-1] for i in range(len(model.f))], 'r*',
         #[model.f[i] for i in range(len(model.f)) if (Nslice:=14.8) in model.N_hc[i][:]], \
         #[model.Ogw[i][model.N_hc[i][:]==Nslice][0]-model.Oj[i][model.N_hc[i][:]==Nslice][0] \
         #for i in range(len(model.f)) if Nslice in model.N_hc[i][:]], 'r*',
         #model.f, np.log10(model.Tensor_power(model.f)),
         #model.f_grid, model.log10OmegaGW_grid, 'k*',
        )
    #ax2.axhline(0, ls='--', c='k', lw=.5)
    ax2.axvspan(np.log10(20), np.log10(1726), alpha=0.15, color='black')
    ax2.axvline(np.log10(25), ls='--', c='k', lw=.5)
    ax2.axvspan(-9, np.log10(1/yr), alpha=0.1, color='black')
    ax2.axvline(model.f_re, ls='-.', c='k', lw=1)
    ax2.axvline(model.f[0], ls='--', c='k', lw=.5)
    ax2.text(model.f[0]+.5,.9, '$f_\mathrm{max}$', transform=ax2.get_xaxis_transform(), fontsize=20)
    ax2.text(np.log10(25),.9, 'LIGO', transform=ax2.get_xaxis_transform(), fontsize=25)
    ax2.text(-9,.9, 'PTA', transform=ax2.get_xaxis_transform(), fontsize=25)
    #ax2.set_xlim(-6.7,-6.3)
    #ax2.set_ylim(-10,-5)
    ax2.set_xlabel('$\log_{10}(\,f\,/\mathrm{Hz})$', fontsize=14)
    ax2.set_ylabel('$\Omega_\mathrm{GW}\,(f)$', fontsize=14)
    #ax2.set_ylabel('$P_\mathrm{T}\,(f)$', fontsize=14)
    ax2.xaxis.set_minor_locator(MultipleLocator(.5))

    ax3.plot(model.N, model.DN_gw, 'b-',
        )
    #ax3.axvline(model.N[len(model.g2[model.g2<=0])-1], ls='--', c='k')
    ax3.axvline(-model.derived_param['N_re'], ls='--', c='k', lw=.5)
    ax3.text(-model.derived_param['N_re']-6,.9, '$T_\mathrm{re}$', transform=ax3.get_xaxis_transform(), fontsize=20)
    ax3.axvline(-N_i, ls='--', c='k')
    ax3.text(-N_i+1,.1, '$T_i=27e9~\mathrm{K}$\nfrom AlterBBN', transform=ax3.get_xaxis_transform(), fontsize=15)
    ax3.set_xlabel('Number of $e$-folds', fontsize=14)
    ax3.set_ylabel('$\Delta N_\mathrm{eff}^\mathrm{GW}\equiv$' +
               '$\\rho_\mathrm{GW}\,a^4\,/\,\\left(\\rho_{\gamma,0}\cdot\\frac{7}{8}(\\frac{4}{11})^{4/3}\\right)$', fontsize=14)

    ax4.plot(model.N, model.sigma, 'm-', lw=3)
    ax4.axhline(4/3, ls='--', c='k')
    ax4.axhline(1, ls='--', c='k')
    ax4.set_xlabel('Number of $e$-folds', fontsize=14)
    ax4.set_ylabel('$1+w$', fontsize=14)

    fig1.tight_layout()
    plt.show()



# In[ ]:


    freqs_N15 = np.load('cobaya/likelihoods/PTA/NANOGrav15yr/freqs.npy')
    T_N15 = 1/freqs_N15[0]
    freqs_N15 = np.log10(freqs_N15[:14])

    #NG15 = np.loadtxt('cobaya/likelihoods/PTA/NANOGrav15yr/sample.txt')
    #NG15_mock = np.loadtxt('cobaya/likelihoods/PTA/NANOGrav15yr/sample_mock.txt')

    #T_N15/yr, freqs_N15, NG15.shape
    print(T_N15/yr, freqs_N15) #, NG15.shape)


# In[ ]:
"""

    hc_NG15 = NG15 + np.log10(T_N15*12*np.pi**2)/2 + 1.5*freqs_N15
    Ogw_NG15 = 2*NG15 + np.log10(T_N15*8) + 4*np.log10(np.pi) + 5*freqs_N15 - 2*np.log10(cosmo.H0.to(u.Hz).value)

    hcmock_NG15 = NG15_mock + np.log10(T_N15*12*np.pi**2)/2 + 1.5*freqs_N15
    Ogwmock_NG15 = 2*NG15_mock + np.log10(T_N15*8) + 4*np.log10(np.pi) + 5*freqs_N15 - 2*np.log10(cosmo.H0.to(u.Hz).value)


# In[ ]:


    EPTA_freqsnew = np.loadtxt('cobaya/likelihoods/PTA/EPTAdr2/freqs_dr2new.txt')
    T_EPTA = 1/EPTA_freqsnew[0]
    EPTA_freqsnew = np.log10(EPTA_freqsnew[:9])

    EPTAnew = np.loadtxt('cobaya/likelihoods/PTA/EPTAdr2/EPTA_dr2new.dat')
    EPTAnew_mock = np.loadtxt('cobaya/likelihoods/PTA/EPTAdr2/EPTA_dr2new_mock.dat')

    #EPTA_freqsnew, T_EPTA/yr, EPTAnew.shape
    print(EPTA_freqsnew, T_EPTA/yr, EPTAnew.shape)


# In[ ]:


    hc_EPTA = EPTAnew + np.log10(T_EPTA*12*np.pi**2)/2 + 1.5*EPTA_freqsnew
    Ogw_EPTA = 2*EPTAnew + np.log10(T_EPTA*8) + 4*np.log10(np.pi) + 5*EPTA_freqsnew - 2*np.log10(cosmo.H0.to(u.Hz).value)

    hcmock_EPTA = EPTAnew_mock + np.log10(T_EPTA*12*np.pi**2)/2 + 1.5*EPTA_freqsnew
    Ogwmock_EPTA = 2*EPTAnew_mock + np.log10(T_EPTA*8) + 4*np.log10(np.pi) + 5*EPTA_freqsnew - 2*np.log10(cosmo.H0.to(u.Hz).value)

"""