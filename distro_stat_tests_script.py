# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 15:31:07 2025

@author: mayc06
"""

from extract_trajectories_from_orcoflashStupski import *
from scipy.stats import kstest
import pycircstat2
from pycircstat2.hypothesis import circ_anova

ann65_train = np.load('ANN65_trainperformance_absval_errorpeaks.npy')

ann66_train = np.load('ANN66_trainperformance_absval_errorpeaks.npy')

ann67_train = np.load('ANN67_trainperformance_absval_errorpeaks.npy')

# pval65vs67, k65vs67, eff65vs67 = kuiper((4+ann65_train)*np.pi/4,(4+ann67_train)*np.pi/4,res=8,axis=0)

# pval66vs67, k66vs67, eff66vs67 = kuiper((4+ann66_train)*np.pi/4,(4+ann67_train)*np.pi/4,res=8,axis=0)

# pval65vs66, k65vs66, eff65vs66 = kuiper((4+ann65_train)*np.pi/4,(4+ann66_train)*np.pi/4,res=8,axis=0)

# pval65vs67, pval66vs67, pval65vs66


# kspval65vs67 = kstest(ann65_train[:500],ann67_train[:500],nan_policy='omit')
# kspval65vs66 = kstest(ann65_train[:500],ann66_train[:500],nan_policy='omit')
# kspval65vs67,kspval65vs66

ps65vs67 = []
effs65vs67 = []
for i in range(10000):
    ann65_train_sub = np.random.choice(ann65_train,size=501,replace=False)
    ann67_train_sub = np.random.choice(ann67_train,size=501,replace=False)
    p,k,e = kuiper((4+ann65_train_sub)*np.pi/4,(4+ann67_train_sub)*np.pi/4,res=8,axis=0)
    ps65vs67.append(p)
    effs65vs67.append(e)
    
ps66vs67 = []
effs66vs67 = []
for i in range(10000):
    ann66_train_sub = np.random.choice(ann66_train,size=501,replace=False)
    ann67_train_sub = np.random.choice(ann67_train,size=501,replace=False)
    p,k,e = kuiper((4+ann66_train_sub)*np.pi/4,(4+ann67_train_sub)*np.pi/4,res=8,axis=0)
    ps66vs67.append(p)
    effs66vs67.append(e)    

    
ps65vs66 = []
effs65vs66 = []
for i in range(10000):
    ann65_train_sub = np.random.choice(ann65_train,size=501,replace=False)
    ann66_train_sub = np.random.choice(ann66_train,size=501,replace=False)
    p,k,e = kuiper((4+ann65_train_sub)*np.pi/4,(4+ann66_train_sub)*np.pi/4,res=8,axis=0)
    ps65vs66.append(p)
    effs65vs66.append(e)
    
np.mean(ps65vs66), np.mean(ps65vs67), np.mean(ps66vs67)
    

ann67_test = np.load('ANN67_n5000randperformance_errorpeaks.npy')

# pval67trainvstest, k67trainvstest,effect67trainvstest = kuiper((4+ann67_train)*np.pi/4,(4+ann67_test)*np.pi/4,res=8,axis=0)

ps67trainvstest = []
effs67trainvstest = []
for i in range(10000):
    ann67_train_sub = np.random.choice(ann67_train,size=501,replace=False)
    ann67_test_sub = np.random.choice(ann67_test,size=501,replace=False)
    p,k,e = kuiper((4+ann67_train_sub)*np.pi/4,(4+ann67_test_sub)*np.pi/4,res=8,axis=0)
    ps67trainvstest.append(p)
    effs67trainvstest.append(e)
    

ann67_errorPeaks = np.load('ann67_errorPeaks_notupwindheading_samplenumover35_wvglessthan1.npy')
ann67_alloafdirminuszeta = np.load('ann67_alloAFdirminuszeta_notupwindheading_samplenumover35_wvglessthan1.npy')

ps67errorvsallogammaminuszeta = []
effs67errorvsallogammaminuszeta = []
for i in range(10000):
    error_sub = np.random.choice(ann67_errorPeaks,size=501,replace=False)
    afminuszeta_sub = np.random.choice(ann67_alloafdirminuszeta,size=501,replace=False)
    p,k,e = kuiper((4+error_sub)*np.pi/4,afminuszeta_sub+np.pi,res=8,axis=0)
    ps67errorvsallogammaminuszeta.append(p)
    effs67errorvsallogammaminuszeta.append(e)
    
ann67_possibleturns = np.load('ann67_predPeaksminuscoursedir_samplenumeq07_notheadingupwind_wvglessthan1.npy')
n5000rand_egoAFdir = np.load('n5000rand_afdirreltocoursedir_samplenumeq07_notheadingupwind_wvglessthan1.npy')

ps67turnsvsegoAFdir = []
effs67turnsvsegoAFdir = []
for i in range(10000):
    turns_sub = np.random.choice(ann67_possibleturns,size=501,replace=False)
    egoAFdir_sub = np.random.choice(n5000rand_egoAFdir,size=501,replace=False)
    p,k,e = kuiper((turns_sub),egoAFdir_sub+np.pi,res=8,axis=0)
    ps67turnsvsegoAFdir.append(p)
    effs67turnsvsegoAFdir.append(e)

# fig,ax = plt.subplots(figsize=(5,5));\
# ax.scatter([1,2,3],[k65vs67,k66vs67,k65vs66]);\
# ax.set_ylim(0,2e11);\
# ax.scatter([4],[k67trainvstest]);\
# # plt.savefig('kuiperstats_ann65-66-67_trainingdata_vs_67test.svg');


realflies_erroroverall = np.load('Stupski_allwspdnozero_head810to1000_means.npy')
# ptest,ktest,effecttest = kuiper(realflies_erroroverall+np.pi,(4+ann67_test)*np.pi/4,res=8,axis=0)
# ptrain,ktrain,effecttrain = kuiper(realflies_erroroverall+np.pi,(4+ann67_train)*np.pi/4,res=8,axis=0)

psfliesvs67test = []
effsfliesvs67test = []
for i in range(10000):
    flies_erroverall_sub = np.random.choice(realflies_erroroverall,size=501,replace=False)
    ann67_test_sub = np.random.choice(ann67_test,size=501,replace=False)
    p,k,e = kuiper(realflies_erroroverall+np.pi,(4+ann67_test_sub)*np.pi/4,res=8,axis=0)
    psfliesvs67test.append(p)
    effsfliesvs67test.append(e)
    
psfliesvs67train = []
effsfliesvs67train = []
for i in range(10000):
    flies_erroverall_sub = np.random.choice(realflies_erroroverall,size=501,replace=False)
    ann67_train_sub = np.random.choice(ann67_train,size=501,replace=False)
    p,k,e = kuiper(realflies_erroroverall+np.pi,(4+ann67_train_sub)*np.pi/4,res=8,axis=0)
    psfliesvs67train.append(p)
    effsfliesvs67train.append(e)

ann65_test = np.load('ANN65_n5000randperformance_errorpeaks.npy')
ann66_test = np.load('ANN66_n5000randperformance_errorpeaks.npy')

psfliesvs65test = []
effsfliesvs65test = []
for i in range(10000):
    flies_erroverall_sub = np.random.choice(realflies_erroroverall,size=501,replace=False)
    ann65_test_sub = np.random.choice(ann65_test,size=501,replace=False)
    p,k,e = kuiper(realflies_erroroverall+np.pi,(4+ann65_test_sub)*np.pi/4,res=8,axis=0)
    psfliesvs65test.append(p)
    effsfliesvs65test.append(e)
    
psfliesvs65train = []
effsfliesvs65train = []
for i in range(10000):
    flies_erroverall_sub = np.random.choice(realflies_erroroverall,size=501,replace=False)
    ann65_train_sub = np.random.choice(ann65_train,size=501,replace=False)
    p,k,e = kuiper(realflies_erroroverall+np.pi,(4+ann65_train_sub)*np.pi/4,res=8,axis=0)
    psfliesvs65train.append(p)
    effsfliesvs65train.append(e)
    
psfliesvs66test = []
effsfliesvs66test = []
for i in range(10000):
    flies_erroverall_sub = np.random.choice(realflies_erroroverall,size=501,replace=False)
    ann66_test_sub = np.random.choice(ann66_test,size=501,replace=False)
    p,k,e = kuiper(realflies_erroroverall+np.pi,(4+ann66_test_sub)*np.pi/4,res=8,axis=0)
    psfliesvs66test.append(p)
    effsfliesvs66test.append(e)
    
psfliesvs66train = []
effsfliesvs66train = []
for i in range(10000):
    flies_erroverall_sub = np.random.choice(realflies_erroroverall,size=501,replace=False)
    ann66_train_sub = np.random.choice(ann66_train,size=501,replace=False)
    p,k,e = kuiper(realflies_erroroverall+np.pi,(4+ann66_train_sub)*np.pi/4,res=8,axis=0)
    psfliesvs66train.append(p)
    effsfliesvs66train.append(e)


realflies_wvglessthan1_head = np.load('Stupski_allwspdnozero_head810to1000_means_wvglessthan1.npy')
realflies_wvgovereqto1_head = np.load('Stupski_allwspdnozero_head810to1000_means_wvgovereqto1.npy')
ann67_wvglessthan1 = np.load('ANN67_n5000randperformance_wvglessthan1_errorPeaks.npy')
ann67_wvgovereqto1 = np.load('ANN67_n5000randperformance_wvgovereqto1_errorPeaks.npy')
# pwvgless,kwvgless,effwvgless = kuiper(realflies_wvglessthan1_head+np.pi,(4+ann67_wvglessthan1)*np.pi/4,res=8,axis=0)
# pwvgover,kwvgover,effwvgover = kuiper(realflies_wvgovereqto1_head+np.pi,(4+ann67_wvgovereqto1)*np.pi/4,res=8,axis=0)

psfliesvs67test_lowwvg = []
effsfliesvs67test_lowwvg = []
for i in range(10000):
    flies_lowwvg_sub = np.random.choice(realflies_wvglessthan1_head,size=501,replace=False)
    ann67_lowwvg_sub = np.random.choice(ann67_wvglessthan1,size=501,replace=False)
    p,k,e = kuiper(flies_lowwvg_sub+np.pi,(4+ann67_lowwvg_sub)*np.pi/4,res=8,axis=0)
    psfliesvs67test_lowwvg.append(p)
    effsfliesvs67test_lowwvg.append(e)
    
psfliesvs67test_hiwvg = []
effsfliesvs67test_hiwvg = []
for i in range(10000):
    flies_hiwvg_sub = np.random.choice(realflies_wvgovereqto1_head,size=501,replace=False)
    ann67_hiwvg_sub = np.random.choice(ann67_wvgovereqto1,size=501,replace=False)
    p,k,e = kuiper(flies_hiwvg_sub+np.pi,(4+ann67_hiwvg_sub)*np.pi/4,res=8,axis=0)
    psfliesvs67test_hiwvg.append(p)
    effsfliesvs67test_hiwvg.append(e)
    
psfliesturnsvsafdir_lowwvg = []
effsfliesturnsvsafdir_lowwvg = []
for i in range(10000):
    turns_lowwvg_sub = np.random.choice(realflies_turns,size=501,replace=False)
    afdir_lowwvg_sub = np.random.choice(realflies_AFdiratdecel,size=501,replace=False)
    p,k,e = kuiper(turns_lowwvg_sub+np.pi,(4+afdir_lowwvg_sub)*np.pi/4,res=8,axis=0)
    psfliesturnsvsafdir_lowwvg.append(p)
    effsfliesturnsvsafdir_lowwvg.append(e)


pval65trainvsdmel,k65trainvsdmel,eff65trainvsdmel = kuiper(realflies_erroroverall+np.pi,(4+ann65_train)*np.pi/4,res=8,axis=0)
pval65testvsdmel,k65testvsdmel,eff65testvsdmel = kuiper(realflies_erroroverall+np.pi,(4+ann65_test)*np.pi/4,res=8,axis=0)
pval66trainvsdmel,k66trainvsdmel,eff66trainvsdmel = kuiper(realflies_erroroverall+np.pi,(4+ann66_train)*np.pi/4,res=8,axis=0)
pval66testvsdmel,k66testvsdmel,eff66testvsdmel = kuiper(realflies_erroroverall+np.pi,(4+ann66_test)*np.pi/4,res=8,axis=0)


realflies_successful_headall = np.reshape(successful_heading,successful_heading.shape[0]*successful_heading.shape[1])
realflies_unsuccessful_headall = np.reshape(unsuccessful_heading,unsuccessful_heading.shape[0]*unsuccessful_heading.shape[1])
np.save('Stupski_allwspdnozero_head810to1000_allpoints_successful.npy',realflies_successful_head)
np.save('Stupski_allwspdnozero_head810to1000_allpoints_unsuccessful.npy',realflies_unsuccessful_head)


p_dmelsuccessvsnot_head,k_dmelsuccessvsnot_head,eff_dmelsuccessvsnot_head = kuiper(realflies_successful_headall,realflies_unsuccessful_headall,res=8,axis=0)
p_dmelsuccessvsnot_afwinddiff,k_dmelsuccessvsnot_afwinddiff,eff_dmelsuccessvsnot_afwinddiff = kuiper(realflies_successful_afwinddiffall,realflies_unsuccessful_afwinddiffall,res=8,axis=0)

ann67_heading_success = np.load('ANN67_n5000randperformance_successful_postodor_heading.npy')
ann67_heading_unsuccess = np.load('ANN67_n5000randperformance_unsuccessful_postodor_heading.npy')
ann67_afwinddiff_success = np.load('ANN67_n5000randperformance_successful_AFWindDiff.npy')
ann67_afwinddiff_unsuccess = np.load('ANN67_n5000randperformance_unsuccessful_AFWindDiff.npy')

p_ann67successvsnot_head,k_ann67successvsnot_head,eff_ann67successvsnot_head = kuiper(ann67_heading_success,ann67_heading_unsuccess,res=8,axis=0)
p_ann67successvsnot_afwinddiff,k_ann67successvsnot_afwinddiff,eff_ann67successvsnot_afwinddiff = kuiper(ann67_afwinddiff_success,ann67_afwinddiff_unsuccess,res=8,axis=0)


# p,m = omnibus(ann67_afwinddiff_unsuccess,axis=0)
rf_wvgless_head = realflies_wvglessthan1_head[~np.isnan(realflies_wvglessthan1_head)]
p,z = rayleigh(rf_wvgless_head,axis=0)

fig,ax = plt.subplots(figsize=(8,5));\
ax.scatter([1,2,3],[k65vs67,k66vs67,k65vs66]);\
# ax.set_ylim(1000000,2e11);\
ax.set_yscale('log');\
ax.scatter([4],[k67trainvstest]);\
ax.scatter([5,6],[ktrain,ktest]);\
ax.scatter([7,8],[kwvgless,kwvgover]);\
ax.scatter([9,10,11,12],[k65trainvsdmel,k65testvsdmel,k66trainvsdmel,k66testvsdmel]);\
ax.scatter([13,14],[k_dmelsuccessvsnot_head,k_dmelsuccessvsnot_afwinddiff]);\
ax.scatter([15,16],[k_ann67successvsnot_head,k_ann67successvsnot_afwinddiff]);

fig,ax = plt.subplots(figsize=(8,5));\
ax.scatter([1,2,3],[eff65vs67,eff66vs67,eff65vs66]);\
ax.scatter([4],[effect67trainvstest]);\
ax.scatter([5,6],[effecttrain,effecttest]);\
ax.scatter([7,8],[effwvgless,effwvgover]);\
ax.scatter([9,10,11,12],[eff65trainvsdmel,eff65testvsdmel,eff66trainvsdmel,eff66testvsdmel]);\
ax.scatter([13,14],[eff_dmelsuccessvsnot_head,eff_dmelsuccessvsnot_afwinddiff]);\
ax.scatter([15,16],[eff_ann67successvsnot_head,eff_ann67successvsnot_afwinddiff]);
# plt.savefig('kuiperstats_all_figure_tests.svg');
