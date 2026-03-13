# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:54:33 2025

@author: mayc06
"""

# Scripts for making Stupski data distributions and scatterplots for May et al 2025

from extract_trajectories_from_orcoflashStupski import *
from math import isnan
from scipy.stats import circmean


traj_list40 = get_trajlist_from_behdata('OrcoCsChrimson_laminar_wind_merged.csv',-500.0,2000.0,windspeed=40.0)
traj_list0 = get_trajlist_from_behdata('orco20_PWM_flash_postQC.csv',-500.0,2000.0,windspeed=0.0)
traj_list04 = get_trajlist_from_behdata('orco60_PWM_flash_postQC.csv',-500.0,2000.0,windspeed=4.0)
traj_list15 = get_trajlist_from_behdata('orco_100PWM_flash_postQC.csv',-500.0,2000.0,windspeed=15.0)


traj_list_all = []
traj_list_all.extend(traj_list04.copy())
traj_list_all.extend(traj_list15.copy())
traj_list_all.extend(traj_list40.copy())

# for traj in traj_list15:
#     o = np.abs(traj['heading'])
#     traj['up-down-cross']=0
#     traj['up-down-cross'] = np.where( o<=np.pi/8, 1.0, 
#                                       traj['up-down-cross'])
#     traj['up-down-cross'] = np.where( (o>np.pi/8) & (o<=3*np.pi/8), 0.5, 
#                                       traj['up-down-cross'])
#     # traj['up-down-cross'] = np.where( o>3*np.pi/8 & o<=5*np.pi/8, 0.0,
#     #                                  traj['up-down-cross'])
#     traj['up-down-cross'] = np.where( (o>5*np.pi/8) & (o<=7*np.pi/8), -0.5,
#                                       traj['up-down-cross'])
#     traj['up-down-cross'] = np.where( o>7*np.pi/8, -1.0,
#                                       traj['up-down-cross'])
#     traj['up-down-cross'] = np.where( o==np.nan, np.nan,
#                                       traj['up-down-cross'])
    
for traj in traj_list_all:
    traj['rel_winddir'] = -1*traj['heading']  ### ???? Why.
    vpara = traj['ground speed'].values
    vperp = np.zeros_like(vpara)
    wspd = traj['windspeed']/100 
    phi = traj['heading']
    zeta = np.ones_like(vpara)*np.pi
    afdir = compute_airspeed_angle(vpara,vperp,wspd,phi,zeta)  ### egocentric AF dir from real fly data.
    traj['airflow_dir'] = afdir
    traj['afdir_minus_rel_winddir'] = afdir - traj['rel_winddir']  ### What even is this. What was I on.
    traj['afdir_allocoord'] = afdir + traj['heading']  ### This makes more sense than the 'rel_winddir' stuff.
    
for traj in traj_list_all:
    o = np.abs(traj['afdir_minus_rel_winddir'])
    traj['afdir_algo_score']=0
    traj['afdir_algo_score'] = np.where( o<=np.pi/8, 1.0, 
                                      traj['afdir_algo_score'])
    traj['afdir_algo_score'] = np.where( (o>np.pi/8) & (o<=3*np.pi/8), 0.5, 
                                      traj['afdir_algo_score'])
    traj['afdir_algo_score'] = np.where( (o>5*np.pi/8) & (o<=7*np.pi/8), -0.5,
                                      traj['afdir_algo_score'])
    traj['afdir_algo_score'] = np.where( o>7*np.pi/8, -1.0,
                                      traj['afdir_algo_score'])
    traj['afdir_algo_score'] = np.where( o==np.nan, np.nan,
                                      traj['afdir_algo_score'])

traj_list_all = []
traj_list_all.extend(traj_list04.copy())
traj_list_all.extend(traj_list15.copy())
traj_list_all.extend(traj_list40.copy())

traj_list = traj_list_all.copy()

w_all_success810to1000 = get_distribution(traj_list,'up-down-cross',810,1000,1)

w_all_wvg150to340 = get_distribution(traj_list,'w_over_g',150,340,1)
w04_wvg150to340 = get_distribution(traj_list04,'w_over_g',150,340,1)
w15_wvg150to340 = get_distribution(traj_list15,'w_over_g',150,340,1)
w40_wvg150to340 = get_distribution(traj_list40,'w_over_g',150,340,1)

w_all_head810to1000 = get_distribution(traj_list,'heading',810,1000,1)
w04_head810to1000 = get_distribution(traj_list04,'heading',810,1000,1)
w15_head810to1000 = get_distribution(traj_list15,'heading',810,1000,1)
w40_head810to1000 = get_distribution(traj_list40,'heading',810,1000,1)

wvglessthan1,wvgovereqto1 = sep_by_abs_value(traj_list,'w_over_g',150,340,1.0)
wvglessthan1_success810to1000 = get_distribution(wvglessthan1,'up-down-cross',810,1000,1)
wvgovereqto1_success810to1000 = get_distribution(wvgovereqto1,'up-down-cross',810,1000,1)
wvglessthan1_head810to1000 = get_distribution(wvglessthan1,'heading',810,1000,1)
wvgovereqto1_head810to1000 = get_distribution(wvgovereqto1,'heading',810,1000,1)

wvglessthan1_head150to340 = get_distribution(wvglessthan1,'heading',150,340,1)
wvgovereqto1_head150to340 = get_distribution(wvgovereqto1,'heading',150,340,1)

upwindatdecel,notupwindatdecel = sep_by_abs_value(wvglessthan1,'heading',150,340,np.pi/3)

notupwindatdecel_head810to1000 = get_distribution(notupwindatdecel,'heading',810,1000,1);\
upwindatdecel_head810to1000 = get_distribution(upwindatdecel,'heading',810,1000,1);

notupwindatdecel_head150to340 = get_distribution(notupwindatdecel,'heading',150,340,1);\
upwindatdecel_head150to340 = get_distribution(upwindatdecel,'heading',150,340,1);

notupwindatdecel_afdir810to1000 = get_distribution(notupwindatdecel,'airflow_dir',810,1000,1);\
upwindatdecel_afdir810to1000 = get_distribution(upwindatdecel,'airflow_dir',810,1000,1);

notupwindatdecel_afdir150to340 = get_distribution(notupwindatdecel,'airflow_dir',150,340,1);\
upwindatdecel_afdir150to340 = get_distribution(upwindatdecel,'airflow_dir',150,340,1);

fig,ax = plt.subplots();\
ax.hist(np.reshape(notupwindatdecel_head150to340,notupwindatdecel_head150to340.shape[0]*notupwindatdecel_head150to340.shape[1]),bins=31,alpha=0.4);\
ax.hist(np.reshape(notupwindatdecel_head810to1000,notupwindatdecel_head810to1000.shape[0]*notupwindatdecel_head810to1000.shape[1]),bins=31,alpha=0.4);\
ax.set_title('headings at decel and postodor');

fig,ax = plt.subplots();\
ax.hist(np.reshape(notupwindatdecel_afdir150to340,notupwindatdecel_afdir150to340.shape[0]*notupwindatdecel_afdir150to340.shape[1]),bins=31,alpha=0.4);\
ax.hist(np.reshape(notupwindatdecel_afdir810to1000,notupwindatdecel_afdir810to1000.shape[0]*notupwindatdecel_afdir810to1000.shape[1]),bins=31,alpha=0.4);\
ax.set_title('afdirs at decel and postodor');

a = circmean(notupwindatdecel_head150to340,high=np.pi,low=-np.pi,axis=1,nan_policy='omit')
b = circmean(notupwindatdecel_head810to1000,high=np.pi,low=-np.pi,axis=1,nan_policy='omit')
turns  = b-a

fig,ax = plt.subplots(figsize=(7,5));\
ax.hist(wrapToPi(turns),bins=np.arange(-np.pi-np.pi/16,np.pi+np.pi/8,np.pi/8),alpha=1);\
ax.hist(circmean(notupwindatdecel_afdir150to340,high=np.pi,low=-np.pi,axis=1,nan_policy='omit'),bins=np.arange(-np.pi-np.pi/16,np.pi+np.pi/8,np.pi/8),alpha=0.4);\
ax.set_title('mean heading change Blue, afdir at decel Orange');\
ax.set_xticks([-3.14,-1.57,0,1.57,3.14]);
realflies_turns = wrapToPi(turns);
# np.save('realflies_turns_wvglessthan1_notupwind.npy',realflies_turns)
realflies_AFdiratdecel = circmean(notupwindatdecel_afdir150to340,high=np.pi,low=-np.pi,axis=1,nan_policy='omit')
# np.save('realflies_AFdiratdecel_wvglessthan1_notupwind.npy',realflies_AFdiratdecel)

fig,ax=plt.subplots();\
ax.set_title('mean headings, decelOrange postodorBlue');\
ax.hist(circmean(notupwindatdecel_head810to1000,high=np.pi,low=-np.pi,axis=1,nan_policy='omit'),bins=np.arange(-np.pi-np.pi/8,np.pi+np.pi/2,np.pi/8),alpha=0.4);\
ax.hist(circmean(notupwindatdecel_head150to340,high=np.pi,low=-np.pi,axis=1,nan_policy='omit'),bins=np.arange(-np.pi-np.pi/8,np.pi+np.pi/2,np.pi/8),alpha=0.4);

fig,ax=plt.subplots(figsize=(5,5));\
ax.set_title('mean headings, xaxisDecel yaxisPostodor');\
ax.scatter(circmean(notupwindatdecel_head150to340,high=np.pi,low=-np.pi,axis=1,nan_policy='omit'),circmean(notupwindatdecel_head810to1000,high=np.pi,low=-np.pi,axis=1,nan_policy='omit'));\
figtitle = 'realflies_scatter_meanHeadings_yPostodor_xDecel.svg';\
# plt.savefig(figtitle);

successful,unsuccessful = sep_by_abs_value(traj_list,'heading',810,1000,np.pi/8)
successful_afdirvszeta150to340 = get_distribution(successful,'afdir_algo_score',150,340,1)
unsuccessful_afdirvszeta150to340 = get_distribution(unsuccessful,'afdir_algo_score',150,340,1)
successful_heading = get_distribution(successful,'heading',810,1000,1)
unsuccessful_heading = get_distribution(unsuccessful,'heading',810,1000,1)
successful_afdirvszetaDEG150to340 = get_distribution(successful,'afdir_minus_rel_winddir',150,340,1)
unsuccessful_afdirvszetaDEG150to340 = get_distribution(unsuccessful,'afdir_minus_rel_winddir',150,340,1)

fig,ax = plt.subplots(figsize = (5,5));\
ax.scatter(np.max(w04_wvg150to340,axis=1),np.nanmean(np.abs(w04_head810to1000),axis=1),alpha=0.8);\
ax.set_xscale('log');\
ax.set_xlim(0.02,4000);\
ax.set_ylim(3.2,-0.1);\
ax.set_yticks([0,1.57/2,1.57,3*1.57/2,3.14]);
figtitle = 'w04_absHeading_vs_maxwvg_scatter_highalpha.svg';\
# plt.savefig(figtitle);

fig,ax = plt.subplots(figsize = (5,5));\
ax.scatter(np.max(w_all_wvg150to340,axis=1),np.nanmean(w_all_success810to1000,axis=1),alpha=0.3);\
ax.set_xscale('log');\
figtitle = 'w_allnozero_success_vs_maxwvg_scatter.svg';\
# plt.savefig(figtitle);

n = w_all_success810to1000.shape[0]
fig,ax = plt.subplots(figsize=(4,6));\
ax.hist(np.mean(w_all_success810to1000,axis=1),histtype='step',bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
# plt.savefig('Stupski_success810to1000_fractiondistro_allwspdsnozero.svg');

n = wvglessthan1_success810to1000.shape[0]
fig,ax = plt.subplots(figsize=(4,6));\
ax.hist(np.mean(wvglessthan1_success810to1000,axis=1),histtype='step',bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
# plt.savefig('Stupski_success810to1000_wvglessthan1_fractiondistro_allwspdsnozero.svg');

n = wvgovereqto1_success810to1000.shape[0]
fig,ax = plt.subplots(figsize=(4,6));\
ax.hist(np.mean(wvgovereqto1_success810to1000,axis=1),histtype='step',bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
# plt.savefig('Stupski_success810to1000_wvgovereqto1_fractiondistro_allwspdsnozero.svg');


fig,ax = plt.subplots(figsize=(4,4));\
n = successful_heading.shape[0]*successful_heading.shape[1]
ax.hist(np.abs(np.reshape(successful_heading,successful_heading.shape[0]*successful_heading.shape[1])),bins=np.arange(-np.pi/8,np.pi+np.pi/4,np.pi/4),weights=np.ones(n)/n);\
n = unsuccessful_heading.shape[0]*unsuccessful_heading.shape[1]
ax.hist(np.abs(np.reshape(unsuccessful_heading,unsuccessful_heading.shape[0]*unsuccessful_heading.shape[1])),bins=np.arange(-np.pi/8,np.pi+np.pi/4,np.pi/4),weights=np.ones(n)/n);\
ax.set_xticks([0,1.57/2,1.57,3*1.57/2,3.14])
# plt.savefig('successful_vs_unsuccessful_absheadingdistros_allwspdsnozero.svg');

fig,ax = plt.subplots(figsize=(4,4));\
n = successful_afdirvszeta150to340.shape[0]*successful_afdirvszeta150to340.shape[1]
ax.hist(np.reshape(successful_afdirvszeta150to340,successful_afdirvszeta150to340.shape[0]*successful_afdirvszeta150to340.shape[1]),bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
n = unsuccessful_afdirvszeta150to340.shape[0]*unsuccessful_afdirvszeta150to340.shape[1]
ax.hist(np.reshape(unsuccessful_afdirvszeta150to340,unsuccessful_afdirvszeta150to340.shape[0]*unsuccessful_afdirvszeta150to340.shape[1]),bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
# plt.savefig('successful_vs_unsuccessful_afdirvszetadistros_allwspdnozeros.svg');

fig,ax = plt.subplots(figsize=(4,4));\
n = successful_afdirvszetaDEG150to340.shape[0]*successful_afdirvszetaDEG150to340.shape[1]
ax.hist(np.abs(np.reshape(successful_afdirvszetaDEG150to340,successful_afdirvszetaDEG150to340.shape[0]*successful_afdirvszetaDEG150to340.shape[1])),bins=np.arange(-np.pi/8,np.pi+np.pi/4,np.pi/4),weights=np.ones(n)/n);\
n = unsuccessful_afdirvszetaDEG150to340.shape[0]*unsuccessful_afdirvszetaDEG150to340.shape[1]
ax.hist(np.abs(np.reshape(unsuccessful_afdirvszetaDEG150to340,unsuccessful_afdirvszetaDEG150to340.shape[0]*unsuccessful_afdirvszetaDEG150to340.shape[1])),bins=np.arange(-np.pi/8,np.pi+np.pi/4,np.pi/4),weights=np.ones(n)/n);
ax.set_xticks([0,1.57/2,1.57,3*1.57/2,3.14])
# plt.savefig('successful_vs_unsuccessful_absAF-WinddirDiffdistros_allwspdsnozero.svg')


wvg40 = np.empty((len(traj_list40),len(traj_list40[0])))
for traj in range(len(traj_list40)):
    wvg40[traj,:] = traj_list40[traj]['w_over_g']
wvg15 = np.empty((len(traj_list15),len(traj_list15[0])))
for traj in range(len(traj_list15)):
    wvg15[traj,:] = traj_list15[traj]['w_over_g']
wvg04 = np.empty((len(traj_list04),len(traj_list04[0])))
for traj in range(len(traj_list04)):
    wvg04[traj,:] = traj_list04[traj]['w_over_g']
fig,ax = plt.subplots();
ax.plot(np.nanmean(wvg40,axis=0));
ax.plot(np.nanmean(wvg15,axis=0));
ax.plot(np.nanmean(wvg04,axis=0));

w40_wvgneg490to1000 = get_distribution(traj_list40,'w_over_g',-490,1000,1)
w15_wvgneg490to1000 = get_distribution(traj_list15,'w_over_g',-490,1000,1)
w04_wvgneg490to1000 = get_distribution(traj_list04,'w_over_g',-490,1000,1)

fig,ax = plt.subplots(figsize=(5,5));\
x = np.arange(-490,1010,10);\
# m = np.nanmean(w40_wvgneg490to1000,axis=0);\
# sem = np.nanstd(w40_wvgneg490to1000,axis=0)/np.sqrt(len(traj_list40)-1);\
v = np.nanvar(w40_wvgneg490to1000,axis=0);\
ax.scatter(x,v);\
# ax.fill_between(x,m-sem,m+sem,alpha=0.2);\
# m = np.nanmean(w15_wvgneg490to1000,axis=0);\
# sem = np.nanstd(w15_wvgneg490to1000,axis=0)/np.sqrt(len(traj_list15)-1);\
v = np.nanvar(w15_wvgneg490to1000,axis=0);\
ax.scatter(x,v);\
# ax.fill_between(x,m-sem,m+sem,alpha=0.2);\
# m = np.nanmean(w04_wvgneg490to1000,axis=0);\
# sem = np.nanstd(w04_wvgneg490to1000,axis=0)/np.sqrt(len(traj_list04)-1);\
v = np.nanvar(w04_wvgneg490to1000,axis=0);\
ax.scatter(x,v);\
# ax.fill_between(x,m-sem,m+sem,alpha=0.2);\
# ax.set_ylim(0,15);\
ax.set_yscale('log')
ax.set_xticks([-400,-200,0,200,400,600,800,1000]);\
figtitle = 'w_all_var_wvg_vs_time.svg';\
plt.savefig(figtitle);


n = w_all_head810to1000.shape[0]
fig,ax = plt.subplots(figsize=(5,5));\
ax.hist(np.nanmean(w_all_head810to1000,axis=1),histtype='step',bins=np.arange(-np.pi-np.pi/8,np.pi+np.pi/4,np.pi/4),weights=np.ones(n)/n);\
ax.set_xticks([np.round(-np.pi,2),np.round(-np.pi/2,2),0.0,np.round(np.pi/2,2),np.round(np.pi,2)]);

realflies_erroroverall = np.nanmean(w_all_head810to1000,axis=1)
np.save('Stupski_allwspdnozero_head810to1000_means.npy',realflies_erroroverall);

realflies_wvglessthan1_head = np.nanmean(wvglessthan1_head810to1000,axis=1)
realflies_wvgovereqto1_head = np.nanmean(wvgovereqto1_head810to1000,axis=1)
np.save('Stupski_allwspdnozero_head810to1000_means_wvglessthan1.npy',realflies_wvglessthan1_head)
np.save('Stupski_allwspdnozero_head810to1000_means_wvgovereqto1.npy',realflies_wvgovereqto1_head)


realflies_successful_head = np.nanmean(successful_heading,axis=1)
realflies_unsuccessful_head = np.nanmean(unsuccessful_heading,axis=1)
np.save('Stupski_allwspdnozero_head810to1000_means_successful.npy',realflies_successful_head)
np.save('Stupski_allwspdnozero_head810to1000_means_unsuccessful.npy',realflies_unsuccessful_head)

realflies_successful_headall = np.reshape(successful_heading,successful_heading.shape[0]*successful_heading.shape[1])
realflies_unsuccessful_headall = np.reshape(unsuccessful_heading,unsuccessful_heading.shape[0]*unsuccessful_heading.shape[1])
np.save('Stupski_allwspdnozero_head810to1000_allpoints_successful.npy',realflies_successful_headall)
np.save('Stupski_allwspdnozero_head810to1000_allpoints_unsuccessful.npy',realflies_unsuccessful_headall)

realflies_successful_afwinddiff = np.nanmean(successful_afdirvszetaDEG150to340,axis=1)
realflies_unsuccessful_afwinddiff = np.nanmean(unsuccessful_afdirvszetaDEG150to340,axis=1)
np.save('Stupski_allwspdnozero_afwinddiff150to340_means_successful.npy',realflies_successful_afwinddiff)
np.save('Stupski_allwspdnozero_afwinddiff150to340_means_unsuccessful.npy',realflies_unsuccessful_afwinddiff)

realflies_successful_afwinddiffall = np.reshape(successful_afdirvszetaDEG150to340,successful_afdirvszetaDEG150to340.shape[0]*successful_afdirvszetaDEG150to340.shape[1])
realflies_unsuccessful_afwinddiffall = np.reshape(unsuccessful_afdirvszetaDEG150to340,unsuccessful_afdirvszetaDEG150to340.shape[0]*unsuccessful_afdirvszetaDEG150to340.shape[1])
np.save('Stupski_allwspdnozero_afwinddiff150to340_allpoints_successful.npy',realflies_successful_afwinddiffall)
np.save('Stupski_allwspdnozero_afwinddiff150to340_allpoints_unsuccessful.npy',realflies_unsuccessful_afwinddiffall)


## Dates metadata for Floris

dates_40 = []
for traj in traj_list40:
    s = list(set(traj.obj_id_unique))
    for i in s:
        if isinstance(i,str):
            oiu = i
    dates_40.append(oiu[-15:-7])
uniq_dates_40 = list(set(dates_40))

dates_0 = []
for traj in traj_list0:
    s = list(set(traj.obj_id_unique))
    for i in s:
        if isinstance(i,str):
            oiu = i
    dates_0.append(oiu[-15:-7])
uniq_dates_0 = list(set(dates_0))

dates_04 = []
for traj in traj_list04:
    s = list(set(traj.obj_id_unique))
    for i in s:
        if isinstance(i,str):
            oiu = i
    dates_04.append(oiu[-15:-7])
uniq_dates_04 = list(set(dates_04))

dates_15 = []
for traj in traj_list15:
    s = list(set(traj.obj_id_unique))
    for i in s:
        if isinstance(i,str):
            oiu = i
    dates_15.append(oiu[-15:-7])
uniq_dates_15 = list(set(dates_15))

# fig,ax = plt.subplots();\
# ax.plot(traj_list40[0]['up-down-cross']);
# w40_success810to1000 = get_distribution(traj_list40,'up-down-cross',810,1000,1)
# w40_success810to1000
# w40_success810to1000.shape
# len(traj_list40)
# np.mean(w40_success810to1000)
# np.mean(w40_success810to1000,axis=1)
# len(np.mean(w40_success810to1000,axis=1))
# len(np.mean(w40_success810to1000,axis=0))
# len(np.mean(w40_success810to1000,axis=1))
# thing = np.mean(w40_success810to1000,axis=1)
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=30);
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=100);
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=20);
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=5);
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=6);
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=4);
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=np.arange(-1.5,1.5,0.5));
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=np.arange(-1.25,1.25,5));
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=np.arange(-1.25,1.25,0.55));
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=np.arange(-1.25,1.25,0.5));
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=np.arange(-1.25,1.75,0.5));
# np.arange(-1.25,1.75,0.5)
# 400+300+300+25+50
# fig,ax = plt.subplots();\
# ax.hist(thing,bins=np.arange(-1.25,1.75,0.5));\
# plt.savefig('Stupski_success810to1000_distro.svg');
# wvglessthan1,wvgovereqto1 = sep_by_value(traj_list40,'w_over_g',150,340,1.0)
# wvglessthan1
# len(wvglessthan1)
# len(wvgovereqto1)
# 190+863
# wvglessthan1_success810to1000 = get_distribution(wvglessthan1,'up-down-cross',810,1000,1)
# wvgovereqto1_success810to1000 = get_distribution(wvgovereqto1,'up-down-cross',810,1000,1)
# meanslessthan1 = np.mean(wvglessthan1_success810to1000,axis=1)
# meansovereqto1 = np.mean(wvgovereqto1_success810to1000,axis=1)
# fig,ax = plt.subplots();\
# ax.hist(meanslessthan1,bins=np.arange(-1.25,1.75,0.5));\
# plt.savefig('Stupski_wvglessthan1_success810to1000_distro.svg');
# fig,ax = plt.subplots();\
# ax.hist(meansovereqto1,bins=np.arange(-1.25,1.75,0.5));\
# plt.savefig('Stupski_wvgovereqto1_success810to1000_distro.svg');



### OLD
# all_wvg150to340 = np.vstack([w4_wvg150to340,w15_wvg150to340,w40_wvg150to340])
# all_success810to1000 = np.vstack([w4_success810to1000,w15_success810to1000,w40_success810to1000])
# fig,ax = plt.subplots();\
# ax.scatter(np.max(all_wvg150to340,axis=1),np.nanmean(all_success810to1000,axis=1),alpha=0.3);\
# ax.set_xscale('log');\
# plt.savefig('all_meanSuccess810to1000_vs_all_maxWvg150to340.svg')

# fig,ax = plt.subplots();\
# ax.scatter(np.max(all_wvg150to340,axis=1),abs(np.nanmean(all_head810to1000,axis=1)),alpha=0.3);\
# ax.set_xscale('log');\
# plt.savefig('all_head810to1000_vs_all_wvg150to340.svg')


# fig,ax = plt.subplots();\
# ax.plot(traj_list40[0]['heading']);\
# ax.plot(np.abs(traj_list40[0]['ang vel']));


# decels = []
# phi0s = []
# for traj in traj_list100:
#     decel = min(traj['ground speed'][0.0:250.0]) - traj['ground speed'][-50.0:0.0].mean()
#     deceltime = traj['ground speed'][0.0:250.0].idxmin() / 1000   # time to min groundspeed from opto on in seconds
#     decels.append(decel/deceltime)
#     phi0s.append(traj['heading'][-50.0:0.0].mean())


# fig,ax = plt.subplots(figsize=(5,5));\
# ax.scatter(phi0s,decels,s=5);\
# ax.set_ylim(-12.5,2);\
# ax.invert_yaxis();
# plt.savefig('Stupski-orco100_phi0s-vs-decels.svg')

# fig,ax = plt.subplots(figsize=(5,5));
# for traj in traj_list100:
#     ax.plot(traj['ground speed'],'k',linewidth=0.1)
    
    
# count=0
# fig,ax = plt.subplots(figsize=(5,5));
# for traj in traj_list20:
#     print(count)
#     a = traj['ground speed']
#     smootha,da = pynumdiff.linear_model.savgoldiff(a,10,[3,10,10])
#     ax.plot(traj.index,da,'k',linewidth=0.1)
#     count+=1
