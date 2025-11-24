# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:54:33 2025

@author: mayc06
"""

# Scripts for making Stupski data distributions and scatterplots for May et al 2025

from extract_trajectories_from_orcoflashStupski import *


traj_list40 = get_trajlist_from_behdata('OrcoCsChrimson_laminar_wind_merged.csv',-500.0,2000.0,windspeed=40.0)


for traj in traj_list40:
    o = np.abs(traj['heading'])
    traj['up-down-cross']=0
    traj['up-down-cross'] = np.where( o<=np.pi/8, 1.0, 
                                      traj['up-down-cross'])
    traj['up-down-cross'] = np.where( (o>np.pi/8) & (o<=3*np.pi/8), 0.5, 
                                      traj['up-down-cross'])
    # traj['up-down-cross'] = np.where( o>3*np.pi/8 & o<=5*np.pi/8, 0.0,
    #                                  traj['up-down-cross'])
    traj['up-down-cross'] = np.where( (o>5*np.pi/8) & (o<=7*np.pi/8), -0.5,
                                      traj['up-down-cross'])
    traj['up-down-cross'] = np.where( o>7*np.pi/8, -1.0,
                                      traj['up-down-cross'])
    traj['up-down-cross'] = np.where( o==np.nan, np.nan,
                                      traj['up-down-cross'])
    
for traj in traj_list40:
    traj['rel_winddir'] = -1*traj['heading']
    vpara = traj['ground speed'].values
    vperp = np.zeros_like(vpara)
    wspd = traj['windspeed']/100 
    phi = traj['heading']
    zeta = np.ones_like(vpara)*np.pi
    afdir = compute_airspeed_angle(vpara,vperp,wspd,phi,zeta)
    traj['airflow_dir'] = afdir
    traj['afdir_minus_rel_winddir'] = afdir - traj['rel_winddir']
    
for traj in traj_list40:
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
    


w40_success810to1000 = get_distribution(traj_list40,'up-down-cross',810,1000,1)

w40_wvg150to340 = get_distribution(traj_list40,'w_over_g',150,340,1)

w40_head810to1000 = get_distribution(traj_list40,'heading',810,1000,1)

wvglessthan1,wvgovereqto1 = sep_by_value(traj_list40,'w_over_g',150,340,1.0)
wvglessthan1_success810to1000 = get_distribution(wvglessthan1,'up-down-cross',810,1000,1)
wvgovereqto1_success810to1000 = get_distribution(wvgovereqto1,'up-down-cross',810,1000,1)


successful,unsuccessful = sep_by_value(traj_list40,'heading',810,1000,np.pi/8)
successful_afdirvszeta150to340 = get_distribution(successful,'afdir_algo_score',150,340,1)
unsuccessful_afdirvszeta150to340 = get_distribution(unsuccessful,'afdir_algo_score',150,340,1)
successful_heading = get_distribution(successful,'heading',810,1000,1)
unsuccessful_heading = get_distribution(unsuccessful,'heading',810,1000,1)
successful_afdirvszetaDEG150to340 = get_distribution(successful,'afdir_minus_rel_winddir',150,340,1)
unsuccessful_afdirvszetaDEG150to340 = get_distribution(unsuccessful,'afdir_minus_rel_winddir',150,340,1)

fig,ax = plt.subplots(figsize = (5,5));\
ax.scatter(np.max(w40_wvg150to340,axis=1),np.nanmean(np.abs(w40_head810to1000),axis=1),alpha=0.3);\
ax.set_xscale('log');\
ax.set_ylim(3.2,-0.1);\
figtitle = 'w40_absHeading_vs_maxwvg_scatter.svg';\
plt.savefig(figtitle);

fig,ax = plt.subplots(figsize = (5,5));\
ax.scatter(np.max(w40_wvg150to340,axis=1),np.nanmean(w40_success810to1000,axis=1),alpha=0.3);\
ax.set_xscale('log');\
figtitle = 'w40_success_vs_maxwvg_scatter.svg';\
plt.savefig(figtitle);

n = w40_success810to1000.shape[0]
fig,ax = plt.subplots(figsize=(4,6));\
ax.hist(np.mean(w40_success810to1000,axis=1),histtype='step',bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
plt.savefig('Stupski_success810to1000_fractiondistro.svg');

n = wvglessthan1_success810to1000.shape[0]
fig,ax = plt.subplots(figsize=(4,6));\
ax.hist(np.mean(wvglessthan1_success810to1000,axis=1),histtype='step',bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
plt.savefig('Stupski_success810to1000_wvglessthan1_fractiondistro.svg');

n = wvgovereqto1_success810to1000.shape[0]
fig,ax = plt.subplots(figsize=(4,6));\
ax.hist(np.mean(wvgovereqto1_success810to1000,axis=1),histtype='step',bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
plt.savefig('Stupski_success810to1000_wvgovereqto1_fractiondistro.svg');


fig,ax = plt.subplots(figsize=(4,4));\
n = successful_heading.shape[0]*successful_heading.shape[1]
ax.hist(np.reshape(successful_heading,successful_heading.shape[0]*successful_heading.shape[1]),bins=np.arange(-np.pi-np.pi/16,np.pi+np.pi/8,np.pi/8),weights=np.ones(n)/n);\
n = unsuccessful_heading.shape[0]*unsuccessful_heading.shape[1]
ax.hist(np.reshape(unsuccessful_heading,unsuccessful_heading.shape[0]*unsuccessful_heading.shape[1]),bins=np.arange(-np.pi-np.pi/16,np.pi+np.pi/8,np.pi/8),weights=np.ones(n)/n);\
ax.set_xticks([-3.14,-1.57,0,1.57,3.14])
plt.savefig('successful_vs_unsuccessful_headingdistros.svg');

fig,ax = plt.subplots(figsize=(4,4));\
n = successful_afdirvszeta150to340.shape[0]*successful_afdirvszeta150to340.shape[1]
ax.hist(np.reshape(successful_afdirvszeta150to340,successful_afdirvszeta150to340.shape[0]*successful_afdirvszeta150to340.shape[1]),bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
n = unsuccessful_afdirvszeta150to340.shape[0]*unsuccessful_afdirvszeta150to340.shape[1]
ax.hist(np.reshape(unsuccessful_afdirvszeta150to340,unsuccessful_afdirvszeta150to340.shape[0]*unsuccessful_afdirvszeta150to340.shape[1]),bins=np.arange(-1.25,1.75,0.5),weights=np.ones(n)/n);\
plt.savefig('successful_vs_unsuccessful_afdirvszetadistros.svg');

fig,ax = plt.subplots(figsize=(4,4));\
n = successful_afdirvszetaDEG150to340.shape[0]*successful_afdirvszetaDEG150to340.shape[1]
ax.hist(np.reshape(successful_afdirvszetaDEG150to340,successful_afdirvszetaDEG150to340.shape[0]*successful_afdirvszetaDEG150to340.shape[1]),bins=np.arange(-np.pi-np.pi/16,np.pi+np.pi/8,np.pi/8),weights=np.ones(n)/n);\
n = unsuccessful_afdirvszetaDEG150to340.shape[0]*unsuccessful_afdirvszetaDEG150to340.shape[1]
ax.hist(np.reshape(unsuccessful_afdirvszetaDEG150to340,unsuccessful_afdirvszetaDEG150to340.shape[0]*unsuccessful_afdirvszetaDEG150to340.shape[1]),bins=np.arange(-np.pi-np.pi/16,np.pi+np.pi/8,np.pi/8),weights=np.ones(n)/n);
ax.set_xticks([-3.14,-1.57,0,1.57,3.14])
plt.savefig('successful_vs_unsuccessful_AF-WinddirDiffdistros.svg')
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
