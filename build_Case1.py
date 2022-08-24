#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:47:09 2022

@author: jakravit
"""
from build_library import *
import os
import pandas as pd
import numpy as np
import pickle
import random

def build_Case1(phy_library, datanap, benthic_lib, adj_lib, aero_lib):
    
    # lambda
    l = np.arange(400, 902.5, 2.5)  
    
    # initiate dictionary
    iops = {}
    
    # VSF angles
    theta901 = np.arange(0, 90.1, 0.1) # length 901
    nang=901
    angles=np.cos(np.deg2rad(theta901)) # length 901
    theta1 = np.deg2rad(theta901) 
    theta2 = np.flipud(np.pi-np.deg2rad(theta901[0:900]))
    theta=np.append(theta1,theta2)
    iops['VSF_angles'] = np.rad2deg(theta)
    
    # lambda
    iops['lambda'] = np.arange(.4, .9025, .0025).astype(np.float32) 
    
    #################### PHYTOPLANKTON ############################################## 
    
    # assign class contributions
    alphas = [.5, 1, 5, 10]
    # groups = ['Haptophytes','Diatoms','Dinoflagellates','Cryptophytes',
    #           'Green_algae','Cyano_blue','Heterokonts','Cyano_red','Rhodophytes',
    #          'Eustigmatophyte', 'Raphidophyte']
    phyto_class_frxn, maxpft = dirichlet_phyto(alphas)
    
    # define species for each class first
    frxns = np.linspace(.05, .95, 50)
    for c, f in phyto_class_frxn.items():
        specs = np.random.choice(list(phy_library[c].keys()),2)
        fx = np.random.choice(frxns)
        for i, sp in enumerate(specs):
            f['sps'].append(sp+'_{}'.format(i))
            if i == 0:
                f['fx'].append(fx)
            else:
                f['fx'].append(1-fx)
    
    # define chl distribution and get chl conc
    sigma,scale = lognorm_params(.5,.5)
    chlDist = lognorm_random(sigma, scale, 20000)
    chl = round(np.random.choice(chlDist), 3)
    
    # phyto classes
    classIOPs = {}
    classIOPs['TotChl'] = chl
    classIOPs = phyto_iops_case1(phyto_class_frxn, phy_library, classIOPs)
    
    # chl fluorescence 
    fqy = np.random.choice(np.linspace(.005,.02,50)).astype(np.float16)
    aphyEuk = []
    aphyCy = []
    for i,k in classIOPs.items():
        if i in ['TotChl','a_tot','b_tot','c_tot','bb_tot','FQY','Qa','fluorescence']:
            continue
        elif i in ['Cyano_blue']:
            aphyCy.append(k['a_tot'])
        else:
            aphyEuk.append(k['a_tot'])
    aphyEukSum = pd.DataFrame(aphyEuk).sum().values
    
    if len(aphyCy) == 0:
        aphyCySum = np.zeros(201)
    else:
        aphyCySum = np.array(aphyCy)[0]
        
    classIOPs['fluorescence'] = {'FQY': fqy,
                                 'aphyEuk' : aphyEukSum,
                                 'aphyCy' : aphyCySum}
    
    iops['Phyto'] = classIOPs  
    
    #%
    ####################### NON ALGAL PARTICLES ####################################
    #%
    # MINERALS
    classes = ['SAN1','AUS1','ICE1','KUW1','NIG1','SAH1','OAH1']
    idx440 = int(np.where(l==440)[0])
    aphy440 = iops['Phyto']['a_tot'][idx440]
    
    # from lee 2002 for case1 waters
    r1 = np.arange(0,1.05,.05)
    # r2 = np.arange(0.05,.1,.005)
    p1 = .1 + (0.5 * np.random.choice(r1) * aphy440) / (.05 + aphy440) 
    sf = np.random.choice(np.arange(0.05,.2,.01))
    anap440 = p1 * aphy440
    amin440 = sf * anap440
    
    # mineral component
    minIOPs = {}
    minIOPs['amin440'] = amin440
    minIOPs = min_iops_case1(datanap, minIOPs)    
    iops['Min'] = minIOPs 
    
    # DETRITUS
    
    detIOPs = {}
    adet440 = anap440 * (1-sf)
    detIOPs['adet440'] = adet440
    detIOPs = det_iops_case1(datanap, detIOPs)
    iops['Det'] = detIOPs
        
    #%
    ##################### CDOM ######################################################
    
    # slopes = np.random.normal(.03,.005,5000)
    # slopes = slopes[slopes > 0]
    # slope = np.random.choice(slopes)
    r1 = np.arange(0, 1.05, .05)
    p2 = 0.3 + (5.7 * np.random.choice(r1,1) * aphy440) / (0.02 + aphy440)
    ag440 = p2 * aphy440
    # ag440 = np.random.choice([.0001,])
    cdomIOPs = cdom_iops(ag440)
    iops['CDOM'] = cdomIOPs
    
    ################### DEPTH FUNCTION #############################################
    #%
    depth = np.random.choice(np.arange(10,31,1))
    s1 = np.arange(.005, .1, .005)
    s2 = np.arange(.1,.6,.05)
    s3 = np.arange(.6,1,.1)
    s = np.concatenate([s1,s2,s3])
    slope = np.random.choice(s)
    c = 30 # hypothetical (doesnt matter for xfactor)
    d = np.arange(0,depth,1)
    yfactor = []
    xfactor = []
    for k in d:
        y = c * np.exp(-slope*-(k*-1))
        x = y/c
        yfactor.append(y)
        if k == 0:
            xfactor.append(1)
        else:
            xfactor.append(x)
    
    dprops = {'Depth':d,
              'xfactor':xfactor,
              'slope':slope,
              'Dmax':d.max()}
    iops['Depth'] = dprops
    
    ##################### Benthic Reflectance #######################################
    #%
    groups = ['Bleached coral','Blue coral','Brown coral','Brown/red algae',
              'CCA/turf/MPB/cyano','Green algae','Green coral','Mud','Sand/rock',
              'Seagrass/weed']
    benthic_frxn = dirichlet_benthic(alphas,groups,benthic_lib)
    iops['Benthic'] = benthic_frxn
    
    ##################### Adjacency Reflectance ####################################
    #%
    groups = ['Ice','Manmade','Non-photo Veg','Soil','Vegetation']
    adj_frxn = dirichlet_adj(alphas,groups,adj_lib)
    iops['Adjacency'] = adj_frxn 
    
    # small water bodies
    wr = np.random.choice(np.arange(1,25,1))
    dist = np.random.choice(np.arange(.1, wr/2, .2))
    # water body radius (km)
    iops['Adjacency']['water_radius'] = wr
    # Distance to landline
    iops['Adjacency']['dist'] = np.round(dist,2)
    
    #################### ATMOSPHERE ################################################
    #%  
    atm =  {'aero': aero_lib.sample(),
            'atm_prof': np.random.choice(['afglt','afglms','afglmw','afglss','afglsw',
                                         'afglus']),
            'aero_prof': np.random.choice(['antarctic', 'continental_average',
                                          'continental_clean', 'continental_polluted',
                                          'desert',
                                          'maritime_clean', 'maritime_polluted',
                                          'maritime_tropical', 'urban']),
            'OZA': np.random.choice(range(10,55,5)),
            'OAA': np.random.choice(range(60,120,5)),
            'SZA': np.random.choice(range(5,70,5)),
            'SAA': np.random.choice(range(30,160,10)),
            'wind': np.random.choice(np.linspace(0,14,29))
           }
    
    # add altitude, water vapor, ozone, wind (for sunglint) !!!!!
    
    iops['Atm'] = atm
    
    #################### Chl Fluorescence ############################################
    
    # fqy = np.random.choice(np.linspace(.005,.015,50)).astype(np.float16)
    # qa = np.random.choice([0.3,0.4,0.5,0.6])
    # iops['chla_fluor'] = {}
    # iops['chla_fluor']['FQY'] = fqy
    # iops['chla_fluor']['Qa'] = qa
    # iops['Phyto']['chla_fluor']['FQY'] = fqy
    # iops['chla_fluor']['Qa'] = qa
    
    
    ############### ARD FORMAT #######################################################
    #%
    cols, row = dict_to_df(iops)
    
    return iops, cols, row     

            
            
                       
  
            
