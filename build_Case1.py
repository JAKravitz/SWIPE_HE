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

def build_Case1(phy_library, datamin, datadet, benthic_lib, adj_lib, aero_lib):
    
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
    groups = ['Haptophytes','Diatoms','Dinoflagellates','Cryptophytes',
              'Green_algae','Cyano_blue','Heterokonts','Cyano_red','Rhodophytes']
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
    sigma,scale = lognorm_params(.05,.15)
    chlDist = lognorm_random(sigma, scale, 20000)
    chl = round(np.random.choice(chlDist), 3)
    
    # phyto classes
    classIOPs = {}
    classIOPs['TotChl'] = chl
    classIOPs = phyto_iops_case1(phyto_class_frxn, phy_library, classIOPs)
    iops['Phyto'] = classIOPs  
    
    #%
    ####################### NON ALGAL PARTICLES ####################################
    #%
    # MINERALS
    idx440 = int(np.where(l==440)[0])
    run_mins = random.choices(list(datamin.keys()), k=2) # allows repeats
    fx = np.random.choice(frxns)
    min_frxn = {run_mins[0]+'1': fx,
                run_mins[1]+'2': 1-fx}
    
    aphy440 = iops['Phyto']['a_tot'][idx440]
    
    # from lee 2002 for case1 waters
    r1 = np.arange(0,1.05,.05)
    r2 = np.arange(0.05,.1,.005)
    p1 = .1 + (0.5 * np.random.choice(r1) * aphy440) / (np.random.choice(r2) + aphy440) 
    sf = np.random.choice(np.arange(0.05,.2,.01))
    anap440 = p1 * aphy440
    amin440 = sf * anap440
    
    # mineral component
    minIOPs = {}
    minIOPs['amin440'] = amin440
    minIOPs = min_iops_case1(min_frxn, datamin, minIOPs)    
    iops['Min'] = minIOPs 
    
    # DETRITUS
    
    detIOPs = {}
    adet440 = anap440 * (1-sf)
    detIOPs['adet440'] = adet440
    detIOPs = det_iops_case1(datadet, detIOPs)
    iops['Det'] = detIOPs
        
    #%
    ##################### CDOM ######################################################
    
    slopes = np.random.normal(.03,.005,5000)
    slopes = slopes[slopes > 0]
    slope = np.random.choice(slopes)
    r1 = np.arange(0, 1.05, .05)
    p2 = 0.3 + (5.7 * np.random.choice(r1,1) * aphy440) / (0.02 + aphy440)
    ag440 = p2 * aphy440
    cdomIOPs = cdom_iops(ag440)
    iops['CDOM'] = cdomIOPs
    
    ################### DEPTH FUNCTION #############################################
    #%
    depth = np.random.choice(np.arange(1,21,1)) * -1
    s1 = np.arange(.005, .1, .005)
    s2 = np.arange(.1,.6,.05)
    s3 = np.arange(.6,1,.1)
    s = np.concatenate([s1,s2,s3])
    slope = np.random.choice(s)
    c = 30 # hypothetical (doesnt matter for xfactor)
    d = np.arange(0,depth,-.5)
    yfactor = []
    xfactor = []
    for k in d:
        y = c * np.exp(-slope*-k)
        x = y/c
        yfactor.append(y)
        if k == 0:
            xfactor.append(0)
        else:
            xfactor.append(x)
    
    dprops = {'Depth':d,
              'xfactor':xfactor,
              'slope':slope,
              'Dmax':d.min()*-1}
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
            'VZA': np.random.choice(range(10,50)),
            'VAA': np.random.choice(range(60,120)),
            'wind': np.random.choice(np.linspace(0,14,29))
           }
    
    # add altitude, water vapor, ozone, wind (for sunglint) !!!!!
    
    iops['Atm'] = atm
    
    ############### ARD FORMAT ###########################################################
    #%
    cols, row = dict_to_df(iops)
    
    return iops, cols, row     

            
            
                       
  
            
