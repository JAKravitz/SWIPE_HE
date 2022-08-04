#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:20:39 2022

@author: jakravit
"""
from ver3.build_case2 import build_case2
import numpy as np
from scipy.interpolate import interp1d
import pickle
import pandas as pd
import uuid
import time

# Data paths 

outpath = '/Users/jakravit/data/hydro/'
datadir = '/Users/jakravit/data/'

# phytoplankton SIOP spectral library
# path = '/nobackup/jakravit/data/EAP_phytoplankton_dataset/'
phy_library = {'Haptophytes': {}, 
               'Diatoms': {},
               'Dinoflagellates': {},
               'Cryptophytes': {},
               'Green_algae': {},
               'Cyano_blue': {},
               'Heterokonts': {},
               'Cyano_red': {},
               'Rhodophytes': {}
               }

for phy in phy_library:
    print (phy)
    with open(datadir+'EAP_phytoplankton_dataset/'+phy+'.p', 'rb') as fp:
        phy_library[phy] = pickle.load(fp)

# NAP spectral libraries
minpath = datadir+'EAP_NAP_dataset/minerals_V2.p'
with open(minpath, 'rb') as fp:
    datamin = pickle.load(fp)  

detpath = datadir+'EAP_NAP_dataset/det_V1.p'
with open(detpath, 'rb') as fp:
    datadet = pickle.load(fp) 

# Benthic library
benthic_lib = pd.read_csv(datadir+'benthic_spec_libary_FT_V4.csv')

# adjacency library
adj_lib = pd.read_csv(datadir+'adjacency_spectra_V2.csv')

# aeronet library
aero_lib = pd.read_csv(datadir+'aeronet_invdata_match.csv')

#%%
INFO = True

iops, cols, row = build_case2(phy_library, datamin, datadet, benthic_lib, adj_lib, aero_lib)
if INFO:
    print ('Chla: {} ug/L'.format(iops['Phyto']['TotChl']))
    print ('Minl: {} g/L'.format(iops['Min']['Tot_conc']))
    print ('ag440: {} m^-1'.format(iops['CDOM']['ag440']))
    print ('Max Depth: {} m'.format(iops['Depth']['Dmax']))
    
    
#%
#% TOTAL IOPS

# pre - assign
a_tot=[]; a_totphy=[]; a_totmin=[]; a_totdet=[]; a_totdom=[]
c_tot=[]; c_totphy=[]; c_totmin=[]; c_totdet=[]; c_totdom=[]
bb_tot=[]; bb_totphy=[]; bb_totmin=[]; bb_totdet=[]; bb_totdom=[]
chlProfile = []

# depth
depth = np.array([0,1,2,3,4,5,6,7,8,9])
depth = depth.reshape(-1, 1)

# chl profile
chlProfile.append([iops['Phyto']['TotChl']] * 10)
chlProfile = np.asarray(chlProfile).reshape(-1, 1)
foot = np.hstack((-1,0))
chlzProfile = np.hstack((depth,chlProfile))
chlzProfile = np.vstack((chlzProfile,foot))

for i in range(10):

    # absorption
    atotphy = iops['Phyto']['a_tot']
    atotmin = iops['Min']['a_tot']
    atotdet = iops['Det']['a_tot']
    atotdom = iops['CDOM']['a_tot'][63:]
    atot = atotphy + atotmin + atotdet + atotdom
    a_tot.append(atot)
    a_totphy.append(atotphy)
    a_totmin.append(atotmin)
    a_totdet.append(atotdet)
    a_totdom.append(atotdom)
    
    # Scatter
    btotphy = iops['Phyto']['b_tot']
    btotmin = iops['Min']['b_tot']
    btotdet = iops['Det']['b_tot']
    btot = btotphy + btotmin + btotdet
    
    # attenuation
    ctotphy = atotphy + btotphy
    ctotmin = atotmin + btotmin
    ctotdet = atotdet + btotdet
    ctotdom = atotdom # no scattering
    ctot = atot + btot
    c_tot.append(ctot)
    c_totphy.append(ctotphy)
    c_totmin.append(ctotmin)
    c_totdet.append(ctotdet)
    c_totdom.append(ctotdom)

    # backscattering
    bbtotphy = iops['Phyto']['bb_tot']
    bbtotmin = iops['Min']['bb_tot']
    bbtotdet = iops['Det']['bb_tot']
    bbtot = bbtotphy + bbtotmin + bbtotdet
    bb_tot.append(bbtot)
    bb_totphy.append(bbtotphy)
    bb_totmin.append(bbtotmin)
    bb_totdet.append(bbtotdet)

#% PREPARE FOR WRITING TO .TXT FILE IN HYDROLIGHT FORMAT

# lambda
l = np.arange(400,902.5,2.5)

llen = len(l)
line11 = np.hstack((llen,l)).reshape(1,-1)
# AC data
foot = np.hstack((-1,[0]*(llen*2)))
# only phy
acProfile_phy = np.hstack((depth,a_totphy,c_totphy))
acProfile_phy = np.vstack((acProfile_phy,foot))
# only min
acProfile_min = np.hstack((depth,a_totmin,c_totmin))
acProfile_min = np.vstack((acProfile_min,foot))
# only det
acProfile_det = np.hstack((depth,a_totdet,c_totdet))
acProfile_det = np.vstack((acProfile_det,foot))
# only dom
acProfile_dom = np.hstack((depth,a_totdom,c_totdom))
acProfile_dom = np.vstack((acProfile_dom,foot))
# all components
acProfile = np.hstack((depth,a_tot,c_tot))
acProfile = np.vstack((acProfile,foot))   

# bb data
foot = np.hstack((-1,[0]*(l)))
# only phy
bbProfile_phy = np.hstack((depth,bb_totphy))
bbProfile_phy = np.vstack((bbProfile_phy,foot))
# only min
bbProfile_min = np.hstack((depth,bb_totmin))
bbProfile_min = np.vstack((bbProfile_min,foot))
# only det
bbProfile_det = np.hstack((depth,bb_totdet))
bbProfile_det = np.vstack((bbProfile_det,foot))
# all components
bbProfile = np.hstack((depth,bb_tot))
bbProfile = np.vstack((bbProfile,foot))

#% WRITE TO TXT
import string
import random
import os 

def random_uid():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=10))
uid = random_uid()

# sname.append(uid)
os.mkdir(outpath+'{}/'.format(uid))
iopzpath = outpath + '{}/'.format(uid)

# acData

# fnames
fnames = {'total': {'acfname': 'acDatatot_' + uid + '.txt',
                    'bbfname': 'bbDatatot_' + uid + '.txt',
                    'acdata': acProfile,
                    'bbdata': bbProfile},
          'phyto': {'acfname': 'acDataphy_' + uid + '.txt',
                    'bbfname': 'bbDataphy_' + uid + '.txt',
                    'acdata': acProfile_phy,
                    'bbdata': bbProfile_phy},
          'min': {'acfname': 'acDatamin_' + uid + '.txt',
                  'bbfname': 'bbDatamin_' + uid + '.txt',
                  'acdata': acProfile_min,
                  'bbdata': bbProfile_min},
          'det': {'acfname': 'acDatadet_' + uid + '.txt',
                  'bbfname': 'bbDatadet_' + uid + '.txt',
                  'acdata': acProfile_det,
                  'bbdata': bbProfile_det},
          }

# write chl data
header = '\n'.join(['CHL CONC DATA PROFILE',
                    'Version 3',
                    '#',
                    '#','#','#','#','#','#','#\n'])
with open(os.path.join(iopzpath,'chlzData_{}.txt'.format(uid)),'w') as f:
    f.writelines(header)
    np.savetxt(f,chlzProfile,delimiter='\t')
    
# write AC data
header = '\n'.join(['TOTAL AC DATA PROFILE',
                    'Version 3',
                    'Does not include water!',
                    '#','#','#','#','#','#','#\n'])
for fname in fnames:
    with open(os.path.join(iopzpath,fnames[fname]['acfname']),'w') as f:
        f.writelines(header)
        np.savetxt(f,line11,fmt='%d',delimiter='\t')
        np.savetxt(f,fnames[fname]['acdata'],delimiter='\t')

# write bb data
header = '\n'.join(['TOTAL BB DATA PROFILE',
                    'Version 3',
                    'Does not include water!',
                    '#','#','#','#','#','#','#\n'])
for fname in fnames:
    with open(os.path.join(iopzpath,fnames[fname]['bbfname']),'w') as f:
        f.writelines(header)
        np.savetxt(f,line11,fmt='%d',delimiter='\t')
        np.savetxt(f,fnames[fname]['bbdata'],delimiter='\t')

#% PLOT IOPS
import matplotlib.pyplot as plt

for k in ['a','bb']:
    fig, ax = plt.subplots()
    # plot phytos
    ax.plot(l, iops['Phyto']['Haptophytes']['{}_tot'.format(k)], label='Hapto')
    ax.plot(l, iops['Phyto']['Diatoms']['{}_tot'.format(k)], label='Diat')
    ax.plot(l, iops['Phyto']['Dinoflagellates']['{}_tot'.format(k)], label='Dino')
    ax.plot(l, iops['Phyto']['Cryptophytes']['{}_tot'.format(k)], label='Crytpo')
    ax.plot(l, iops['Phyto']['Green_algae']['{}_tot'.format(k)], label='Grn')
    ax.plot(l, iops['Phyto']['Cyano_blue']['{}_tot'.format(k)], label='Cy_bl')
    ax.plot(l, iops['Phyto']['Heterokonts']['{}_tot'.format(k)], label='Hetero')
    ax.plot(l, iops['Phyto']['Cyano_red']['{}_tot'.format(k)], label='Cy_rd')
    ax.plot(l, iops['Phyto']['Rhodophytes']['{}_tot'.format(k)], label='Rhodo')
    ax.plot(l ,iops['Phyto']['{}_tot'.format(k)], label='TOT phyto')
    # minerals
    ax.plot(l, iops['Min']['{}_tot'.format(k)], label='TOT Min')
    # detritus
    ax.plot(l, iops['Det']['{}_tot'.format(k)], label='TOT det')
    # cdom
    if k == 'a':
        ax.plot(l, a_totdom[0], label='TOT CDOM')
        ax.plot(l, a_tot[0], label='TOTAL')
    else:
        ax.plot(l, bb_tot[0], label='TOTAL')
    # total
    
    # plot info
    ax.legend(loc='right')
    ax.set_xlim(400,900)
    ax.set_ylabel('m$^{-1}$')
    ax.set_xlabel('lambda')
    ax.set_title(k)
    # fig.savefig(outpath + '{}/{}.png'.format(fname,k), 
    #             dpi= 300, bbox_inches='tight')