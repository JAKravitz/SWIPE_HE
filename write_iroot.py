#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:47:16 2021

@author: jakravit
"""
import os
import numpy as np
import pandas as pd
import shutil
import io

# filepaths
runlist_title = 'case2_test_runlist'
datapath = '/Users/jakravit/data/hydro/rmfrocm50n/'
# runlist = '/Users/jkravz311/.wine/drive_c/HE52/run/'
# batchPath = '/Users/jkravz311/.wine/drive_c/HE52/run/batch/'

# solar zenith
thetaData = np.arange(5,70,5)
theta = np.random.choice(thetaData)

# wind speed (m/s)
windData = np.arange(0,15,.5)
ws = np.random.choice(windData)

##
''' --------- BEGIN WRITE NEW IROOT FILE --------- '''

# for i, name in enumerate(snames):
    
# create .txt filename
fname0 = datapath.split('/')[5]
fname1 = 'I_' + fname0 + '.txt'
fname = os.path.join(datapath, fname1)

''' --------- SPECIFY LINES FOR IROOT FILE --------- '''

# LINE 1 - RECORD 1: DEFAULT PARAMETERS
# Specify quantum yield based on chl concentration
line1 = np.array([0, 400, 700, .01, 488, 0.00026, 1, 5.3]).reshape(1, -1)

# LINE 2 - RECORD 2: RUN TITLE
line2 = fname1 + '\n'

# LINE 3 - RECORD 3: ROOTNAME
line3 = fname0 + '\n'

# LINE 4 - RECORD 4A: OUTPUT OPTIONS
line4 = np.array([-1, 0, 0, 1, 0, 1]).reshape(1, -1)

# LINE 5 - RECORD 4B: MODEL OPTIONS
line5 = np.array([3, 1, -1, 4, 0]).reshape(1, -1)

# LINE 6 - RECORD 5A: NUMBER OF COMPONENTS
line6 = np.array([2, 2]).reshape(1, -1)

# LINE 7 - RECORD 5B: COMPONENT CONCENTRATIONS
line7 = np.array([0, 0]).reshape(1, -1)

# LINE 8-11 - RECORD 5C: SPECIFIC ABSORPTION PARAMETERS
# fluorescence off
line8_11 = np.array([[0, 3, 440, 0.1, 0.014],  # water
                     [2, -666, 440, 0.1, 0.014]]).reshape(2, -1)  # total

# LINE 12-15 - RECORD 5D: SPECIFIC ABSORPTION DATA FILE NAMES
line12_15 = '\n'.join(['C:\HE52_data\SIOPs\H2OabDefaults_FRESHwater.txt',
                       'dummyastar.txt\n'])

# LINE 16-17 - RECORD 5E: SPECIFIC SCATTERING PARAMTERS
line16_17 = np.array([[0, -999, -999, -999, -999, -999],
                      [-666, -999, -999, -999, -999, -999]]).reshape(2, -1)

# LINE 18-19 - RECORD 5F: SPECIFIC SCATTERING DATA FILE NAMES
line18_19 = '\n'.join(['bstarDummy.txt',
                       'dummybstar.txt\n'])

# LINE 20-21 - RECORD 5G: TYPES OF CONCENTRATIONS AND PHASE FUNCTIONS
line20 = np.array([[0, 0, 550, 0.01, 0]]).reshape(1, -1)
line21 = np.array([-2, 0, 0, 0, 0]).reshape(1, -1)

# LINE 22-23 - RECORD 5H: PHASE FUNCITON FILE NAMES
line22_23 = '\n'.join(['pureh2o.dpf',
                       'isotrop.dpf\n'])

# LINE 24-75 - RECORD 6: WAVELENGTHS
# 400:900 in 2.5 nm intervals = 200 'bands'
line24 = np.array([200]).reshape(1, -1)
line25_73 = np.arange(400, 900, 2.5).reshape(20, -1)
line75 = np.array([900]).reshape(1, -1)

# LINE 76 - RECORD 7: INELASTIC SCATTERING
# AND LINE 77 - RECORD 8: SKY MODEL
line76_77 = np.array([[0,0,0,0,4],
                      [2,3,theta,0,0]]).reshape(2,-1)

# REST OF LINES DO NOT NEED ATTENTION AND CAN STAY THE SAME
# UNLESS YOU WANT TO CHANGE ATMOSPHERICS, DEPTHS, PATHS TO IOPS, ETC..
line78 = np.array([-1,0,0,29.92,1,80,2.5,15,ws,300]).reshape(1,-1) # atmospherics
line79 = np.array([ws,1.34,20,35]).reshape(1,-1) # surface info
line80 = np.array([0,0]).reshape(1,-1) # bottom reflectance
line81 = np.array([0,11,0,1,2,3,4,5,6,7,8,9,10]).reshape(1,-1) # depths

# REST OF LINES ARE DATA FILE PATHS (82-93)...
line82_93 = '\n'.join(['C:\HE52_data\SIOPs\H2OabDefaults_FRESHwater.txt',
                       '1',
                       '{}acDatatot_{}.txt'.format(datapath,fname0),
                       'dummyFilteredAc9.txt',
                       '{}bbDatatot_{}.txt'.format(datapath,fname0),
                       '{}chlzData_{}.txt'.format(datapath,fname0),
                       'dummyCDOMdata.txt',
                       'dummyR.bot',
                       'dummydata.txt',
                       'dummyComp.txt',
                       'C:\HE52_data\Ed_files\Ed_inputs.txt',
                       '..\data\MyBiolumData.txt'])    

#
''' --------- WRITE LINES TO IROOT FILE --------- '''

with open(fname, 'w+') as f:
    np.savetxt(f, line1, fmt='%d,%d,%d,%1.3f,%d,%1.5f,%d,%1.1f')
    f.write(line2)
    f.write(line3)
    np.savetxt(f, line4, delimiter=',', fmt='%d')
    np.savetxt(f, line5, delimiter=',', fmt='%d')
    np.savetxt(f, line6, delimiter=',', fmt='%d')
    np.savetxt(f, line7, delimiter=',', fmt='%d')
    np.savetxt(f, line8_11, fmt='%d,%d,%d,%1.1f,%1.3f')
    f.writelines(line12_15)
    np.savetxt(f, line16_17, delimiter=',', fmt='%d')
    f.writelines(line18_19)
    np.savetxt(f, line20, fmt='%d,%d,%d,%1.2f,%d')
    np.savetxt(f, line21, delimiter=',', fmt='%d')
    f.writelines(line22_23)
    np.savetxt(f, line24, fmt='%d')
    np.savetxt(f, line25_73, delimiter=',', fmt='%1.1f')
    #np.savetxt(f, line74, delimiter=',', fmt='%d')
    np.savetxt(f, line75, fmt='%d')
    np.savetxt(f, line76_77, delimiter=',', fmt='%d')
    np.savetxt(f, line78, fmt='%d,%d,%d,%2.2f,%d,%d,%1.1f,%d,%d,%d')
    np.savetxt(f, line79, fmt='%d,%1.2f,%d,%d')
    np.savetxt(f, line80, delimiter=',', fmt='%d')
    np.savetxt(f, line81, delimiter=',', fmt='%d')
    f.writelines(line82_93)

    # WRITE FILE TO RUNLIST.TXT FOR HE5 BATCH PROCESSING
    with io.open(os.path.join(datapath, '{}.txt'.format(runlist_title)), 'a+') as r:
        r.write(str(fname1) + '\r\n')

 
    