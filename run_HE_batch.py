# -*- coding: utf-8 -*-
"""
Script to run all Iroot.txt files in runlist.txt in HE52.
Script should be kept in and run from C:\HE52\run folder

@author: jeremy.kravitz@gmail.com
"""
#%%
import os
import pandas as pd
import numpy as np
from optparse import OptionParser
from append_results import get_rrs
import subprocess

def main(fname):
    #fname = 'runlist.txt's
    # listPath = '/nobackup/jakravit/data/HE60/run/{0}'.format(fname)
    outPath = '/nobackup/jakravit/data/HE60/output/EcoLight/excel/'
    batchPath = '/nobackup/jakravit/data/HE60/batch_data/'
    # create empty outputs df
    output = pd.DataFrame()
    outputCols = []
    def col_append (name):
        l = np.arange(400, 902.5, 2.5).astype(str) 
        l2 = ['{}_'.format(name) + lx for lx in l] 
        return l2
    outputCols.append(col_append('rrs'))
    outputCols.append(col_append('rrsTot'))
    outputCols.append(col_append('ed'))
    
    runlist = pd.read_csv(fname,header=None)
    
    for run in runlist[0]:
        print ('\nPROCESSING {0} ...\n'.format(run))
        # os.system('wine /Users/jkravz311/.wine/drive_c/HE52/Code/Ecolight/mainEL_stnd.exe < batch/{0}'.format(run))
        subprocess.call('/nex/modules/m/hydrolight/HE60/backend/./EcoLight6 < /nobackup/jakravit/data/HE60/run/batch/{}'.format(run))
        
        # get run info
        info = run.split('_')
        batch_name = info[1]
        uid = info[2]
        outFile = outPath + 'MI_{}_{}'.format(batch_name, uid)
        
        # get output data
        rrs, rrstot, rrsg, rrsgmean, ed = get_rrs(outFile)
        
        # create new sunglint sim uid
        uidSG = uid + '_SG'
        
        # import table ARD data and append SG info  
        batchFile = batchPath + '{}_inputs.csv'.format(batch_name)
        batch_data = pd.read_csv(batchFile)
        batch_data.at[uidSG,'Sg_amplitude'] = rrsgmean
        
        # append output data
        row1 = np.concatenate([rrs,ed])
        row2 = np.concatenate([rrstot,ed])
        output[uid] = row1
        output[uidSG] = row2
        
        # fluorescence
    
    batch_data.to_csv(batchPath + '{}_inputs.csv'.format(batch_name))
    output.to_csv(batchPath + '{}_outputs.csv'.format(batch_name))
        

#%%
if __name__ == '__main__':
    
    info = ('Run multiple Iroot files from runlist.txt.')
    
    usage = 'usage: python <script> arg1 [options]'
    
    parser = OptionParser(usage = usage,
                          description = info)
        
    (options, args) = parser.parse_args()
    
    if args == []:
        print ('\nERROR: need to supply runlist.txt.')
        parser.print_help()

    for fname in args:
         main(fname) 