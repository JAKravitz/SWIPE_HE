# -*- coding: utf-8 -*-
"""
Script to run all Iroot.txt files in runlist.txt in HE52.
Script should be kept in and run from C:\HE52\run folder

@author: jeremy.kravitz@gmail.com
"""
#%%
def get_rrs (file):
    rrsheader = ['in air', 'Rrs']
    
    line_num = 0
    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_num += 1
        if all(x in line.strip().replace('"', '').replace("'", "") for x in rrsheader):
            header = line_num
            l = line
            # print (line_num, line)
        else:
            pass
        
    lines = np.arange(header,header+200,1)
    end = header+200
    rrs = []
    lw = []
    ed = []
    wl = []
    lu = []

    i = 0
    with open(file, "r+") as fp:
        # access each line
        while True:
            line = fp.readline()
            # check line number
            if i in lines:
                wl.append(line.strip().split('   ')[0] )
                rrs.append(line.strip().split('   ')[1] )
                ed.append(line.strip().split('   ')[2] )
                lw.append(line.strip().split('   ')[3])
                lu.append(line.strip().split('   ')[4] )
            # line number greater than needed exit the loop
            # lines[-1] give last item from list
            if i > end:
                break;
            i = i + 1
    
    wl = np.array(wl).astype(float)
    rrs = np.array(rrs).astype(float)
    lw = np.array(lw).astype(float)
    lu = np.array(lu).astype(float)
    ed = np.array(ed).astype(float)
    lg = lu-lw
    rrsg = lg / ed
    rrsgmean = rrsg.mean()
    rrstot = lu / ed
    
    return rrs, rrstot, rrsg, rrsgmean, ed 

# def fluorescence_calc ():
#     kparheader = ['in air', 'Rrs']
    
#     line_num = 0
#     with open(file, 'r') as f:
#         lines = f.readlines()

#     for line in lines:
#         line_num += 1
#         if all(x in line.strip().replace('"', '').replace("'", "") for x in rrsheader):
#             header = line_num
#             l = line
#             # print (line_num, line)
#         else:
#             pass    