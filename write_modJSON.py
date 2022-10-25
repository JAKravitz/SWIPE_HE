#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def modjson(tp, iops, aero_lib, batch_name, uid, modPath):

    for i,k in enumerate(tp['MODTRAN']):

        # randomly tweek atm parameters for each case
        aero = aero_lib.sample(n=1)
        # aero = aero[0]
        atm_prof = np.random.choice(['ATM_TROPICAL','ATM_MIDLAT_SUMMER','ATM_MIDLAT_WINTER',
                                      'ATM_SUBARC_SUMMER','ATM_SUBARC_WINTER', 'ATM_US_STANDARD_1976'])
        aero_prof = np.random.choice(['AER_RURAL', 'AER_RURAL_DENSE', 'AER_MARITIME_NAVY', 'AER_MARITIME',
                                       'AER_URBAN','AER_TROPOSPHERIC', 'AER_DESERT'])
        OZA = np.random.choice(range(10,60,5))
        OAA = np.random.choice(range(60,120,5))
        SAA = iops['Geo']['SAA']
        SZA = iops['Geo']['SZA']
        SRA = SAA - OAA
        if SRA < 0:
            SRA = SRA + 360
        GNDALT = np.random.choice(np.linspace(0,4,40)),
        GNDALT = round(GNDALT[0],2)
        Wv = np.random.choice(np.linspace(0, 5, 100))
        Wv = round(Wv,2)
        vis = (aero['AOD_Coincident_Input[440nm]'] + aero['AOD_Coincident_Input[675nm]']) / 2 
        vis = round(vis.values[0],2)

        # name
        k['MODTRANINPUT']['NAME'] = f'a{i}'
        # description - pass
        # case
        k['MODTRANINPUT']['CASE'] = i
        # RT options
        k['MODTRANINPUT']['RTOPTIONS'] = {
            'MODTRN': 'RT_MODTRAN',
            'LYMOLC': False,
            'T_BEST': False,
            'IEMSCT': 'RT_SOLAR_AND_THERMAL',
            'IMULT': 'RT_DISORT',
            'DISALB': True,
            'NSTR': 8,
            'SOLCON': 0.0}
        # atmosphere
        k['MODTRANINPUT']['ATMOSPHERE'] = {
            'MODEL': atm_prof,
            'CO2MX': 390.0,
            'H2OSTR': Wv,
            'H2OUNIT': 'a',
            'O3STR': 0.29,
            'O3UNIT': 'a',}
        # aerosols
        k['MODTRANINPUT']['AEROSOLS'] = {
            'CDASTM': 'B',
            'ASTMX': aero['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values[0],
            'IHAZE': aero_prof,
            'ARUSS': 'USS',
            'VIS': -vis,
            'IPH': 0,
            'SSALB': {
                'NSSALB': 4,
                'AWAVLN': [0.400, 0.675, 0.870, 1.02],
                'ASSALB': [aero['Single_Scattering_Albedo[440nm]'].values[0], aero['Single_Scattering_Albedo[675nm]'].values[0],
                           aero['Single_Scattering_Albedo[870nm]'].values[0], aero['Single_Scattering_Albedo[1020nm]'].values[0]]},
            'IREGSPC': [{'IREG': 1,
                'NARSPC': 7,
                'AWCCON': 0.0,
                'AERNAM': '',
                'VARSPC': [440.0, 675.0, 870.0, 1020.0, 0.0, 0.0, 0.0],
                'EXTC': [aero['AOD_Extinction-Total[440nm]'].values[0], aero['AOD_Extinction-Total[675nm]'].values[0], 
                       aero['AOD_Extinction-Total[870nm]'].values[0], aero['AOD_Extinction-Total[1020nm]'].values[0], 0.0, 0.0, 0.0],
                'ABSC': [aero['Absorption_AOD[440nm]'].values[0], aero['Absorption_AOD[675nm]'].values[0], 
                       aero['Absorption_AOD[870nm]'].values[0], aero['Absorption_AOD[1020nm]'].values[0], 0.0, 0.0, 0.0],
                'ASYM': [aero['Asymmetry_Factor-Total[440nm]'].values[0], aero['Asymmetry_Factor-Total[675nm]'].values[0], 
                       aero['Asymmetry_Factor-Total[870nm]'].values[0], aero['Asymmetry_Factor-Total[1020nm]'].values[0], 0.0, 0.0, 0.0]}]}
        # geometries
        k['MODTRANINPUT']['GEOMETRY'] = {
            'ITYPE': 3,
            'H1ALT': 18.0,
            'H2ALT': GNDALT + .0001,
            'OBSZEN': 180 - OZA,
            'IPARM': 12,
            'PARM1': SRA,
            'PARM2': SZA}
        # surface
        k['MODTRANINPUT']['SURFACE'] = {
            'SURFTYPE': 'REFL_LAMBER_MODEL',
            'GNDALT': GNDALT,
            'NSURF': 2,
            'SURFP': {'CSALB': 'LAMB_OCEAN_WATER'},
            'SURFA': {'CSALB': 'LAMB_OCEAN_WATER'}}
        # spectral
        k['MODTRANINPUT']['SPECTRAL'] = {
            'V1': 390.0,
            'V2': 962.0,
            'DV': 0.05,
            'FWHM': 0.1,
            'YFLAG': 'R',
            'XFLAG': 'N',
            'FLAGS': 'NTAA  T',
            'MLFLX': 1}
        # file options
        k['MODTRANINPUT']['FILEOPTIONS'] = {
            'JSONOPT': 'WRT_OUTPUT', 'CSVPRNT': f'{modPath}{batch_name}_{uid}.csv', 'JSONPRNT': f'{modPath}{batch_name}_{uid}.json'}
    
    # save json
    with open(f'{modPath}{batch_name}_{uid}.json', 'w') as f:
        json.dump(tp, f, cls=NpEncoder)