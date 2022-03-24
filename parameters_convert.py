#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:00:04 2021

@author: jolenebritton
"""

import numpy as np

def parameter_calc(len_scalar, time_scalar, amount_scalar):
    """

    Parameters
    ----------
    len_scalar : int
        if 1, then units are microns (um)
        elseif 1000, then units are milimeters (mm)
    time_scalar : int
        if 1, then units are seconds (s)
        elseif 60, then units are minutes (min)
        elseif 3600, then units are hours (hours)
    amount_scalar : int
        if 1, then units are moles (mol)
        elseif 1000, then units are milimoles (mmol)
        
    Returns
    -------
    None.

    """

    # --- HYPHAL SPATIAL PARAMETERS ---
    # Diameter of hyphae (um if len_scalar = 1)
    diam = 5 / len_scalar
    # Cross-sectional area
    cross_area = np.pi * ( 0.5 * diam )**2
    # Compartment length (um if len_scalar = 1)
    comp_len = 6 * diam
    # Compartment volume
    comp_vol = cross_area * comp_len
    
    # --- EXTERNAL GRID PARAMETERS ---
    # Length & width of grid cell (um if len_scalar = 1)
    grid_len = 20 / len_scalar
    grid_height = 20 / len_scalar
    # Grid cell volume
    grid_vol = grid_len*grid_len*grid_height
    
    # --- ANGLES ---
    # Mean & standard deviation of branching angles (degrees)
    branch_ang_mean = 32.3
    branch_ang_std = 2.1
    # Mean & standard devidation of extension angles (degrees)
    extend_ang_mean = 0
    extend_ang_std = 1
    
    # --- DIFFUSION & VELOCITIES ---
    # Diffusion coefficient of glucose in agar (um^2 per s if len_scalar = 1, time_scalar = 1)
    diff_gluc = 577 / ( len_scalar**2 ) * time_scalar
    # Velocity (um per s if len_scalar = 1, time_scalar = 1)
    vel_cellwall = 1 / len_scalar * time_scalar
    
    # --- GROWTH RATES ---
    # Max rate of elongation (um per s if len_scalar = 1, time_scalar = 1)
    growth_max = 0.005 / len_scalar * time_scalar
    growth_max_nc = 1.22222222 / len_scalar * time_scalar
    growth_ratio = growth_max / growth_max_nc
    # Molecular weight of cell wall (mg per mole)
    mw_cellwall = 484.17 / amount_scalar # mw_cellwall is in gm/mol or mg/mmol before modification
    # Fraction of hyphal  mass that is cell wall (unitless)
    f_cellwall = 0.15
    f_drywet = 0.21
    # Density of (soil) fungi (mg per um^3 if len_scalar = 1)
    density = 1.09e-9 * ( len_scalar**3 )
    # Hyphal length per mass of cell wall consumed
    gamma = ( f_cellwall * f_drywet ) / ( density * cross_area )
    # Monod constant for growth rate
    # monod_growth_const = 
    
    # --- UPTAKE RATES - AVG ---
    # Max rate of uptake - in terms of dry mass (mol per s per mg if time_scalar = 1)
    uptake_max_dry = 7.83333e-10 * time_scalar * amount_scalar
    # Ratio of protein to dry mass - N. crassa (unitless)
    f_pd = 0.3
    # Biomaterial mass
    bio_mass = comp_vol * density
    # Max rate of uptake - in terms of protein 
    uptake_max_pro_nc = uptake_max_dry * ( 1 / f_pd ) * bio_mass
    # Scaled rate for lacarria
    uptake_max_pro = growth_ratio * uptake_max_pro_nc
    # Initial concentration amount (mol per um^3)
    gluc_conc = 2e-17 * ( len_scalar**3 ) * amount_scalar
    # Initial concentration in a grid cell
    grid_conc = grid_vol * gluc_conc
    # Time to consume all in a grid cell
    consump_time = grid_conc / uptake_max_pro
    
    # --- UPTAKE RATES - REPRESSED ---
    # Max rate of uptake - in terms of dry mass (mol per s per mg if time_scalar = 1)
    rep_uptake_max_dry = 4e-10 * time_scalar * amount_scalar
    # Max rate of uptake - in terms of protein 
    rep_uptake_max_pro_nc = rep_uptake_max_dry * ( 1 / f_pd ) * bio_mass
    # Scaled rate for lacarria
    rep_uptake_max_pro = growth_ratio * rep_uptake_max_pro_nc
    # Time to consume all in a grid cell
    rep_consump_time = grid_conc / rep_uptake_max_pro
    
    # --- UPTAKE RATES - DEREPRESSED ---
    # Max rate of uptake - in terms of dry mass (mol per s per mg if time_scalar = 1)
    derep_uptake_max_dry = 9.5e-10 * time_scalar * amount_scalar
    # Max rate of uptake - in terms of protein 
    derep_uptake_max_pro_nc = derep_uptake_max_dry * ( 1 / f_pd ) * bio_mass
    # Scaled rate for lacarria
    derep_uptake_max_pro = growth_ratio * derep_uptake_max_pro_nc
    # Time to consume all in a grid cell
    derep_consump_time = grid_conc / derep_uptake_max_pro
    
    
    # --- TEMPORAL PARAMS ---
    # dt < (dx^2 + dy^2) / (8*D) = dy^2 / (4*D) since dx=dy
    time_step = 0.99* ( grid_len )**2 / ( 4 * diff_gluc )
    steps_in_a_day = (86400 / time_scalar) / time_step
    
    # --- PRINT VALUES ---
    print('----------------------------------------------------')
    if len_scalar == 1:
        print('LENGTH UNITS: microns')
        l_str = 'um'
    elif len_scalar == 1000:
        print('LENGTH UNITS: mm')
        l_str = 'mm'
        
    if time_scalar == 1:
        print('TIME UNITS: seconds')
        t_str = 's'
    elif time_scalar == 60:
        print('TIME UNITS: minutes')
        t_str = 'min'
    elif time_scalar == 3600:
        print('TIME UNITS: hours')
        t_str = 'hr'
        
    if amount_scalar == 1:
        print('AMOUNT UNITS: moles')
        a_str = 'mol'
    elif amount_scalar == 1000:
        print('AMOUNT UNITS: milimoles')
        a_str = 'mmol'
        
    print('MASS UNITS: mg')
    print('----------------------------------------------------')
    print('Diameter:                  ', diam, l_str)
    print('Cross-Sectional Area:      ', cross_area, l_str, '^2')
    print('Hyphal Compartment Length: ', comp_len, l_str)
    print('Hyphal Compartment Volume: ', comp_vol, l_str, '^3')
    print(' ')
    print('Grid Cell Length/Width:    ', grid_len, l_str)
    print('Grid Cell Height:          ', grid_height, l_str)
    print('Grid Cell Volume:          ', grid_vol, l_str, '^3')
    print(' ')
    print('Branching Angles:           {}+/-{} degrees'.format(branch_ang_mean, branch_ang_std))
    print('Extension Angles:           {}+/-{} degrees'.format(extend_ang_mean, extend_ang_std))
    print(' ')
    print('Glucose Diffusion Coeff:   ', diff_gluc, l_str, '^2', t_str, '^-1')
    print('Cell Wall Velocity:        ', vel_cellwall, l_str, t_str, '^-1')
    print(' ')
    print('Max Growth Rate (L.B.):    ', growth_max, l_str, t_str, '^-1')
    print('Max Growth Rate (N.C.):    ', growth_max_nc, l_str, t_str, '^-1')
    print('Growth Ratio:              ', growth_ratio)
    print('C.W. Molecular Weight:     ', mw_cellwall, 'mg ', a_str, '^-1')
    print('C.W. to Hyphal Dry Mass:   ', f_cellwall)
    print('Dry to Wet Hyphal Mass:    ', f_drywet)
    print('Hyphal Density:            ', density, 'mg', l_str, '^-3')
    print('Length per C.W. Consumed:  ', gamma, l_str, 'mg^-1')
    print('Length per cell:            {:e} {} {}^-1'.format(gamma*mw_cellwall, l_str, a_str))
    print(' ')
    print('Protein to Dry Mass:       ', f_pd)
    print('Biomaterial Mass:          ', bio_mass, 'mg protein')
    print('Initial Glucose Conc       ', gluc_conc, a_str, l_str, '^-3')
    print('Initial Glucose Grid Cell: ', grid_conc, a_str)
    print('AVG STATE')
    print('Max Uptake - Dry Mass:     ', uptake_max_dry, a_str, t_str, '^-1 (mg dry mass)^-1')
    print('Max Uptake - Protein(N.C.):', uptake_max_pro_nc, a_str, t_str, '^-1')
    print('Max Uptake - Protein(L.B.):', uptake_max_pro, a_str, t_str, '^-1')
    print('Consumption Time:          ', consump_time, t_str)
    print('REPRESED STATE')
    print('Max Uptake - Dry Mass:     ', rep_uptake_max_dry, a_str, t_str, '^-1 (mg dry mass)^-1')
    print('Max Uptake - Protein(N.C.):', rep_uptake_max_pro_nc, a_str, t_str, '^-1')
    print('Max Uptake - Protein(L.B.):', rep_uptake_max_pro, a_str, t_str, '^-1')
    print('Consumption Time:          ', rep_consump_time, t_str)
    print('DEREPRESSED STATE')
    print('Max Uptake - Dry Mass:     ', derep_uptake_max_dry, a_str, t_str, '^-1 (mg dry mass)^-1')
    print('Max Uptake - Protein(N.C.):', derep_uptake_max_pro_nc, a_str, t_str, '^-1')
    print('Max Uptake - Protein(L.B.):', derep_uptake_max_pro, a_str, t_str, '^-1')
    print('Consumption Time:          ', derep_consump_time, t_str)
    print(' ')
    print('Time Step Size:            ', time_step, t_str)
    print('Num. Steps Per Day:        ', steps_in_a_day)
    print('----------------------------------------------------')
    print('')
    
# See the parameters with varying units
#   Parameter 1: 1 for um & 1000 for mm
#   Parameter 2: 1 for sec, 60 for min, 3600 for hr
#   Parameter 3: 1 for mol & 1000 for mmol
    
# # Units: um & s & mol
# parameter_calc(1,1,1)
# # Units: mm & s & mol
# parameter_calc(1000,1,1)

# Units: um & min & mol
# parameter_calc(1,60,1)
# Units: mm & min & mol
# parameter_calc(1000,60,1)

# Units: um & hr & mol
# parameter_calc(1,3600,1)
# Units: mm & hr & mol
# parameter_calc(1000,3600,1)

# Units: um & s & mmol
parameter_calc(1,1,1000)
# Units: mm & s & mol
#parameter_calc(1000,1,1000) #Commented out on 12/30/2021
    
    
    
    
    
