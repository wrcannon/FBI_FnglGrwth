#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:43:39 2021

@author: jolenebritton
"""


import numpy as np
import helper_functions as hf
import nutrient_functions as nf
import growth_functions as gf
import math 

params, config = hf.get_configs('parameters.ini')

# ----------------------------------------------------------------------------
# SET UP FUNCTIONS
# ----------------------------------------------------------------------------

def mycelia_dict():
    """
    Returns
    -------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    """
    
    # Read in the final time (in seconds) and convert to days
    final_time_secs = params['final_time']
    final_time_days = final_time_secs/(60*60*24)
    
    # Determine max radii of colony (units: cm)
    # Values from 2014 Fronteirs paper by Labbe et al.
    # Used to determine number of max possible segments
    if params['environ_type'] == 'control':
        if final_time_days <= 12:
            max_radii = (1.27 / 2) / (12 / final_time_days)
        elif final_time_days == 16:
            max_radii = 2.85 / 2
        elif final_time_days == 20:
            max_radii = 4.43 / 2
        elif final_time_days == 24:
            max_radii = 6.01 / 2
        elif final_time_days == 28:
            max_radii = 7.59 / 2
        elif final_time_days == 31:
            max_radii = 9.00 / 2
    elif params['environ_type'] == 'gm41':
        if final_time_days <= 12:
            max_radii = (4.25 / 2) / (12 / final_time_days)
        elif final_time_days == 16:
            max_radii = 6.78 / 2
        elif final_time_days == 20:
            max_radii = 9.31 / 2
        elif final_time_days == 24:
            max_radii = 11.5 / 2
        elif final_time_days == 28:
            max_radii = 13.6 / 2
        elif final_time_days == 31:
            max_radii = 15.0 / 2
    
    # Max number of segments on each branch
    max_num_segs = int(np.ceil(max_radii / params['sl']))
    print('max_num_segs : ', max_num_segs)
    
    # Max number of hyphal branches - convert to function of final time?
    max_num_branches = 160000#20000#10000 #5000
    print('max_num_branches : ', max_num_branches)
    
    # Max total number of segments in fungal colony
    max_total_segs = max_num_segs*max_num_branches
    print('max_total_segs : ', max_total_segs)
    
    # Dictionary in which data for each branch is stored
    # Create array for each trait - index of array corresponds to segment ID
    mycelia = {
        'branch_id': np.zeros((max_total_segs, 1)),
        'seg_id': np.zeros((max_total_segs, 1)),
        'xy1': np.zeros((max_total_segs, 2)),
        'xy2': np.zeros((max_total_segs, 2)),
        'angle': np.zeros((max_total_segs, 1)),
        'seg_length': np.zeros((max_total_segs, 1)),
        'seg_vol': np.zeros((max_total_segs, 1)),
        'dist_to_septa': np.zeros((max_total_segs, 1)),
        'xy_e_idx': np.zeros((max_total_segs, 2)),
        'share_e': [None]*max_total_segs,
        'cw_i': np.zeros((max_total_segs, 1)),
        'gluc_i': np.zeros((max_total_segs, 1)),
        'can_branch': np.zeros((max_total_segs, 1), dtype=bool),
        #'can_branch': np.ones((max_total_segs, 1), dtype=bool),
        'is_tip': np.zeros((max_total_segs, 1), dtype=bool),
        'septa_loc': np.zeros((max_total_segs, 1), dtype=bool),
        'nbr_idxs': [None]*max_total_segs,
        'nbr_num': np.zeros((max_total_segs, 1)),
        'bypass': np.zeros((max_total_segs, 1), dtype=bool),
        'treha_i' : np.zeros((max_total_segs,1)),
        'dist_from_center': np.zeros((max_total_segs, 1))
        }
    #'bypass' indicates that the segment is bypassed due to failing CFL condition, not
    # because it gets null-out due to fusion (anastomosis).
    
    return mycelia

# ----------------------------------------------------------------------------

def initial_conditions_cross(mycelia, num_segs, x_vals, y_vals):
    
    # Initial mycelia centered at origin with line segments of same length extending from it
    # num_branches = 2
    num_branches = 4#2
    # num_segs = 3
    num_total_segs = num_branches*num_segs
    
    # Assign branch IDs
    mycelia['branch_id'][0:num_total_segs:num_branches] = 0
    mycelia['branch_id'][1:num_total_segs:num_branches] = 1
    mycelia['branch_id'][2:num_total_segs:num_branches] = 2
    mycelia['branch_id'][3:num_total_segs:num_branches] = 3
        
    # Assign segment IDs (for on a given branch)
    mycelia['seg_id'][:num_total_segs] = (np.arange(num_total_segs).reshape(-1,1) - mycelia['branch_id'][:num_total_segs])/num_branches
    
    # Neighbor assignments - append the segment where it originates from
    # The first two segments come from one another
    # mycelia['nbr_idxs'][0] = [1]
    # mycelia['nbr_idxs'][1] = [0]
    mycelia['nbr_idxs'][0] = [1,2,3]#[1]
    mycelia['nbr_idxs'][1] = [0,2,3]#[0]
    mycelia['nbr_idxs'][2] = [0,1,3]
    mycelia['nbr_idxs'][3] = [0,1,2]
    # The rest of the segments come from segment with index that is 2 smaller
    for idx in range(num_branches, num_total_segs):
        mycelia['nbr_idxs'][idx] = [idx - num_branches]
    # breakpoint()
    # Neighbor assignments - append the segment where it extends to
    # The first num_total_segs-2 branch to segment with index that is 2 larger
    # Also keep track of how many neighbors a segment has
    if (num_total_segs - num_branches != 0):
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        for idx in range(num_total_segs - num_branches):
            mycelia['nbr_idxs'][idx].append(idx + num_branches)
            mycelia['nbr_num'][idx] = 2
        
        mycelia['nbr_num'][num_total_segs - num_branches:num_total_segs] = 1
    else:
        # breakpoint()
        print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
        for idx in range(num_branches):
            mycelia['nbr_num'][idx] = 3
    # Angles of each branch
    mycelia['angle'][0:num_total_segs:num_branches] = np.random.normal(0, params['angle_sd'], (num_segs,1))
    mycelia['angle'][1:num_total_segs:num_branches] = np.pi + np.random.normal(0, params['angle_sd'], (num_segs,1))
    mycelia['angle'][2:num_total_segs:num_branches] = np.pi/2.0 + np.random.normal(0, params['angle_sd'], (num_segs,1))
    mycelia['angle'][3:num_total_segs:num_branches] = np.pi*(3.0/2.0) + np.random.normal(0, params['angle_sd'], (num_segs,1))
    
    # Length of each segments
    mycelia['seg_length'][:num_total_segs - num_branches] = params['sl']
    if (num_total_segs - num_branches == 0):
        mycelia['seg_length'][num_total_segs - num_branches:num_total_segs] = params['sl']
    else:
        mycelia['seg_length'][num_total_segs - num_branches:num_total_segs] = 0.75*params['sl']
    mycelia['seg_vol'] =     mycelia['seg_length']*params['cross_area']
    
    # Concentration of glucose in 0th segment of each branch
    # mycelia['gluc_i'][:num_branches] = params['init_sub_i_gluc']*params['cross_area']*params['sl']
    # All initial segments have the same concentration
    mycelia['gluc_i'][:num_total_segs] = params['init_sub_i_gluc']*params['cross_area']*params['sl']
    
    # Position of segments 
    #   First index - corresponds to idx
    #   Second indx - if 0 => x-value, if 1 => y-value
    # Segment 0 on each branch originates at the origin
    # Determine other endpoint for segment 0 on each branch with angle and segment length
    mycelia['xy2'][0,0] = mycelia['seg_length'][0] * np.cos(mycelia['angle'][0])
    mycelia['xy2'][0,1] = mycelia['seg_length'][0] * np.sin(mycelia['angle'][0])
    mycelia['xy2'][1,0] = mycelia['seg_length'][1] * np.cos(mycelia['angle'][1])
    mycelia['xy2'][1,1] = mycelia['seg_length'][1] * np.sin(mycelia['angle'][1])
    mycelia['xy2'][2,0] = mycelia['seg_length'][2] * np.cos(mycelia['angle'][2])
    mycelia['xy2'][2,1] = mycelia['seg_length'][2] * np.sin(mycelia['angle'][2])
    mycelia['xy2'][3,0] = mycelia['seg_length'][3] * np.cos(mycelia['angle'][3])
    mycelia['xy2'][3,1] = mycelia['seg_length'][3] * np.sin(mycelia['angle'][3])
    # Determine other segment endpoints
    for idx in range(num_branches, num_total_segs):
        # Starting endpoint of idx = ending endpoint of idx-2
        mycelia['xy1'][idx,:] = mycelia['xy2'][idx - num_branches,:]
        
        # Determine ending endpoint od idx using angle and length of the segment
        mycelia['xy2'][idx,0] = mycelia['xy1'][idx,0] + mycelia['seg_length'][idx] * np.cos(mycelia['angle'][idx])
        mycelia['xy2'][idx,1] = mycelia['xy1'][idx,1] + mycelia['seg_length'][idx] * np.sin(mycelia['angle'][idx])
         
    # Denote the tip segments
    mycelia['is_tip'][num_total_segs - num_branches:num_total_segs] = True
    
    # Keep track of distance between tip and septa ('i.e. center of colony in this case)
    mycelia['dist_to_septa'][np.where(mycelia['is_tip'])[0]] = ((num_segs-1)*params['sl'])+(0.75*params['sl'])
    
    # Map to external grid
    for idx in range(num_total_segs):
        mycelia = gf.map_to_grid(mycelia, idx, num_total_segs, x_vals, y_vals)
            
    # Distance to tip
    dtt = nf.distance_to_tip_new(mycelia, num_total_segs)
    # breakpoint()
            
    return mycelia, num_branches, num_total_segs, dtt

def initial_conditions_line(mycelia, num_segs, x_vals, y_vals):
    
    # Initial mycelia centered at origin with line segments of same length extending from it
    num_branches = 2
    # num_segs = 3
    num_total_segs = num_branches*num_segs
    
    # Assign branch IDs
    mycelia['branch_id'][0:num_total_segs:num_branches] = 0
    mycelia['branch_id'][1:num_total_segs:num_branches] = 1
    
        
    # Assign segment IDs (for on a given branch)
    mycelia['seg_id'][:num_total_segs] = (np.arange(num_total_segs).reshape(-1,1) - mycelia['branch_id'][:num_total_segs])/num_branches
    
    # Neighbor assignments - append the segment where it originates from
    # The first two segments come from one another
    # mycelia['nbr_idxs'][0] = [1]
    # mycelia['nbr_idxs'][1] = [0]
    mycelia['nbr_idxs'][0] = [1]
    mycelia['nbr_idxs'][1] = [0]
    
    # The rest of the segments come from segment with index that is 2 smaller
    for idx in range(num_branches, num_total_segs):
        mycelia['nbr_idxs'][idx] = [idx - num_branches]
    
    # Neighbor assignments - append the segment where it extends to
    # The first num_total_segs-2 branch to segment with index that is 2 larger
    # Also keep track of how many neighbors a segment has
    if (num_total_segs - num_branches != 0):
        for idx in range(num_total_segs - num_branches):
            mycelia['nbr_idxs'][idx].append(idx + num_branches)
            mycelia['nbr_num'][idx] = 2
        
        mycelia['nbr_num'][num_total_segs - num_branches:num_total_segs] = 1
    else:
        for idx in range(num_branches):
            # mycelia['nbr_idxs'][idx].append(idx + num_branches)
            mycelia['nbr_num'][idx] = 1
    # Angles of each branch
    mycelia['angle'][0:num_total_segs:num_branches] = np.random.normal(0, params['angle_sd'], (num_segs,1))
    mycelia['angle'][1:num_total_segs:num_branches] = np.pi + np.random.normal(0, params['angle_sd'], (num_segs,1))
   
    # Length of each segments
    # mycelia['seg_length'][:num_total_segs - num_branches] = params['sl']
    # mycelia['seg_length'][num_total_segs - num_branches:num_total_segs] = 0.75*params['sl']
    # mycelia['seg_vol'] =     mycelia['seg_length']*params['cross_area']
    
    # Length of each segments
    mycelia['seg_length'][:num_total_segs - num_branches] = params['sl']
    if (num_total_segs - num_branches == 0):
        mycelia['seg_length'][num_total_segs - num_branches:num_total_segs] = params['sl']
    else:
        mycelia['seg_length'][num_total_segs - num_branches:num_total_segs] = 0.75*params['sl']
    mycelia['seg_vol'] =     mycelia['seg_length']*params['cross_area']
    
    # Concentration of glucose in 0th segment of each branch
    # mycelia['gluc_i'][:num_branches] = params['init_sub_i_gluc']*params['cross_area']*params['sl']
    # All initial segments have the same concentration
    mycelia['gluc_i'][:num_total_segs] = params['init_sub_i_gluc']*params['cross_area']*params['sl']
    
    # Position of segments 
    #   First index - corresponds to idx
    #   Second indx - if 0 => x-value, if 1 => y-value
    # Segment 0 on each branch originates at the origin
    # Determine other endpoint for segment 0 on each branch with angle and segment length
    mycelia['xy2'][0,0] = mycelia['seg_length'][0] * np.cos(mycelia['angle'][0])
    mycelia['xy2'][0,1] = mycelia['seg_length'][0] * np.sin(mycelia['angle'][0])
    mycelia['xy2'][1,0] = mycelia['seg_length'][1] * np.cos(mycelia['angle'][1])
    mycelia['xy2'][1,1] = mycelia['seg_length'][1] * np.sin(mycelia['angle'][1])
    
    # Determine other segment endpoints
    for idx in range(num_branches, num_total_segs):
        # Starting endpoint of idx = ending endpoint of idx-2
        mycelia['xy1'][idx,:] = mycelia['xy2'][idx - num_branches,:]
        
        # Determine ending endpoint od idx using angle and length of the segment
        mycelia['xy2'][idx,0] = mycelia['xy1'][idx,0] + mycelia['seg_length'][idx] * np.cos(mycelia['angle'][idx])
        mycelia['xy2'][idx,1] = mycelia['xy1'][idx,1] + mycelia['seg_length'][idx] * np.sin(mycelia['angle'][idx])
         
    # Denote the tip segments
    mycelia['is_tip'][num_total_segs - num_branches:num_total_segs] = True
    
    # Keep track of distance between tip and septa ('i.e. center of colony in this case)
    mycelia['dist_to_septa'][np.where(mycelia['is_tip'])[0]] = ((num_segs-1)*params['sl'])+(0.75*params['sl'])
    
    # Map to external grid
    for idx in range(num_total_segs):
        mycelia = gf.map_to_grid(mycelia, idx, num_total_segs, x_vals, y_vals)
            
    # Distance to tip
    dtt = nf.distance_to_tip(mycelia, num_total_segs)
    # breakpoint()
            
    return mycelia, num_branches, num_total_segs, dtt


          
# ----------------------------------------------------------------------------  
            
def external_grid():
    # Define external domain grid
    scale_val = params['grid_scale_val']
    x_vals = np.arange(-params['sl']*scale_val, params['sl']*scale_val+params['dy'], params['dy'])
    y_vals = np.arange(-params['sl']*scale_val, params['sl']*scale_val+params['dy'], params['dy'])
    xe, ye = np.meshgrid(x_vals, y_vals)
    # breakpoint()
    
    # Matrix for external nutrients; usnits are mmoles of glucose, not concentrations
    sub_e_gluc = params['init_sub_e_gluc']*np.ones(xe.shape)
    sub_e_treha = params['init_sub_e_treha']*np.ones(xe.shape)
    num_ycells = len(y_vals)-1
    if params['init_sub_e_dist'] == 'heterogeneous':
        center = int(num_ycells/2)
        center4 = int((4/5)*center)
        center2 = int((2/5)*center)
        center1 = int((1/5)*center)
        sub_e_gluc[center-center4:center+center4, center-center4:center+center4] = 0.0
        sub_e_gluc[center-center2:center+center2, center-center2:center+center2] = params['init_sub_e_gluc']
        sub_e_gluc[center-center1:center+center1, center-center1:center+center1] = 0.0
    elif params['init_sub_e_dist'] == 'pseudo_bacteria':
        sub_e_gluc = 0.05*sub_e_gluc
        center = int(num_ycells/2)
        dist = 0.05
        big = int((1-dist)*center)
        small = int((1+dist)*center)
        colony_size = 3
        factor = 2
        sub_e_gluc[big-colony_size:big+colony_size, center-colony_size:center+colony_size] += factor*params['init_sub_e_gluc']
        sub_e_treha[big-colony_size:big+colony_size, center-colony_size:center+colony_size] += factor*params['init_sub_e_treha']
        # sub_e_gluc[small-colony_size:small+colony_size, center-colony_size:center+colony_size] += factor*params['init_sub_e_gluc']
        # sub_e_gluc[center-colony_size:center+colony_size, big-colony_size:big+colony_size] += factor*params['init_sub_e_gluc']
        # sub_e_gluc[center-colony_size:center+colony_size, small-colony_size:small+colony_size] += factor*params['init_sub_e_gluc']
                
    return x_vals, y_vals, sub_e_gluc, sub_e_treha

##############################################################################

def external_grid_patchy(set,seed=6):
    # Define external domain grid
    scale_val = params['grid_scale_val']
    # Number of grid x points = 2*params['sl']*scale_val/params['dy'] + 1:
    # Total number of grids: (2*params['sl']*scale_val/params['dy'] + 1)^2
    # Range of grids: +/-params['sl']*scale_val
    x_vals = np.arange(-params['sl']*scale_val, params['sl']*scale_val+params['dy'], params['dy'])
    y_vals = np.arange(-params['sl']*scale_val, params['sl']*scale_val+params['dy'], params['dy'])
    xe, ye = np.meshgrid(x_vals, y_vals)
    # breakpoint()
    covered_grid = 0
    
    # Matrix for external nutrients; usnits are mmoles of glucose, not concentrations
    sub_e_gluc = 0.0*np.ones(xe.shape)
    sub_e_treha = 0.0*np.ones(xe.shape)
    
    ## Note that for this type of initial condition, we want the nutrient 
    ## distribution to cover roughly 30% of the domain. This 30% will be 
    ## calculated using the number of grid points. 
    ## x_vals describes how many grid points on on the x-direction and 
    ## similarly for y_vals.
    ## Depending on how big each "focus" of the nutrient spots, we need to 
    ## adjust its "radius" (M in the later part of the function).
    # In other words, the width of each patch of nutrient is determined by the variable M.
    # The width of a patch is M*params['dy']
    
    ## For grid_scale_val = 80
    if (set == 0):
        ## Set 0
        M = 9
        #r1 = [round(len(x_vals)/2)]
        rng = np.random.default_rng(seed)
        r1 =  [round(len(x_vals)/2)] + rng.integers(0, high=226, size=(49,), endpoint=False).tolist()
        #r1.append(np.random.randint(low=1, high=240,size=(50,)).tolist())
        #np.round(np.random.uniform(low=1, high=240,(50,))
        r2 =  [round(len(y_vals)/2)] + rng.integers(0, high=240, size=(49,), endpoint=False).tolist()
            
    if (set == 1):
        ## Set 1
        M = 9 # width of each focus of nutrient spots
        r1 = [round(len(x_vals)/2),   210,    51,    69,    43,
        41,   201,   138,   131,    43,
       198,   147,    88,   123,    99,
        28,    64,    38,    51,    64,
       102,    22,   209,   218,   118,
       118,    85,   208,    92,    35,
       182,    96,    64,    99,    32,
        40,   217,   220,   137,    24,
        62,    88,   191,    14,    20,
        48,   153,   171,   153,   110]
        
        r2 = [round(len(y_vals)/2),    76,   174,    52,   161,
        51,    92,   148,   182,    29,
       215,   181,   118,   106,   109,
        78,   122,   123,   190,   185,
       152,    94,   189,   128,    88,
       217,   203,   131,   147,   140,
        56,    77,   114,    61,   196,
        54,    60,    48,    61,   106,
        79,   213,   105,    51,   209,
       226,   107,    35,    68,   101]
    if (set == 2):
        ## Set 2
        M = 9
        r1 = [round(len(x_vals)/2), 210,    38,   212,   150,
        32,    72,   131,   222,   223,
        45,   225,   222,   117,   187,
        41,   103,   212,   185,   222,
       155,    18,   198,   216,   160,
       177,   174,    97,   155,    48,
       166,    17,    71,    20,    31,
       192,   164,    80,   220,    18,
       107,    94,   179,   186,    51,
       118,   108,   153,   167,   177]
        
        r2 = [round(len(y_vals)/2), 160,   155,    46,    36,
       120,   222,    85,   139,    59,
       176,    66,   122,   164,   207,
       222,   131,    41,    43,    67,
       196,    66,   190,    64,   215,
        87,    53,    65,   146,   115,
        88,   194,   139,   131,   213,
        73,   177,   177,    94,   135,
        27,    22,   127,   182,   216,
        39,   136,   114,    13,    85]
    if (set == 3):
        ## Set 3
        M = 9
        r1 = [round(len(x_vals)/2), 186,    79,   127,    47,
       143,    68,   155,   162,   175,
       110,    29,    61,   212,    44,
       193,   129,   230,    27,   108,
        34,   223,    11,   181,   191,
       202,    29,    98,    67,   187,
       105,   211,    50,    68,    42,
        40,   202,   138,   132,    42,
       199,   147,    88,   123,    99,
        27,    63,    37,    51,    63]
        r2 = [round(len(y_vals)/2), 21,   210,   219,   118,
       118,    85,   209,    92,    35,
       182,    96,    63,    99,    31,
        39,   218,   221,   137,    23,
        62,    88,   191,    13,    20,
        47,   153,   172,   153,   110,
       131,    75,   175,    52,   162,
        51,    91,   148,   182,    28,
       215,   181,   118,   106,   109,
        78,   122,   123,   191,   186,]
    
    if (set == 4):
        ## Set 4
        M = 4
        r1 = [round(len(x_vals)/2),  156 ,   48 ,   75 ,  220 ,   77,    17 ,  152 ,   66 ,  #210,
           161 ,  125 ,   96 ,  105 ,  213 ,  165 ,  134 ,   93,    74 ,  204,
            19 ,  225 ,  194 ,   13 ,   22 ,  157 ,  205 ,   52,   146 ,  191,
            26 ,  153 ,  188 ,  227 ,  173 ,  129 ,  158 ,  105,    69 ,   68,
           125 ,  187 ,   23 ,   47 ,   69 ,  164 ,   52 ,  117,   192 ,  141,
            31 ,  110 ,   98 ,   33 ,  103 ,  157 ,   92 ,   37,   227 ,   15,
           191 ,  106 ,  126 ,   92 ,  131 ,   49 ,  112 ,  140,   171 ,  104,
           191 ,  192 ,  102 ,   54 ,  218 ,   38 ,  227 ,   60,    86 ,   79,
           170 ,   28 ,  155 ,  118 ,  102 ,  231 ,   45 ,   95,   139 ,   46,
            43 ,   39 ,  149 ,   85 ,  227 ,   48 ,  199 ,  139,    34 ,   50]
        r2 = [round(len(y_vals)/2), 182,   124,    66,   212,    49,    69,   153,   111,
            31,   155,   104,   219,    60,    10,   170,   179,   160,   156,
           142,   100,    30,   151,   158,   112,   115,    52,   151,   180,
           114,   191,    69,   222,   197,   104,    44,    74,   219,    87,
           164,   169,    44,    63,    86,   112,    85,    30,    56,   156,
           165,   224,    72,   159,   182,   180,   144,   137,   167,   102,
           151,   127,   107,    74,   159,    81,    52,   161,    62,   196,
            17,    82,   126,   158,    11,   183,   173,   131,    36,   194,
            25,    33,   111,   164,   143,   114,    64,   104,   144,    67,
            81,   145,   203,    25,    95,    18,   213,   152,   109,   146]
    
    if (set == 5):
        ## Set 5
        M = 4
        r1 = [round(len(x_vals)/2), 99 ,  164 ,   87 ,  173 ,  192,   185 ,  124,   128,
           130 ,  153 ,  109 ,   31 ,   43 ,   97 ,  105 ,  220 ,  205 ,   30,
           202 ,  130 ,   91 ,  126 ,  140 ,  161 ,  206 ,   82 ,  140 ,   35,
            69 ,  169 ,  179 ,  127 ,   68 ,  166 ,   96 ,  158 ,   44 ,   40,
            80 ,  125 ,  149 ,  200 ,   20 ,  108 ,  180 ,  107 ,   54 ,  160,
            36 ,  230 ,  181 ,  117 ,  177 ,   14 ,   98 ,  194 ,  100 ,  119,
           218 ,   58 ,  216 ,   97 ,   64 ,   83 ,  189 ,  180 ,  175 ,   52,
           153 ,   33 ,  225 ,  158 ,  108 ,  104 ,  177 ,   47 ,  192 ,  119,
           116 ,   34 ,   52 ,  174 ,  162 ,   70 ,   93 ,  200 ,  185 ,   43,
           151 ,   24 ,   41 ,  125 ,   89 ,   54 ,   58 ,  229 ,   80 ,   22]
        r2 = [round(len(y_vals)/2),  173 ,  180 ,  132  ,  42  ,  37 ,  219 ,  173,    40,
           134,   229,   140,   138,   149,    52,   118,    28,    24,    17,
           215,   127,    65,   215,    17,    19,   199,    33,   200,   218,
           164,   116,   157,   138,   146,   150,   203,    41,   217,    77,
           139,   187,    28,    14,    90,    72,    70,    47,   228,    75,
           190,    60,   148,    37,    21,   129,    56,   147,   200,    84,
           204,   120,   156,   201,   118,   164,   135,   137,   184,   113,
           229,   209,   171,   117,    53,   120,   152,    22,   123,   153,
            10,   137,   207,   197,    37,   128,   102,   216,    49,    16,
           201,   197,   227,    56,    55,   108,    56,   171,    98,   196]
     
     # r1, r2 were set up for a 241x241 grid with a grid width of 20 microns
    nrows = len(xe)
    ngrids = nrows*nrows
    np_array_r1 = np.array(r1)
    np_array_r2 = np.array(r2)
    r1[1:] = (np.int_(2*np.round(np_array_r1[1:]* (np_array_r1[0])/240))).tolist() 
    r2[1:] = (np.int_(2*np.round(np_array_r2[1:]* (np_array_r2[0])/240))).tolist() 
    #r1[1:] = (np.int_(np.round(np_array_r1[1:]* (ngrids-1)/(240*240)))).tolist() 
    #r2[1:] = (np.int_(np.round(np_array_r2[1:]* (ngrids-1)/(240*240)))).tolist() 

     # M was originally set up for a grid width of 20
    M = np.int_(2*np.round(M*np_array_r1[0]/240)) #*params['dy']/20))
    #M = np.int_(round(M*(ngrids-1)/(240*240)*params['dy']/20))
    
    for i in range(len(r1)):
        #print(i)
        center_x = r1[i]
        center_y = np.flip(r2[i])
        for j in range(M):
            for k in range(M):
                sub_e_gluc[center_x+j,center_y+k] = params['init_sub_e_gluc']
                sub_e_gluc[center_x-j,center_y-k] = params['init_sub_e_gluc']
                sub_e_gluc[center_x+j,center_y-k] = params['init_sub_e_gluc']
                sub_e_gluc[center_x-j,center_y+k] = params['init_sub_e_gluc']
                covered_grid += 1
    
    print('The number of grid points covered : ', covered_grid)
    print('Total number of grid points in the domain : ', len(x_vals)*len(y_vals))
                
    return x_vals, y_vals, sub_e_gluc, sub_e_treha
            
def grid_density(mycelia, sub_e_gluc, num_total_segments):
    count = np.zeros(sub_e_gluc.shape)
    for i in range(num_total_segments):
        tpl = tuple(np.int_(mycelia['xy_e_idx'][i]))
        count[tpl] = count[tpl] +1
    return count

            
            
