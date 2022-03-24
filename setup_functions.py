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
    max_num_branches = 30000#20000#10000 #5000
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
    mycelia['seg_length'][:num_total_segs - num_branches] = params['sl']
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

def external_grid_patchy():
    # Define external domain grid
    scale_val = params['grid_scale_val']
    x_vals = np.arange(-params['sl']*scale_val, params['sl']*scale_val+params['dy'], params['dy'])
    y_vals = np.arange(-params['sl']*scale_val, params['sl']*scale_val+params['dy'], params['dy'])
    xe, ye = np.meshgrid(x_vals, y_vals)
    # breakpoint()
    
    # Matrix for external nutrients; usnits are mmoles of glucose, not concentrations
    sub_e_gluc = 0.0*np.ones(xe.shape)
    sub_e_treha = 0.0*np.ones(xe.shape)
    
    
    # r1 = [150,
    #  158,   192,   123,   238,   209,
    # 279,   158,   101,    40,   180,   
    # 226,   128,    36,    84,    53,
    #  88,   132,   157,   137,   253,
    # 154,   272,   187,   276,    77,
    # 198,    91,   197,   203,    29,
    #  81,    73,   195,   244,   106,
    # 227,   198,    12,   177,   118,
    # 264,    11,   139,   128,   138,
    # 224,   100,   228,   141,    20]
    
    # r2 = [150,
    #  59,   210,   142,    53,   105,
    # 179,    64,   215,    78,   265,
    #  85,   223,    63,    90,    36,
    # 170,   200,   162,   128,   189,
    # 190,   199,   187,   272,    68,
    # 207,    76,    44,   179,   135,
    # 138,   194,   224,   108,   194,
    # 126,   244,   241,    82,   180,
    # 172,   160,   251,    84,    99,
    #  44,   271,   189,   143,   188]
    
   #  for i in range(len(r1)):
   #      center_x = r1[i]
   #      center_y = r2[i]
   #      for j in range(12):
   #          for k in range(12):
   #              sub_e_gluc[center_x+j,center_y+k] = params['init_sub_e_gluc']
   #              sub_e_gluc[center_x-j,center_y-k] = params['init_sub_e_gluc']
   
    r1 = [31,
           134,
           113,
            65,
           112,
            95,
            98,
           106,
            95,
            46,
            23,
           100,
           133,
            67,
           109,
            35,
            55,
            46,
            26,
           134,
            86,
           107,
            76,
            75,
            81,
            30,
           119,
            90,
            56,
            86]
     
    r2 = [48,
           107,
            84,
            18,
           111,
            28,
            71,
            54,
           113,
            79,
            88,
            95,
           107,
            22,
           128,
           139,
            68,
           134,
           110,
           122,
           113,
           128,
            45,
            74,
            85,
            30,
            27,
           132,
            74,
           126]
    
    # r1 = [38]
    # r2 = [38]
     
    for i in range(len(r1)):
        center_x = r1[i]
        center_y = np.flip(r2[i])
        for j in range(5):
            for k in range(5):
                sub_e_gluc[center_x+j,center_y+k] = params['init_sub_e_gluc']
                sub_e_gluc[center_x-j,center_y-k] = params['init_sub_e_gluc']
                sub_e_gluc[center_x+j,center_y-k] = params['init_sub_e_gluc']
                sub_e_gluc[center_x-j,center_y+k] = params['init_sub_e_gluc']
                
    return x_vals, y_vals, sub_e_gluc, sub_e_treha
            
            
            
            
