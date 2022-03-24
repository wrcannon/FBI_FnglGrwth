#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:43:32 2020

@author: jolenebritton
"""

import numpy as np
# import operator
from scipy.linalg import solve_banded
import helper_functions as hf
import growth_functions as gf

params, config = hf.get_configs('parameters.ini')


# ----------------------------------------------------------------------------
# EXTERNAL NUTRIENT FUNCTIONS
# ----------------------------------------------------------------------------


def diffusion_ADI(sub_e):
    
    """
    Parameters
    ----------
    sub_e : array (2D)
        The 2D grid storing values of glucose in the external domain.
        
    Returns
    -------
    sub_e_step2 : array (2D)
        The updated 2D grid storing values of glucose in the external domain
        after diffusion using the finite difference alternating-direction method
        which is implicit.
        
    Purpose
    -------
    Want to solve Ax = b, where
      A = tri-diagonal matrix (-r, 1+2r, -r)
      b = -r*u^k_{i-1,j} + (1-2r)*u^k_{i,j} - r*u^k_{i+1,j}  (for Step 1)
       or
      b = -r*u^{k+1/2}_{i,j-1} + (1-2r)*u^{k+1/2}_{i,j} - r*u^{k+1/2}_{i,j+1}  (for Step 2)
    But we can write the tri-diag matrix as a banded matrix
    """
    
    # Create tri-diagonal matrix
    num_rows = np.shape(sub_e)[0]
    num_cols = np.shape(sub_e)[1]
    r_coeff = (params['dt']*params['diffusion_e_gluc'])/(2*params['dy']**2)
    banded_mat_rows = np.tile(np.array([-r_coeff, (1+2*r_coeff), -r_coeff]).reshape(3,1),num_cols)
    banded_mat_cols = np.tile(np.array([-r_coeff, (1+2*r_coeff), -r_coeff]).reshape(3,1),num_rows)
    
    # Adjust the matrices
    banded_mat_rows[0, 0] = 0
    banded_mat_rows[2, num_cols-1] = 0
    banded_mat_cols[0, 0] = 0
    banded_mat_cols[2, num_cols-1] = 0
    
    # Step 1: k+1/2 values - Loop through the rows
    sub_e_step1 = np.zeros((num_rows,num_cols))
    for row in range(num_rows):
        # Create the right hand side
        if row == 0:
            rhs_step1 = (r_coeff*params['init_sub_e_gluc']
               + (1-2*r_coeff)*sub_e[row,:]
               + r_coeff*sub_e[row+1,:])
        elif row == num_rows-1:
            rhs_step1 = (r_coeff*sub_e[row-1,:] 
               + (1-2*r_coeff)*sub_e[row,:]
               + r_coeff*params['init_sub_e_gluc'])
        else:
            rhs_step1 = (r_coeff*sub_e[row-1,:] 
                   + (1-2*r_coeff)*sub_e[row,:]
                   + r_coeff*sub_e[row+1,:])
        
        rhs_step1[0] += r_coeff*params['init_sub_e_gluc']
        rhs_step1[num_cols-1] += r_coeff*params['init_sub_e_gluc']
        
        # Solve the matrix problem: tri_diag_rows*sub_e^{k+1/2} = rhs
        sub_e_step1[row,:] = solve_banded((1,1), banded_mat_rows, rhs_step1)
    
    # breakpoint()
    # Step 2: k+1 values - Loop through the columns
    sub_e_step2 = np.zeros((num_rows,num_cols))
    for col in range(num_cols):
        # Create the right hand side
        if col == 0:
            rhs_step2 = (r_coeff*params['init_sub_e_gluc']
               + (1-2*r_coeff)*sub_e_step1[:,col]
               + r_coeff*sub_e_step1[:,col+1])
        elif col == num_cols-1:
            rhs_step2 = (r_coeff*sub_e_step1[:,col-1] 
               + (1-2*r_coeff)*sub_e_step1[:,col]
               + r_coeff*params['init_sub_e_gluc'])
        else:
            rhs_step2 = (r_coeff*sub_e_step1[:,col-1] 
                   + (1-2*r_coeff)*sub_e_step1[:,col]
                   + r_coeff*sub_e_step1[:,col+1])
        
        rhs_step2[0] += r_coeff*params['init_sub_e_gluc']
        rhs_step2[num_rows-1] += r_coeff*params['init_sub_e_gluc']
        
        # Solve the matrix problem: tri_diag_rows*sub_e^{k+1} = rhs
        sub_e_step2[:,col] = solve_banded((1,1), banded_mat_cols, rhs_step2)
       
    # breakpoint()
    # if np.min(sub_e_step2) < 0:
    #     breakpoint()
    return sub_e_step2

def diffusion_ADI_treha(sub_e):
    
    """
    Parameters
    ----------
    sub_e : array (2D)
        The 2D grid storing values of glucose in the external domain.
        
    Returns
    -------
    sub_e_step2 : array (2D)
        The updated 2D grid storing values of glucose in the external domain
        after diffusion using the finite difference alternating-direction method
        which is implicit.
        
    Purpose
    -------
    Want to solve Ax = b, where
      A = tri-diagonal matrix (-r, 1+2r, -r)
      b = -r*u^k_{i-1,j} + (1-2r)*u^k_{i,j} - r*u^k_{i+1,j}  (for Step 1)
       or
      b = -r*u^{k+1/2}_{i,j-1} + (1-2r)*u^{k+1/2}_{i,j} - r*u^{k+1/2}_{i,j+1}  (for Step 2)
    But we can write the tri-diag matrix as a banded matrix
    """
    
    # Create tri-diagonal matrix
    # breakpoint()
    num_rows = np.shape(sub_e)[0]
    num_cols = np.shape(sub_e)[1]
    r_coeff = (params['dt']*params['diffusion_e_gluc'])/(2*params['dy']**2)
    banded_mat_rows = np.tile(np.array([-r_coeff, (1+2*r_coeff), -r_coeff]).reshape(3,1),num_cols)
    banded_mat_cols = np.tile(np.array([-r_coeff, (1+2*r_coeff), -r_coeff]).reshape(3,1),num_rows)
    
    # Adjust the matrices
    banded_mat_rows[0, 0] = 0
    banded_mat_rows[2, num_cols-1] = 0
    banded_mat_cols[0, 0] = 0
    banded_mat_cols[2, num_cols-1] = 0
    
    # Step 1: k+1/2 values - Loop through the rows
    sub_e_step1 = np.zeros((num_rows,num_cols))
    # for row in range(num_rows):
    for row in range(1,num_rows-1):
        # Create the right hand side
        if row == 0:
            rhs_step1 = (r_coeff*params['init_sub_e_treha']
               + (1-2*r_coeff)*sub_e[row,:]
               + r_coeff*sub_e[row+1,:])
        elif row == num_rows-1:
            rhs_step1 = (r_coeff*sub_e[row-1,:] 
               + (1-2*r_coeff)*sub_e[row,:]
               + r_coeff*params['init_sub_e_treha'])
        else:
            rhs_step1 = (r_coeff*sub_e[row-1,:] 
                   + (1-2*r_coeff)*sub_e[row,:]
                   + r_coeff*sub_e[row+1,:])
        
        rhs_step1[0] += r_coeff*params['init_sub_e_treha']
        rhs_step1[num_cols-1] += r_coeff*params['init_sub_e_treha']
        
        # Solve the matrix problem: tri_diag_rows*sub_e^{k+1/2} = rhs
        sub_e_step1[row,:] = solve_banded((1,1), banded_mat_rows, rhs_step1)
    
    # breakpoint()
    # Step 2: k+1 values - Loop through the columns
    sub_e_step2 = np.zeros((num_rows,num_cols))
    # for col in range(num_cols):
    for col in range(1,num_cols-1):
        # Create the right hand side
        if col == 0:
            rhs_step2 = (r_coeff*params['init_sub_e_treha']
               + (1-2*r_coeff)*sub_e_step1[:,col]
               + r_coeff*sub_e_step1[:,col+1])
        elif col == num_cols-1:
            rhs_step2 = (r_coeff*sub_e_step1[:,col-1] 
               + (1-2*r_coeff)*sub_e_step1[:,col]
               + r_coeff*params['init_sub_e_treha'])
        else:
            rhs_step2 = (r_coeff*sub_e_step1[:,col-1] 
                   + (1-2*r_coeff)*sub_e_step1[:,col]
                   + r_coeff*sub_e_step1[:,col+1])
        
        rhs_step2[0] += r_coeff*params['init_sub_e_treha']
        rhs_step2[num_rows-1] += r_coeff*params['init_sub_e_treha']
        
        # Solve the matrix problem: tri_diag_rows*sub_e^{k+1} = rhs
        sub_e_step2[:,col] = solve_banded((1,1), banded_mat_cols, rhs_step2)
       
    # breakpoint()
    # if np.min(sub_e_step2) < 0:
    #     breakpoint()
    return sub_e_step2


# ----------------------------------------------------------------------------
# TRANSLOCATION FUNCTIONS
# ----------------------------------------------------------------------------

## This is the old method where the distance to tip is calculated solely on
## how close a segment is to a tip
def distance_to_tip(mycelia, num_total_segs):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.

    Returns
    -------
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.
    """
    
    # Initialize dist to tip as all ones
    dtt = -1*np.ones((num_total_segs,1))
    non_null_segs = np.where(mycelia['branch_id'][:num_total_segs] > -1)[0]
    null_segs = np.where(mycelia['branch_id'][:num_total_segs] == -1)[0]
    if any(i in null_segs for i in non_null_segs):
        breakpoint()
    # If a segment is a tip, it has a distance to tip of 0
    tip_segs = np.where(mycelia['is_tip'][:num_total_segs])[0]
    dtt[tip_segs] = 0
    dtt[null_segs] = 1e12
    
    # If a segment is a neighbor of a tip, it has a distance to tip (dtt) of 1
    # If a segment is a neighbor of a segment with dtt of i, it has a dtt of i+1
    current_dist = 1
    while min(dtt[non_null_segs]) < 0:
        #breakpoint()
        # print('current_dist = ', current_dist)
        # Loop through all segments
        for idx in range(num_total_segs):
            # breakpoint()
            # print(idx, dtt[idx], mycelia['nbr_idxs'][idx], dtt[mycelia['nbr_idxs'][idx]])
            #breakpoint()
            # Only consider segments that have not yet been assigned (i.e dtt = -1)
            # Only consider segments with a neighbor that has dtt=current_dist-1 
            if dtt[idx] == -1 and (current_dist-1) in dtt[mycelia['nbr_idxs'][idx]]:
                dtt[idx] = current_dist
                
        # Increase the current distance from the tip         
        current_dist += 1
        # This should not happen
        # Only triggered if a segment is missed
        # if current_dist > num_total_segs:
        #     breakpoint()
        
    # breakpoint()
    
    return dtt

## Newer version of active transport, where distance to tip is calculated 
## with bias toward tip of the same branch.    
def distance_to_tip_new(mycelia, num_total_segs):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.

    Returns
    -------
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.
    """
    
    allow_reverse_transport = 0
    if allow_reverse_transport == 1:
        
        # Initialize dist to tip as all ones
        dtt = -1*np.ones((num_total_segs,1))
        non_null_segs = np.where(mycelia['branch_id'][:num_total_segs] > -1)[0]
        null_segs = np.where(mycelia['branch_id'][:num_total_segs] == -1)[0]
        if any(i in null_segs for i in non_null_segs):
            breakpoint()
        # If a segment is a tip, it has a distance to tip of 0
        tip_segs = np.where(mycelia['is_tip'][:num_total_segs])[0]
        dtt[tip_segs] = 0
        dtt[null_segs] = 1e12
        
        # If a segment is a neighbor of a tip, it has a distance to tip (dtt) of 1
        # If a segment is a neighbor of a segment with dtt of i, it has a dtt of i+1
        current_dist = 1
        while min(dtt[non_null_segs]) < 0:
            #breakpoint()
            # print('current_dist = ', current_dist)
            # Loop through all segments
            for idx in range(num_total_segs):
                # breakpoint()
                # print(idx, dtt[idx], mycelia['nbr_idxs'][idx], dtt[mycelia['nbr_idxs'][idx]])
                #breakpoint()
                # Only consider segments that have not yet been assigned (i.e dtt = -1)
                # Only consider segments with a neighbor that has dtt=current_dist-1 
                if dtt[idx] == -1 and (current_dist-1) in dtt[mycelia['nbr_idxs'][idx]]:
                    dtt[idx] = current_dist
                    
            # Increase the current distance from the tip         
            current_dist += 1
            # This should not happen
            # Only triggered if a segment is missed
            # if current_dist > num_total_segs:
            #     breakpoint()
            
        # breakpoint()
        
        return dtt
    else:
        
        # Initialize dist to tip as all ones
        dtt = -1*np.ones((num_total_segs,1))
        non_null_segs = np.where(mycelia['branch_id'][:num_total_segs] > -1)[0]
        null_segs = np.where(mycelia['branch_id'][:num_total_segs] == -1)[0]
        if any(i in null_segs for i in non_null_segs):
            breakpoint()
        # If a segment is a tip, it has a distance to tip of 0
        tip_segs = np.where(mycelia['is_tip'][:num_total_segs])[0]
        dtt[tip_segs] = 0
        dtt[null_segs] = 1e12
        
        # If a segment is a neighbor of a tip, it has a distance to tip (dtt) of 1
        # If a segment is a neighbor of a segment with dtt of i, it has a dtt of i+1
        current_dist = 1
        while min(dtt[non_null_segs]) < 0:
            #breakpoint()
            # print('current_dist = ', current_dist)
            # Loop through all segments
            for idx in range(num_total_segs):
                # Only consider segments that have not yet been assigned (i.e dtt = -1)
                # Only consider segments with a neighbor that has dtt=current_dist-1 
                if dtt[idx] == -1 and (current_dist-1) in dtt[mycelia['nbr_idxs'][idx]]:
                    lead_dist = np.where(dtt[mycelia['nbr_idxs'][idx]]==(current_dist-1))[0]
                    if len(lead_dist) == 1 and (mycelia['branch_id'][idx] not in mycelia['branch_id'][mycelia['nbr_idxs'][idx][lead_dist[0]]]):
                        check_if_no_tip_in_branch = np.where(mycelia['branch_id'][:num_total_segs] == mycelia['branch_id'][idx])
                        if all(mycelia['is_tip'][check_if_no_tip_in_branch]==False):
                            dtt[idx] = current_dist
                        else:
                            continue
                    else:    
                        dtt[idx] = current_dist
                    
            # Increase the current distance from the tip         
            current_dist += 1
            # This should not happen
            # Only triggered if a segment is missed
            # if current_dist > num_total_segs:
            #     breakpoint()
            
        # breakpoint()
        
        return dtt
# # ----------------------------------------------------------------------------
    
def transloc(mycelia, num_total_segs, dtt, isActiveTrans, whichInitialCondition,
             isConvectDependOnMetabo_cw,
             isConvectDependOnMetabo_gluc,
             isConvectDependOnMetabo_treha):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.
        
    Purpose
    -------
    Calculate the change in nutrients due to translocation (diffusion of glucose,
    conversion of glucose to cell wall materials, active transport of cell wall
    materials)
    """
    # if (num_total_segs >= 9):
    #     breakpoint()
    # Conversion Term: How much glucose is used by metabolism? (Actually, all of it
    # so I think the update of gluc_i needs to reflect that)
    use_original = 0
    if(use_original != 1):
    	convert_term = gf.michaelis_menten(params['kc1_gluc'], 
                          params['Kc2_gluc'], 
                          mycelia['gluc_i'][:num_total_segs])
    else:
    	convert_term = gf.michaelis_menten(params['kc1_gluc'], 
                                                params['Kc2_gluc'], 
                                                mycelia['gluc_i'][:num_total_segs])
    if (np.isnan(np.sum(convert_term))):
            breakpoint()
    #convert_term[np.where(mycelia['is_tip'])] = 0 #Why do this? Why can't the tip have metabolism?

    # Matrix of values for seg j
    # This next line is not correct - the glucose values are at steady state with respect to metabolism
    # mycelia['gluc_i'][:num_total_segs] = mycelia['gluc_i'][:num_total_segs] - params['dt']*convert_term
    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_gluc_i_idx)>0:
        print('Glucose below 0.0:',np.min(mycelia['gluc_i'][:num_total_segs]))
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;
        #breakpoint()
    gluc_curr = mycelia['gluc_i'][:num_total_segs]
#    if(np.any(gluc_curr < 0)):
#        print('Glucose below 0.0:',np.min(gluc_curr))
#        breakpoint()
    if(np.any(mycelia['gluc_i'][:num_total_segs] < 0)):
        breakpoint()
    cw_curr = mycelia['cw_i'][:num_total_segs]
    treha_curr = mycelia['treha_i'][:num_total_segs]
    # if (np.max(mycelia['treha_i'][:num_total_segs])>1e1):
    #     breakpoint()
    

    
    # Glucose & cell wall concs in neighboring cells summed up
    nbr_curr = mycelia['nbr_idxs'][:num_total_segs]
    to_nbrs = []
    from_nbrs = []
    gluc_nbrs = np.zeros((num_total_segs,1))
    treha_nbrs = np.zeros((num_total_segs,1))
    for idx in range(num_total_segs):
        
        # if idx == 2:
        #     breakpoint()
        
        nbr_of_idx = np.array(nbr_curr[idx])
        
        if len(nbr_of_idx) > 0:
            TESTO = np.where(mycelia['bypass'][nbr_of_idx]==True)[0]
            # if TESTO:
            #     print('wow wow wow... WTF')
            #     breakpoint()
            #if len(nbr_of_idx)<1:
               # breakpoint()
          
        if (mycelia['bypass'][idx]==True):
            to_nbrs.append([])
            from_nbrs.append([])
            continue
        # breakpoint()
        if len(nbr_of_idx)<1:
            to_nbrs.append([])
        elif len(np.where(dtt[nbr_of_idx]<=dtt[idx])[0]) and (mycelia['branch_id'][idx])>-1: 
            chosen_idx = np.array(np.where(dtt[nbr_of_idx]<=dtt[idx])[0])
            
            
            if len(chosen_idx)>len(dtt[nbr_of_idx]):
                breakpoint()
            elif len(chosen_idx) < 1:
                breakpoint()
            chosen_idx = list(chosen_idx)
            candidate_for_deletion = chosen_idx.copy()
            for i in range(len(candidate_for_deletion)):
                # print(candidate_for_deletion[i])
                if mycelia['branch_id'][nbr_of_idx[candidate_for_deletion[i]]]==-1:
                    # print('Removing : ', candidate_for_deletion[i])
                    if candidate_for_deletion[i] not in chosen_idx:
                        breakpoint()
                    chosen_idx.remove(candidate_for_deletion[i])
            to_nbrs.append(nbr_of_idx[chosen_idx].tolist())
           
        else:
            to_nbrs.append([])
                
        if len(nbr_of_idx)<1:
            from_nbrs.append([])
        elif len(np.where(dtt[nbr_of_idx]>=dtt[idx])[0]) and (mycelia['branch_id'][idx])>-1:
            chosen_idx = np.array(np.where(dtt[nbr_of_idx]>=dtt[idx])[0])
            
            if len(chosen_idx)>len(dtt[nbr_of_idx]):
                breakpoint()
            elif len(chosen_idx) < 1:
                breakpoint()
            
            chosen_idx = list(chosen_idx)
            candidate_for_deletion = chosen_idx.copy()
            for i in range(len(candidate_for_deletion)):
                # print(candidate_for_deletion[i])
                if mycelia['branch_id'][nbr_of_idx[candidate_for_deletion[i]]]==-1:
                    # print('Removing : ', candidate_for_deletion[i])
                    if candidate_for_deletion[i] not in chosen_idx:
                        breakpoint()
                    chosen_idx.remove(candidate_for_deletion[i])
            from_nbrs.append(nbr_of_idx[chosen_idx].tolist())
        else:
            from_nbrs.append([])

        gluc_nbrs[idx] = np.sum(mycelia['gluc_i'][nbr_curr[idx]]) 
        treha_nbrs[idx] = np.sum(mycelia['treha_i'][nbr_curr[idx]])
        
    to_nbrs = np.array(to_nbrs)
    len_to_nbrs = np.array([len(to_nbrs[i]) for i in range(len(to_nbrs))]).reshape(-1,1)

    # Diffusion Term: sum_{nbr in nbrs} (D/L)*(nbr - self)
    seg_lengths = mycelia['seg_length'][:num_total_segs]
 
    gluc_diff_term = params['diffusion_i_gluc']/(seg_lengths*0.5*(seg_lengths + params['sl']))*(
        gluc_nbrs - mycelia['nbr_num'][:num_total_segs]*gluc_curr) 
    if (np.any(gluc_diff_term)<0):
        breakpoint()
    treha_diff_term = params['diffusion_i_gluc']/(seg_lengths*0.5*(seg_lengths + params['sl']))*(
        treha_nbrs - mycelia['nbr_num'][:num_total_segs]*treha_curr) 
    # Sometimes the seg_lengths are too short for diffusion to be occuring. Set the diffusion term for these seg_lengths to zero
    gluc_diff_term[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0
    treha_diff_term[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0
    # gluc_diff_term[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0.1*gluc_diff_term[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]]
    # treha_diff_term[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0.1*treha_diff_term[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]]
    # if (np.max(treha_diff_term)>1):
    #     breakpoint()

    # Conversion Term: How much glucose is used by metabolism? (Actually, all of it
    # so I think the update of gluc_i needs to reflect that)
    
    # Convection Term: (set conc in tip segments to )
    cw_from_scaled = np.zeros((num_total_segs,1))
    treha_from_scaled = np.zeros((num_total_segs,1))
    gluc_from_scaled = np.zeros((num_total_segs,1))
    # cw_to_scaled = np.zeros((num_total_segs,1))
    # treha_to_scaled = np.zeros((num_total_segs,1))
    # gluc_to_scaled = np.zeros((num_total_segs,1))
    
    for idx in range(num_total_segs):
        if mycelia['branch_id'][idx] == -1:
            continue
        
        if idx >= len(from_nbrs):
            breakpoint()
        from_nbrs_idx = from_nbrs[idx]
        if len(from_nbrs_idx):
            if np.isnan(sum(seg_lengths[from_nbrs_idx])):
                breakpoint()
            # The amount of cell wall material transported is the product of the cell wall concentration in the vessicle and
            # the velocity of translocation the vessicle (params['vel_wall'])
            # divided by the distance it must be transported (seg_length). Rather than divided the velocity of 
            # transport of the vessicle by the length (vel_wall/seg_length), it is more convenient to divide the
            # concentration in the vessible by the length:
            # cw_from_scaled[idx] = sum(cw_curr[from_nbrs_idx]/(seg_lengths[from_nbrs_idx]*len_to_nbrs[from_nbrs_idx]))
            cw_from_scaled[idx] = np.nansum(cw_curr[from_nbrs_idx]/(seg_lengths[from_nbrs_idx]*len_to_nbrs[from_nbrs_idx]))
            treha_from_scaled[idx] = np.nansum(treha_curr[from_nbrs_idx]/(seg_lengths[from_nbrs_idx]*len_to_nbrs[from_nbrs_idx]))
            gluc_from_scaled[idx] = np.nansum(gluc_curr[from_nbrs_idx]/(seg_lengths[from_nbrs_idx]*len_to_nbrs[from_nbrs_idx]))
            
            # cw_to_scaled[idx] = np.nansum(cw_curr[idx]/(seg_lengths[idx]*len_to_nbrs[idx]))
            # treha_to_scaled[idx] = np.nansum(treha_curr[idx]/(seg_lengths[idx]*len_to_nbrs[idx]))
            # gluc_to_scaled[idx] = np.nansum(gluc_curr[idx]/(seg_lengths[idx]*len_to_nbrs[idx]))
            if np.isnan(cw_from_scaled[idx]):
                breakpoint()
            if np.isnan(treha_from_scaled[idx]):
                breakpoint()
            # if (np.max(treha_from_scaled)>1):
            #     breakpoint()
            if np.any(gluc_from_scaled<0):
                breakpoint()
            
    cw_curr_mod = cw_curr/seg_lengths #This is the amount of cell wall material already present (before metabolism made more)
                                      #in the hyphal compartment 
                                      #that can also be transported out of the compartment.
    treha_curr_mod = treha_curr/seg_lengths
    gluc_curr_mod = gluc_curr/seg_lengths
    
    if (whichInitialCondition == 0):
        whichTipIsOrigin = np.where(mycelia['is_tip'][:num_total_segs])[0]
        #whichTipIsOrigin = np.delete(whichTipIsOrigin, np.where(whichTipIsOrigin<4)[0])
        cw_curr_mod[whichTipIsOrigin] = 0
        treha_curr_mod[whichTipIsOrigin] = 0
        gluc_curr_mod[whichTipIsOrigin] = 0
    elif (whichInitialCondition == 1):
        whichTipIsOrigin = np.where(mycelia['is_tip'][:num_total_segs])[0]
        #whichTipIsOrigin = np.delete(whichTipIsOrigin, np.where(whichTipIsOrigin<4)[0])
        #if 0 in whichTipIsOrigin:
        #    whichTipIsOrigin.pop(0)
        #if 1 in whichTipIsOrigin:
        #    whichTipIsOrigin.pop(1)
        cw_curr_mod[whichTipIsOrigin] = 0
        treha_curr_mod[whichTipIsOrigin] = 0
        gluc_curr_mod[whichTipIsOrigin] = 0
    
    # cw_to_scaled[np.where(mycelia['is_tip'][:num_total_segs])[0]] = 0
    # treha_to_scaled[np.where(mycelia['is_tip'][:num_total_segs])[0]] = 0
    # gluc_to_scaled[np.where(mycelia['is_tip'][:num_total_segs])[0]] = 0
    # if (np.max(treha_curr_mod)>1):
    #     breakpoint()
    
    # cw_curr_mod[np.where(mycelia['branch_id'][:num_total_segs]<0)[0]] = 0
    # treha_curr_mod[np.where(mycelia['branch_id'][:num_total_segs]<0)[0]] = 0

    cw_diff = cw_from_scaled - cw_curr_mod
    treha_diff = treha_from_scaled - treha_curr_mod
    gluc_diff = gluc_from_scaled - gluc_curr_mod
    # cw_diff = (cw_from_scaled - cw_to_scaled) + cw_curr_mod
    # treha_diff = (treha_from_scaled - treha_to_scaled) + treha_curr_mod
    # gluc_diff = (gluc_from_scaled - gluc_to_scaled) + gluc_curr_mod
    # if (np.max(treha_diff)>1):
    #     breakpoint()
    # Try different ways of forumating the translocation term for cell wall material:

    #cw_convect_term = params['vel_wall']*cw_diff 
    # cw_convect_term = params['vel_wall']*cw_diff/(
    #                     5.0e-11 + cw_diff)*cw_diff
    # treha_convect_term = params['vel_wall']*treha_diff/(
    #                     5.0e-11 + treha_diff)*treha_diff
    # gluc_convect_term = params['vel_wall']*gluc_diff/(
    #                     5.0e-11 + gluc_diff)*gluc_diff
    cw_convect_term = np.zeros((num_total_segs,1))
    gluc_convect_term = np.zeros((num_total_segs,1))
    treha_convect_term = np.zeros((num_total_segs,1))
    
    if (isConvectDependOnMetabo_cw == 1 or isConvectDependOnMetabo_gluc == 1 or isConvectDependOnMetabo_treha == 1):
        # for idx in range(num_total_segs):
            if (isConvectDependOnMetabo_cw == 1):
                # cw_convect_term[idx] = gf.michaelis_menten(cw_convect_term[idx], 
                #                                       params['Ku2_gluc'], 
                #                                       mycelia['gluc_i'][idx])
                cw_convect_term = cw_diff*gf.michaelis_menten(params['vel_wall'], 
                                                      params['Ku2_gluc'], 
                                                      mycelia['gluc_i'][:num_total_segs])
                
            else:
                cw_convect_term = params['vel_wall']*cw_diff/(
                        5.0e-11 + cw_diff)*cw_diff
                # breakpoint()
            if (isConvectDependOnMetabo_gluc == 1):
                # Amount in mmol taken up by each segment -RATE IS NOT CONCENTRATION/sec
                # gluc_convect_term[idx] = gf.michaelis_menten(gluc_convect_term[idx], 
                #                                         params['Ku2_gluc'], 
                #                                         mycelia['gluc_i'][idx])
                gluc_convect_term = gluc_diff*gf.michaelis_menten(params['vel_wall'], 
                                                      params['Ku2_gluc'], 
                                                      mycelia['gluc_i'][:num_total_segs])
                # breakpoint()
            else:
                gluc_convect_term = params['vel_wall']*gluc_diff/(
                                    5.0e-11 + gluc_diff)*gluc_diff
                # breakpoint()
            if (isConvectDependOnMetabo_treha == 1):
                # treha_convect_term[idx] = gf.michaelis_menten(treha_convect_term[idx], 
                #                                          params['Ku2_gluc'], 
                #                                          mycelia['gluc_i'][idx])
                treha_convect_term = treha_diff*gf.michaelis_menten(params['vel_wall'], 
                                                      params['Ku2_gluc'], 
                                                      mycelia['gluc_i'][:num_total_segs])
                # print('treha_convect_term : ', treha_convect_term)
            else:
                treha_convect_term = params['vel_wall']*treha_diff/(
                                    5.0e-11 + treha_diff)*treha_diff
    else:
        cw_convect_term = params['vel_wall']*cw_diff/(
                            5.0e-11 + cw_diff)*cw_diff
        treha_convect_term = params['vel_wall']*treha_diff/(
                            5.0e-11 + treha_diff)*treha_diff
        gluc_convect_term = params['vel_wall']*gluc_diff/(
                            5.0e-11 + gluc_diff)*gluc_diff                
                # breakpoint()
        # breakpoint()
    cw_convect_term[np.where(mycelia['branch_id'][:num_total_segs]<0)[0]] = 0
    treha_convect_term[np.where(mycelia['branch_id'][:num_total_segs]<0)[0]] = 0
    gluc_convect_term[np.where(mycelia['branch_id'][:num_total_segs]<0)[0]] = 0
    if (np.isnan(np.sum(cw_convect_term))):
            breakpoint()
    if (np.isnan(np.sum(treha_convect_term))):
            breakpoint()
    if (np.isnan(np.sum(gluc_convect_term))):
            breakpoint()
        
    # Update glucose & cell wall concs
    #mycelia['gluc_i'][:num_total_segs] += params['dt']*(gluc_diff_term - convert_term)
    #mycelia['gluc_i'][:num_total_segs] = params['dt']*(gluc_diff_term - convert_term)
    mycelia['gluc_i'][:num_total_segs] += params['dt']*gluc_diff_term
    mycelia['treha_i'][:num_total_segs] += params['dt']*treha_diff_term

    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_gluc_i_idx)>0:
        mycelia['gluc_i'][negative_gluc_i_idx] = 0.0;
    # 
    # yield_c is the fraction of glucose mass converted to mass of chitobiose and glucan,
    # so need yield_c * mw_glucose / mw_cell_wall_component
    if(use_original != 1):
        mycelia['cw_i'][:num_total_segs] += params['dt']*(params['yield_c_in_mmoles']*convert_term)
        #mycelia['treha_i'][:num_total_segs] += params['dt']*(convert_term*0.3*0.1)
        mycelia['treha_i'][:num_total_segs] += params['dt']*(convert_term*0.1)
        if (isActiveTrans == 1):
            mycelia['cw_i'][:num_total_segs] += params['dt']*(cw_convect_term)
            # mycelia['treha_i'][:num_total_segs] += params['dt']*(treha_convect_term + params['yield_c_in_mmoles']*convert_term*0.3*0.1)
            mycelia['treha_i'][:num_total_segs] += params['dt']*(treha_convect_term)
            mycelia['gluc_i'][:num_total_segs] += params['dt']*(gluc_convect_term)
            # if (np.max(mycelia['gluc_i'][:num_total_segs])>1):
            #     breakpoint()
        # else:
        #     mycelia['cw_i'][:num_total_segs] += params['dt']*(params['yield_c_in_mmoles']*convert_term*0.3)
        #     # mycelia['treha_i'][:num_total_segs] += params['dt']*(treha_convect_term + params['yield_c_in_mmoles']*convert_term*0.3*0.1)
        #     mycelia['treha_i'][:num_total_segs] += params['dt']*(convert_term*0.3*0.1)
        #     mycelia['gluc_i'][:num_total_segs] += params['dt']*(gluc_convect_term)
            # if (np.max(mycelia['treha_i'][:num_total_segs])>1e1):
            #     breakpoint()
    else:
    	mycelia['cw_i'][:num_total_segs] += params['dt']*(cw_convect_term + convert_term)
    # Finally, update concentrations due to metabolic activity:
    convert_term = gf.michaelis_menten(params['kc1_gluc'], 
                          params['Kc2_gluc'], 
                          mycelia['gluc_i'][:num_total_segs])
    if (np.any(mycelia['gluc_i'][:num_total_segs] < params['dt']*convert_term)):
        breakpoint()
    mycelia['gluc_i'][:num_total_segs] = mycelia['gluc_i'][:num_total_segs] - params['dt']*convert_term

    negative_cw_i_idx = np.where(mycelia['cw_i'][:num_total_segs] < 0)[0]
    if len(negative_gluc_i_idx)>0:
            print('Glucose below 0.0 after convert_term :',np.min(mycelia['gluc_i'][:num_total_segs]))
            mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;
    negative_treha_i_idx = np.where(mycelia['treha_i'][:num_total_segs] < 0)[0]
    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_cw_i_idx)>0:
        mycelia['cw_i'][negative_cw_i_idx] = np.finfo(np.float64).tiny;
    if len(negative_treha_i_idx)>0:
        mycelia['treha_i'][negative_treha_i_idx] = np.finfo(np.float64).tiny;
    if len(negative_gluc_i_idx)>0:
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;

    # if min(mycelia['gluc_i'][:num_total_segs] < 0):
    #     breakpoint()
    # if min(mycelia['cw_i'][:num_total_segs] < 0):
    #     breakpoint()
    
    # breakpoint()
    if(np.any(mycelia['gluc_i'][:num_total_segs] < 0)):
        breakpoint()
    
    return mycelia
    

# ----------------------------------------------------------------------------
# UPTAKE FUNCTIONS
# ----------------------------------------------------------------------------
    
def uptake(sub_e_gluc, mycelia, num_total_segs):
    """
    Parameters
    ----------
    sub_e_gluc : array (2D)
        The 2D grid storing values of glucose in the external domain.
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    """
    
    # All indicies of external grid used
    xy_e_idx_og = mycelia['xy_e_idx'][:num_total_segs, :].astype(int)
    # breakpoint()
    # Reformat indicies
    xy_e_idx = tuple(np.transpose(xy_e_idx_og))
    
    # Glucose mmole values at grid points, not mMolar!
    gluc_e = sub_e_gluc[xy_e_idx].copy()
    
    # Glucose inside the hyphae
    gluc_i = mycelia['gluc_i'][:num_total_segs].flatten()
    #if(np.any(gluc_i < 1.0e-16)):
    #    breakpoint()
    
    # Amount in mmol taken up by each segment -RATE IS NOT CONCENTRATION/sec
    # gluc_uptake = params['dt']*gluc_e*gf.michaelis_menten(params['ku1_gluc'], 
    #                                                       params['Ku2_gluc'], 
    #                                                       gluc_i)
    # Ku2_gluc units are concentration in mmole/(micron)^ in parameters.ini
    # but are changed to units of mmole in helper_functions.get_configs()
    # Since a hyphae is considered to live at a single grid point, the uptake 
    # of glucose in the hyphae is from the grid at which the center of the hyphae
    #gluc_uptake = params['dt']*gf.michaelis_menten(params['ku1_gluc'], 
    #                                              params['Ku2_gluc'], 
    #                                              gluc_e)
    # Could use a different rate of uptake depending on the hyphal size,
    # but I have not seen anyone indicate that the uptake rate is a function of cell/hyphae size.
    # But this is probably the case
    relative_seg_vol = mycelia['seg_vol'][:num_total_segs].flatten()/params['init_vol_seg']
    
    #if any(relative_seg_vol == 0):
    #    breakpoint()
    gluc_uptake = params['dt']*gf.michaelis_menten(params['ku1_gluc'], 
                                                  params['Ku2_gluc']/relative_seg_vol, 
                                                  gluc_e)
    #gluc_uptake[np.where(relative_seg_vol <1e-15)] = 0
    seg_lengths = mycelia['seg_length'][:num_total_segs]
    gluc_uptake[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0.0
    # gluc_uptake[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0.1*gluc_uptake[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]]

    for i in range(num_total_segs):
        if mycelia['branch_id'][i] < 0:
            gluc_uptake[i] = 0.0
    
    # List of a list containing segment IDs, if inner list length > 1, the IDs in same grid cell
    my_share = mycelia['share_e'][:num_total_segs]

    # Original amount taken up in each grid cell
    gluc_up_sum = np.array([np.sum(gluc_uptake[i]) for i in my_share])
    # breakpoint()
    
    if np.min(gluc_e - gluc_up_sum) >= 0:
        #breakpoint()
        # Update the amount of glucose in external grid cells
        sub_e_gluc[xy_e_idx] = gluc_e - gluc_up_sum
        
        # Update the amount of glucose in mmoles in internal segments
        mycelia['gluc_i'][:num_total_segs] += params['yield_u']*gluc_uptake.reshape(-1,1)
        # breakpoint()
    else:
        #breakpoint()
        # Find the cells where too much is taken up
        raw_difference = gluc_e - gluc_up_sum
        raw_difference_neg_idx = np.where(raw_difference < 0)
        raw_difference_pos_idx = np.where(raw_difference >=0)
        sub_e_gluc[xy_e_idx] = gluc_e - gluc_up_sum
        
        sub_e_gluc[xy_e_idx[0][raw_difference_neg_idx], xy_e_idx[1][raw_difference_neg_idx]] = 0.0
        mycelia['gluc_i'][:num_total_segs] += params['yield_u']*gluc_uptake.reshape(-1,1)
        # Modify gluc_uptake & gluc_up_sum
        #breakpoint()
    # breakpoint()
    return mycelia

def release(sub_e_treha, mycelia, num_total_segs, isTipRelease):
    """
    Parameters
    ----------
    sub_e_gluc : array (2D)
        The 2D grid storing values of glucose in the external domain.
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    """
    
    tip_release = isTipRelease#1
    # All indicies of external grid used
    xy_e_idx_og = mycelia['xy_e_idx'][:num_total_segs, :].astype(int)
    # breakpoint()
    # Reformat indicies
    xy_e_idx = tuple(np.transpose(xy_e_idx_og))
    
    # Trehalose mmole values at grid points
    treha_e = sub_e_treha[xy_e_idx].copy()
    
    # Trehalose inside the hyphae
    treha_i = mycelia['treha_i'][:num_total_segs].flatten()
    # if (np.max(treha_i)>1e1):
    #     breakpoint()
    
    relative_seg_vol = mycelia['seg_vol'][:num_total_segs].flatten()/params['init_vol_seg']
    #treha_release = gf.michaelis_menten(params['kc1_gluc'], 
    #                      params['Kc2_gluc'], 
    #                      treha_i)*params['dt']
    #treha_release = (treha_i/mycelia['seg_vol']) / (1.0e-18 + treha_e/params['vol_grid'] )
    treha_release = (treha_i/mycelia['seg_vol'][:num_total_segs][0] - treha_e/params['vol_grid'] )/2

    seg_lengths = mycelia['seg_length'][:num_total_segs]
    treha_release[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0.0
    # treha_release[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] = 0.1*treha_release[np.where(seg_lengths*seg_lengths < 0.1*params['diffusion_i_gluc'])[0]] 
    # treha_release = treha_i*0.1
    if tip_release == 0:
        negative_branch_ids = np.where(mycelia['branch_id'][:num_total_segs]<0)[0]
    else:
        negative_branch_ids = np.where(mycelia['branch_id'][:num_total_segs]<0)[0]  
        nontip_ids = np.where(mycelia['is_tip'][:num_total_segs]==False)[0]
    treha_release[negative_branch_ids] = 0.0
    if tip_release == 1:
        treha_release[nontip_ids] = 0.0
    # for i in range(num_total_segs):
    #     if mycelia['branch_id'][i] < 0:
    #         treha_release[i] = 0.0
    
    sub_e_treha[xy_e_idx] = sub_e_treha[xy_e_idx]+treha_release
    # breakpoint()
    mycelia['treha_i'][:num_total_segs] = mycelia['treha_i'][:num_total_segs] - (treha_release).reshape(-1,1)
    # if (np.max(mycelia['treha_i'][:num_total_segs])>1e1):
    #     breakpoint()
    
    return mycelia


