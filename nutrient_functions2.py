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
    r_coeff = (params['dt_e']*params['diffusion_e_gluc'])/(2*params['dy']**2)
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
    r_coeff = (params['dt_e']*params['diffusion_e_gluc'])/(2*params['dy']**2)
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
    # Conversion Term: How much glucose is used by metabolism? (Actually, all of it
    # so I think the update of gluc_i needs to reflect that)
    use_original = 0
    alpha_gluc = gf.michaelis_menten(1, 
                          params['Kc2_gluc'], 
                          mycelia['gluc_i'][:num_total_segs])


    # Matrix of values for seg j
    # This next line is not correct - the glucose values are at steady state with respect to metabolism
    # mycelia['gluc_i'][:num_total_segs] = mycelia['gluc_i'][:num_total_segs] - params['dt']*convert_term
    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_gluc_i_idx)>0:
        print('Glucose below 0.0:',np.min(mycelia['gluc_i'][:num_total_segs]))
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;
        #breakpoint()
    gluc_curr = mycelia['gluc_i'][:num_total_segs]
    #seg_volume = mycelia['seg_vol'][:num_total_segs] mycelia['seg_vol'] doesn't appear to be getting updated 
    seg_lengths = mycelia['seg_length'][:num_total_segs]
    seg_volume = seg_lengths*params['cross_area']
    gluc_curr_concentrations = gluc_curr/seg_volume

    if(np.any(gluc_curr < 0)):
        print('Glucose below 0.0:',np.min(gluc_curr))
        breakpoint()
    cw_curr = mycelia['cw_i'][:num_total_segs]
    treha_curr = mycelia['treha_i'][:num_total_segs]
    cw_curr_concentrations = cw_curr/seg_volume
    treha_curr_concentrations = treha_curr/seg_volume

    # Diffusion Term: sum_{nbr in nbrs} (D/L)*(nbr - self)
    d2gluc_dx2 = np.zeros((num_total_segs,1))
    d2treha_dx2 = np.zeros((num_total_segs,1))
    
    # Glucose & cell wall concs in neighboring cells summed up
    nbr_curr = mycelia['nbr_idxs'][:num_total_segs]
    to_nbrs = []
    from_nbrs = []
    gluc_nbrs = np.zeros((num_total_segs,1))
    treha_nbrs = np.zeros((num_total_segs,1))

    delta_gluc_conc_nbrs = np.zeros((num_total_segs,1))
    nbr_length = np.zeros((num_total_segs,1))
    nbr_volume = np.zeros((num_total_segs,1))
    volume_use = np.zeros((num_total_segs,1))
    nbr_dist = np.zeros((num_total_segs,1))
    nbr_dist_sqr = np.zeros((num_total_segs,1))
    
    # Calculate neighbor lists and simultaneously determine diffusion
    for idx in range(num_total_segs):
        if(idx == 35367):
            xc = 1

        delta_gluc_conc_nbrs = delta_gluc_conc_nbrs*0.0
        nbr_length = nbr_length*0.0
        nbr_volume = nbr_volume*0.0
        volume_use = volume_use*0.0
        nbr_dist = nbr_dist*0.0
        nbr_dist_sqr = nbr_dist_sqr*0.0
        
        nbr_of_idx = np.array(nbr_curr[idx])
        if 35367 in nbr_of_idx:
            xc = 1
          
        if (mycelia['bypass'][idx]==True):
            to_nbrs.append([])
            from_nbrs.append([])
            continue
        
        # Advection is to the closest tip. Find the immediate neighbors that are closer to tips.
        # If a segment is equally close to two tips, then the advection is toward both tips.
        if len(np.where(dtt[nbr_of_idx] < dtt[idx])[0]) and (mycelia['branch_id'][idx])>-1: 
            chosen_idx = np.array(np.where(dtt[nbr_of_idx] < dtt[idx])[0])
            
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
        elif len(nbr_of_idx)<1:
            to_nbrs.append([])   
        else:
            to_nbrs.append([])

        # Find the immediate neighbors that are further from tip.
        # A hyphal segment will accept flow from these neighbors        
        if len(np.where(dtt[nbr_of_idx] > dtt[idx])[0]) and (mycelia['branch_id'][idx])>-1:
            chosen_idx = np.array(np.where(dtt[nbr_of_idx] > dtt[idx])[0])
            
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
                # If a neighbor is a tip, don't export material from tip - it goes into growth instead
                if mycelia['is_tip'][nbr_of_idx[candidate_for_deletion[i]]]==True:
                    # print('Removing : ', candidate_for_deletion[i])
                    if candidate_for_deletion[i] not in chosen_idx:
                        breakpoint()
                    chosen_idx.remove(candidate_for_deletion[i])
            from_nbrs.append(nbr_of_idx[chosen_idx].tolist())
        elif len(nbr_of_idx)<1:
            from_nbrs.append([])
        else:
            from_nbrs.append([])

        # This is the total count of gluc_i (glucose) in neighbors of idx, not concentrations!
        gluc_nbrs[idx] = np.sum(mycelia['gluc_i'][nbr_curr[idx]]) 
        treha_nbrs[idx] = np.sum(mycelia['treha_i'][nbr_curr[idx]])
        
        # These are concentrations:
        delta_gluc_conc_nbrs = gluc_curr_concentrations[nbr_curr[idx]] - gluc_curr_concentrations[idx]
        delta_treha_conc_nbrs = treha_curr_concentrations[nbr_curr[idx]] - treha_curr_concentrations[idx]

        # If the concentration of the neighbors is higher than teh concentration of idx, then the net flow
        # is out of idx to neighbors
        # diffuse2nbrs is True/False flag for which neigbors to diffuse to
        diffuse2nbrs = [(delta_gluc_conc_nbrs < 0)]
        frxn_delta_gluc_conc2nbrs = np.zeros((len(nbr_curr[idx]),1))
        total_delta_gluc_conc_nbrs = np.sum(delta_gluc_conc_nbrs[diffuse2nbrs])
        # Determine the frxn of glucose in idx that will go to neighbors
        frxn_delta_gluc_conc2nbrs[diffuse2nbrs] = delta_gluc_conc_nbrs[diffuse2nbrs] \
            /total_delta_gluc_conc_nbrs

        nbr_length = seg_lengths[nbr_curr[idx]]
        nbr_volume = seg_volume[nbr_curr[idx]]
        volume_use_gluc = np.zeros((len(nbr_curr[idx]),1))

        # Determine whether to remove material from the neighbor or from the current segment:
        volume_use_gluc[(delta_gluc_conc_nbrs >= 0)] = nbr_volume[(delta_gluc_conc_nbrs >= 0)]
        volume_use_gluc[(delta_gluc_conc_nbrs < 0)] = seg_volume[idx]

        # Count of the Number of Neighbors (cnn) that neighbors of idx have:
        #nn = np.zeros((len(nbr_curr[idx]),1))
        #cnn = [len(nbr_curr[i]) for i in nbr_of_idx]
        #nn[(delta_gluc_conc_nbrs > 0)] = cnn[(delta_gluc_conc_nbrs > 0)]
        #nn[(delta_gluc_conc_nbrs <= 0)] = len(nbr_of_idx)
        # The amount taken from a cell must be split amoung its neighbors, so
        # that the amount taken doesn't exceed the total available.
        # Likewise the amount added to a cell is only a fraction of what is transported out of the neighbor

        volume_use_treha = np.zeros((len(nbr_curr[idx]),1))
        volume_use_treha[(delta_treha_conc_nbrs >= 0)] = nbr_volume[(delta_treha_conc_nbrs >= 0)]
        volume_use_treha[(delta_treha_conc_nbrs < 0)] = seg_volume[idx]

        # The distance of transport is assumed to be half the segment length of the current segment 
        # and half of the previous segment. 
        nbr_dist = 0.5*(nbr_length+seg_lengths[idx])
        nbr_dist_sqr = nbr_dist*nbr_dist
        # d2gluc_dx2[idx] = np.sum(delta_gluc_conc_nbrs/nbr_dist_sqr) would be the total change in concentration
        # due to diffusion. But we need to know what the change in counts are. So multiply the change in concentration 
        # due to each neighbor by the volume of the compartment that is losing concentration.
        #d2gluc_dx2[idx] =  np.sum(delta_gluc_conc_nbrs/nbr_dist_sqr*volume_use_gluc)
        
        #d2gluc_dx2[idx] =  d2gluc_dx2[idx] + np.sum(frxn_delta_gluc_conc2nbrs*delta_gluc_conc_nbrs \
        #                    /nbr_dist_sqr *volume_use_gluc)
        d2gluc_dx2[idx] =  d2gluc_dx2[idx] + np.sum(delta_gluc_conc_nbrs*volume_use_gluc/seg_volume[idx] \
                            /nbr_dist_sqr )
        
        # Note that nbr_of_idx contains both ''to_nbrs' and 'from_nbrs'.
        # nbr_of_idx can be used instead of 'to_nbr_idx' because the fractional change of 
        # 'from_nbr_idx' is zero.
        #d2gluc_dx2[nbr_of_idx] = d2gluc_dx2[nbr_of_idx] \
        #                    -frxn_delta_gluc_conc2nbrs*delta_gluc_conc_nbrs \
        #                    /nbr_dist_sqr *volume_use_gluc
        d2gluc_dx2[nbr_of_idx] =  d2gluc_dx2[nbr_of_idx] - delta_gluc_conc_nbrs*volume_use_gluc/seg_volume[idx] \
                            /nbr_dist_sqr

        #d2treha_dx2[idx] =  np.sum(delta_treha_conc_nbrs/nbr_dist_sqr*volume_use_treha)

    to_nbrs = np.array(to_nbrs,dtype=object)
    # len_to_neighbors = Number of neighbors including self:
    len_to_nbrs = np.array([len(to_nbrs[i]) for i in range(len(to_nbrs))]).reshape(-1,1)


    gluc_diff_term = params['diffusion_i_gluc']*d2gluc_dx2
    treha_diff_term = params['diffusion_i_gluc']*d2treha_dx2
    
    # Count tips  
    print("Number branches, segements:", max(mycelia['branch_id'])[0]+1, num_total_segs)

    # Update due to diffusion:
    mycelia_before = mycelia['gluc_i'][:num_total_segs].copy()
    mycelia['gluc_i'][:num_total_segs] += params['dt_i']*gluc_diff_term
    mycelia['treha_i'][:num_total_segs] += params['dt_i']*treha_diff_term
    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_gluc_i_idx)>0:
        print('Glucose before diffusion_term :',mycelia_before[negative_gluc_i_idx])
        print('Glucose below 0.0 after diffusion_term :',mycelia['gluc_i'][negative_gluc_i_idx])
        print('diffusion_term',gluc_diff_term[negative_gluc_i_idx])
        print('Indices:',negative_gluc_i_idx)
        print('Segment lengths:',mycelia['seg_length'][negative_gluc_i_idx])
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;
        breakpoint()
    negative_treha_i_idx = np.where(mycelia['treha_i'][:num_total_segs] < 0)[0]

    mycelia_before = mycelia['treha_i'][:num_total_segs].copy()
    if len(negative_treha_i_idx)>0:
        print('Trehalose before diffusion_term :',mycelia_before[negative_treha_i_idx])
        print('Trehalose below 0.0 after diffusion_term :',mycelia['treha_i'][negative_treha_i_idx])
        print('diffusion_term',gluc_diff_term[negative_treha_i_idx])
        mycelia['treha_i'][negative_treha_i_idx] = np.finfo(np.float64).tiny;
        breakpoint()

    print('Min, Max glucose counts:',np.min(gluc_curr), np.max(gluc_curr))
    print('Sum net trehalose diffusion',np.sum(treha_diff_term[:num_total_segs]))
    print('Sum net glucose diffusion',np.sum(gluc_diff_term[:num_total_segs]))
    print('Mean glucose diffusion',np.mean(np.abs(gluc_diff_term[:num_total_segs])))
    print('Max glucose diffusion',np.max(gluc_diff_term[:num_total_segs]))
    if(num_total_segs == 16):
        x = 1
    if(num_total_segs == 12):
        x = 1
    if(num_total_segs == 8):
        x = 1

    # Metabolism:
    # Update concentrations due to metabolic activity:
    alpha_gluc = gf.michaelis_menten(1, 
                          params['Kc2_gluc'], 
                          mycelia['gluc_i'][:num_total_segs])
    convert_term = params['kc1_gluc']*alpha_gluc
    if (np.isnan(np.sum(convert_term))):
            breakpoint()
    #convert_term[np.where(mycelia['is_tip'])] = 0 #Why do this? Why can't the tip have metabolism?
    if (np.any(mycelia['gluc_i'][:num_total_segs] - params['dt_i']*convert_term < 0)):
        bad_idx = np.where((mycelia['gluc_i'][:num_total_segs] - params['dt_i']*convert_term) < 0)
        print(bad_idx)
        print('Glucose before conversion:',mycelia['gluc_i'][bad_idx])
        print('Amount converted:',convert_term[bad_idx]*params['dt_i'])
        print('Convert rate:',convert_term[bad_idx])
        breakpoint()
    mycelia_before = mycelia['gluc_i'][:num_total_segs].copy()
    # Here glucose is converted to other metabolites:
    mycelia['gluc_i'][:num_total_segs] -= params['dt_i']*convert_term
    mycelia['cw_i'][:num_total_segs] += params['dt_i']*(params['yield_c_in_mmoles']*convert_term)
    #mycelia['treha_i'][:num_total_segs] += params['dt']*(convert_term*0.3*0.1)
    mycelia['treha_i'][:num_total_segs] += params['dt_i']*(convert_term*params['convert_for_export'])

    negative_cw_i_idx = np.where(mycelia['cw_i'][:num_total_segs] < 0)[0]
    negative_treha_i_idx = np.where(mycelia['treha_i'][:num_total_segs] < 0)[0]
    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_gluc_i_idx)>0:
        print('Glucose before convert_term :',mycelia_before[negative_gluc_i_idx])
        print('Glucose below 0.0 after convert_term :',mycelia['gluc_i'][negative_gluc_i_idx])
        print(mycelia['is_tip'][negative_gluc_i_idx])
        #if (np.min(mycelia['gluc_i'][:num_total_segs]) < 0.0):
        breakpoint()
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;

    if len(negative_cw_i_idx)>0:
        mycelia['cw_i'][negative_cw_i_idx] = np.finfo(np.float64).tiny;
    if len(negative_treha_i_idx)>0:
        mycelia['treha_i'][negative_treha_i_idx] = np.finfo(np.float64).tiny;
    if len(negative_gluc_i_idx)>0:
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;

    #print('Metabolism - gluc_curr:',mycelia['gluc_i'][:num_total_segs])

    # Advection:

    # Get current counts/concentrations after diffusion:
    gluc_curr = mycelia['gluc_i'][:num_total_segs]
    gluc_curr_concentrations = gluc_curr/seg_volume
    cw_curr = mycelia['cw_i'][:num_total_segs]
    treha_curr = mycelia['treha_i'][:num_total_segs]
    cw_curr_concentrations = cw_curr/seg_volume
    treha_curr_concentrations = treha_curr/seg_volume     
    
    # The concentration imported from neighboring segments is scaled by the number of neighbors that the neighboring
    #  segment must export to. For a linear hyphae, a middle segment must export to two neighbors. For a segment that is
    # one leg of an X, there are three neighbors. for a segment in the middle of a Y_ structure, there are likewise three - two at the top of the Y and
    # one at the bottom

    cw_curr_mod = cw_curr_concentrations #This is the amount of cell wall material already present (before metabolism made more)
                                      #in the hyphal compartment 
                                      #that can also be transported out of the compartment.
    treha_curr_mod = treha_curr_concentrations
    gluc_curr_mod = gluc_curr_concentrations

    # Don't export from tip
    cw_curr_mod[np.where(mycelia['is_tip'][:num_total_segs])[0]] = 0
 
    cw_delta_count = np.zeros((num_total_segs,1))
    gluc_delta_count = np.zeros((num_total_segs,1))
    treha_delta_count = np.zeros((num_total_segs,1))

    # Rate/velocity of active transport. The units here are concentraton/sec
    advection_vel_cw = params['advection_vel_cw']
    advection_vel_gluc = advection_vel_cw*params['yield_c']
    
    K_cw = advection_vel_cw*params['dt_i'] #K_cw is now the maximum amount of material that can be transported in dt.
    K_gluc = advection_vel_gluc*params['dt_i']
    
    # These logistic functions (alpha_*) will be used to scale the velocity due to metabolic activity. 
    # They prevent more material being removed than is present in the segment:
    alpha_cw = gf.michaelis_menten(1, K_cw, 
                                mycelia['cw_i'][:num_total_segs])   
    alpha_gluc2 = gf.michaelis_menten(1, K_gluc, 
                                mycelia['gluc_i'][:num_total_segs])

    cw_convect_term = np.zeros((num_total_segs,1))
    gluc_convect_term = np.zeros((num_total_segs,1))
    treha_convect_term = np.zeros((num_total_segs,1))

    for idx in range(num_total_segs):


        if mycelia['branch_id'][idx] == -1:
            continue
        
        if idx >= len(from_nbrs):
            breakpoint()
        from_nbrs_idx = from_nbrs[idx]
        to_nbrs_idx = to_nbrs[idx]
        from_nbr_volume = seg_volume[from_nbrs_idx]
        
        cw_from_scaled_nbrs = np.zeros((len(from_nbrs_idx),1))
        treha_from_scaled_nbrs = np.zeros((len(from_nbrs_idx),1))
        gluc_from_scaled_nbrs = np.zeros((len(from_nbrs_idx),1))

        if (len(from_nbrs_idx) + len(to_nbrs_idx) >0):
            if np.isnan(sum(seg_lengths[from_nbrs_idx])):
                breakpoint()
            # The amount of cell wall material transported is the product of the cell wall concentration in the vessicle and
            # the velocity of translocation the vessicle (determined by metabolism)
            # divided by the distance it must be transported (seg_length). Rather than divided the velocity of 
            # transport of the vessicle by the length (vel_wall/seg_length), it is more convenient to divide the
            # concentration in the vessicle by the length.
            # Also, a neigbor may export to many hyphae, so need to divide its contributino by the number of 
            # neighbors that it exports to = len_to_nbrs_idx.
            # cw_from_scaled[idx] = sum(cw_curr[from_nbrs_idx]/(seg_lengths[from_nbrs_idx]*len_to_nbrs[from_nbrs_idx]))
            # A hyphae may have many neighbors from which material is imported, such as in the initial x-structure, so take the sum
            # over all of these neighbors.
            cw_from_scaled_nbrs = cw_curr_concentrations[from_nbrs_idx]#/len_to_nbrs[from_nbrs_idx]
            treha_from_scaled_nbrs = treha_curr_concentrations[from_nbrs_idx]#/len_to_nbrs[from_nbrs_idx]
            gluc_from_scaled_nbrs = gluc_curr_concentrations[from_nbrs_idx]#/len_to_nbrs[from_nbrs_idx]
            volume_use_cw = np.zeros((len(from_nbrs_idx),1))
            
            # This is an array of differences in concentration
            cw_conc_diff = cw_from_scaled_nbrs - cw_curr_mod[idx]
            treha_conc_diff = treha_from_scaled_nbrs - treha_curr_mod[idx]
            gluc_conc_diff = gluc_from_scaled_nbrs - gluc_curr_mod[idx]
            # Get segment volumes for conversion of concentration to counts:
            exprt = (cw_conc_diff < 0)
            imprt = (cw_conc_diff >= 0)
            volume_use_cw[(cw_conc_diff < 0)] = seg_volume[idx] # If conc_diff < 0, outflow from idx
            volume_use_cw[(cw_conc_diff >= 0)] = from_nbr_volume[(cw_conc_diff > 0)] # If conc_diff > 0, inflow from idx
            # Change to counts taking from teh correct segment volume
            cw_delta_count[idx] = np.sum(cw_from_scaled_nbrs*from_nbr_volume) - cw_curr_mod[idx]*seg_volume[idx]
            exprt_amt = (len(to_nbrs[idx]) > 0) * 1.0
            # Advection without taking into account metabolism due to glucose
            #cw_convect_term[idx] = advection_vel_cw* \
            #    (np.sum(1/len_to_nbrs[from_nbrs_idx]*alpha_cw[from_nbrs_idx])-exprt_amt*alpha_cw[idx])
            # Advection taking into account metabolism due to glucose
            cw_convect_term[idx] = advection_vel_cw* \
                (np.sum(1/len_to_nbrs[from_nbrs_idx]*alpha_cw[from_nbrs_idx]*alpha_gluc[from_nbrs_idx])-exprt_amt*alpha_cw[idx]*alpha_gluc[idx])

            gluc_convect_term[idx] = advection_vel_gluc* \
                (np.sum(1/len_to_nbrs[from_nbrs_idx]*alpha_gluc[from_nbrs_idx]*alpha_gluc2[from_nbrs_idx])-exprt_amt*alpha_gluc[idx]*alpha_gluc2[idx])


            if np.isnan(cw_delta_count[idx]):
                breakpoint()
     
    if (np.isnan(np.sum(cw_convect_term))):
            breakpoint()
    if (np.isnan(np.sum(treha_convect_term))):
            breakpoint()
    if (np.isnan(np.sum(gluc_convect_term))):
            breakpoint()
    
    print('Mean glucose advection',np.mean(np.abs(gluc_convect_term[:num_total_segs])))
    print('Net glucose advection',np.sum(gluc_convect_term[:num_total_segs]))
    print('Net cell wl diff',np.sum(cw_delta_count[:num_total_segs]))
    print('Net cell wl advection',np.sum(cw_convect_term[:num_total_segs]))
    print('Mean cell wl difference',np.mean(np.abs(cw_delta_count[:num_total_segs])))
    print('Mean cell wl advection',np.mean(np.abs(cw_convect_term[:num_total_segs])))
    print('Max cell wl advection',np.max(cw_convect_term[:num_total_segs]))
    print('Max segment length',np.max(mycelia['seg_length']))
    ndensity = [len(i) for i in mycelia['share_e'][:num_total_segs]]
    print('Max segment density: ', np.max(ndensity))
    print('Mean segment density: ', np.mean(ndensity))
    if (np.abs(np.sum(cw_convect_term[:num_total_segs])) > 1.0e-24):
        print('Net Convection of CW greater than zero')
        #for idx in range(num_total_segs):
        #    diffcw =  -cw_convect_term[idx]
    # Sum/net advection should be zero:
    relative_net_advection = np.abs(np.sum(gluc_convect_term[:num_total_segs]))/np.mean(np.abs(gluc_convect_term[:num_total_segs]))
    if (relative_net_advection > 1.0e-15):
        print('Glucose advection too high: ', relative_net_advection)

    # Update concentrations due to convection
    if (np.any(mycelia['cw_i'][:num_total_segs] + params['dt_i']*cw_convect_term < 0)):
        bad_idx = np.where((mycelia['cw_i'][:num_total_segs] + params['dt_i']*cw_convect_term[:num_total_segs]) < 0)
        print('Bad indices: ',bad_idx)
        print('Cell Wall before conversion:',mycelia['cw_i'][bad_idx])
        print('Amount converted:',cw_convect_term[bad_idx]*params['dt_i'])
        print('Convert rate:',cw_convect_term[bad_idx])
        print('Cell Wall after conversion:',mycelia['cw_i'][bad_idx] + params['dt_i']*(cw_convect_term[bad_idx]))
        #breakpoint()
       # Update concentrations due to convection
    if (np.any(mycelia['gluc_i'][:num_total_segs] + params['dt_i']*gluc_convect_term < 0)):
        bad_idx = np.where((mycelia['gluc_i'][:num_total_segs] + params['dt_i']*gluc_convect_term[:num_total_segs]) < 0)
        print('Bad indices: ',bad_idx)
        print('Glucose before conversion:',mycelia['gluc_i'][bad_idx])
        print('Amount converted:',gluc_convect_term[bad_idx]*params['dt_i'])
        print('Convert rate:',gluc_convect_term[bad_idx])
        print('Glucose after conversion:',mycelia['gluc_i'][bad_idx] + params['dt_i']*(gluc_convect_term[bad_idx]))
        #breakpoint()

    mycelia['cw_i'][:num_total_segs] += params['dt_i']*(cw_convect_term)
    mycelia['gluc_i'][:num_total_segs] += params['dt_i']*(gluc_convect_term)
    
    negative_cw_i_idx = np.where(mycelia['cw_i'][:num_total_segs] < 0)[0]
    negative_gluc_i_idx = np.where(mycelia['gluc_i'][:num_total_segs] < 0)[0]
    if len(negative_cw_i_idx)>0:
        mycelia['cw_i'][negative_cw_i_idx] = np.finfo(np.float64).tiny;
    if len(negative_treha_i_idx)>0:
        mycelia['treha_i'][negative_treha_i_idx] = np.finfo(np.float64).tiny;
    if len(negative_gluc_i_idx)>0:
        mycelia['gluc_i'][negative_gluc_i_idx] = np.finfo(np.float64).tiny;

    # breakpoint()
    if(np.any(mycelia['gluc_i'][:num_total_segs] < 0)):
        print('Glucose count below zero:', np.where(mycelia['gluc_i'][:num_total_segs] < 0))
        breakpoint()

    #print('Advection - gluc_curr:',mycelia['gluc_i'][:num_total_segs])

    return mycelia
    

# ----------------------------------------------------------------------------
# UPTAKE FUNCTIONS
# ----------------------------------------------------------------------------
    
def uptake(sub_e_gluc, mycelia, num_total_segs, var_nutrient_backgrnd):
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
    gluc_uptake = params['dt_e']*gf.michaelis_menten(params['ku1_gluc'], 
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
        sub_e_gluc[xy_e_idx] = gluc_e - gluc_up_sum*var_nutrient_backgrnd
        
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


