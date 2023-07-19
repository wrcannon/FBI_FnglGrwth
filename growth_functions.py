#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:21:05 2020

@author: jolenebritton
"""

import numpy as np
import helper_functions as hf
import nutrient_functions as nf

params, config = hf.get_configs('parameters.ini')


# ----------------------------------------------------------------------------
# GENERAL FUNCTIONS
# ----------------------------------------------------------------------------

def michaelis_menten(k, K, s):
    """
    Parameters
    ----------
    k : double
        Maximum rate achieved by the system, at saturating substrate concentration.
    K : double
        Michaelis constant.
    s : double
        Concentration of substrate/nutrient.

    Returns
    -------
    double
        Reaction rate.

    """

    return (k * s) / (K + s)

# ----------------------------------------------------------------------------

def calc_dist(xy1, xy2):
    """
    Parameters
    ----------
    xy1 : double
        x- & y-coordinate of starting point.
    xy2 : double
        x- & y-coordinate of ending point.

    Returns
    -------
    double
        The distance between (xy1[0],xy1[1]) and (xy2[0],xy2[1]).

    """

    # breakpoint()
    return np.sqrt((xy2[1] - xy1[1])**2 + (xy2[0] - xy1[0])**2)

# ----------------------------------------------------------------------------

def check_negative(condition):
    """
    Parameters
    ----------
    condition : double
        A scalar value to check the sign of.
    """

    # if condition < 0:
    #     breakpoint()


# ----------------------------------------------------------------------------
# EXTENSION & BRANCHING FUNCTIONS
# ----------------------------------------------------------------------------

def map_to_grid(mycelia, idx, num_total_segs, x_vals, y_vals):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idx : int
        Segment index.
    num_total_segs : int
        Current total number of segments in the mycelium.
    x_vals : list/array
        The x-values of the external grid.
    y_vals : list/array
        The y-values of the external grid.

    Returns
    -------
    mycelia : TYdictionaryPE
        Updated structural information of mycelia colony for all hyphal segments.

    Purpose
    -------
    Map the starting endpoint of a hyphal segment to the nearest external grid
    coordinates. Note if multiples segments are mapped to the same external grid
    point.
    """
    # Include self in list of shared endpoints
    mycelia['share_e'][idx] = [idx]
    # Get the index of the external grid point
    xy_e = np.array([int(np.argmin(abs(mycelia['xy1'][idx,0]-x_vals))),int(np.argmin(abs(mycelia['xy1'][idx,1]-y_vals)))])
    # print(xy_e)
    if xy_e is None:
        print('None value')
    # If another segments also maps to the same location
    e_overlap = list(map(lambda x: np.array_equal(x, xy_e), mycelia['xy_e_idx'][:num_total_segs,:]))
    # print(e_overlap)
    mycelia['xy_e_idx'][idx,:] = xy_e
    if any(e_overlap):
        mycelia['share_e'][np.where(e_overlap)[0][0]].append(idx)
        mycelia['share_e'][idx].append(np.where(e_overlap)[0][0])

    return mycelia

# ----------------------------------------------------------------------------

def cost_of_growth(mycelia, idxs, grow_len):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idxs : int or array of ints
        Segment index or array of segment indicies.
    grow_len : double
        The amount in which a segment grows/extends.

    Returns
    -------
    grow_len : double
        The possibly updated amount in which a segment grows/extends.
    cost_grow_gluc : double
        The amount of glucose required to grow a length of grow_len.
    cost_grow_cw : double
        The amount of cell wall materials required to grow a length of grow_len.

    Purpose
    -------
    Calculate the amount of nutrients needed to grow. If the cost is higher
    than the amount available, adjust the amount of growth.
    """
    if any(grow_len > 1e2):
        breakpoint()
    # Cost to grow predetermined amount

    # Here we are testing to see if there is enough material in the
    # segment to extend the segment the length of length grow_len.
    #cost_grow_cq = grow_len* area * density* (dry mass/wet mass/)(cellwallmass/dry mass) (millimeters/cubic micron) * mw(cw_i)
    # gamma is mmoles cw_i/micron
    gamma = params['f_dw']* params['f_wall'] * params['f_cw_cellwall']/params['mw_cw']* \
           params['hy_density']*params['cross_area']
    cost_grow_cw1 = grow_len * gamma # cost in mmoles of cw_i
    #cost_grow_cw1 = grow_len * mycelia['cw_i'][idxs]
    #cost_grow_cw1 = grow_len* np.pi*(5.0/2)**2 * params['hy_density'] * params['f_dw']* \
    #    params['f_wall']  * params['f_cw_cellwall']/params['mw_cw']
    
    #cost_grow_cw1 = grow_len* np.pi*(5.0/2)**2 * params['hy_density'] * params['f_dw']* \
    #    params['f_wall']  * params['f_cw_cellwall']/params['mw_cw']

    # Next test to see if the new tip volume will deplete all the material
    # from the segment without regard to the amount of material required
    # to form the cell wall; That is, scale the length of the new tip so that the 
    # concentration of the preceding segment
    # doesn't become negative.
    #cost_grow_cw2 = (mycelia['cw_i'][idxs]/mycelia['seg_length'][idxs])  * (grow_len)
    scale = michaelis_menten(1, cost_grow_cw1, 
                                mycelia['cw_i'][idxs])
    cost_grow_cw2 = cost_grow_cw1 * scale    
    #cost_grow_cw = np.maximum(np.max(cost_grow_cw1), np.max(cost_grow_cw2))
    #cost_grow_cw = np.maximum((cost_grow_cw1), (cost_grow_cw2))
    cost_grow_cw = np.minimum((cost_grow_cw1), (cost_grow_cw2))
    #cost_grow_cw = cost_grow_cw2
    #Here we are testing to see if the new tip volume will deplete all the material
    #from the preceding segment
    # Need to make extend_len shorter if cost is too much
    cost_grow_gluc = (mycelia['gluc_i'][idxs]/mycelia['seg_length'][idxs]) * (grow_len)
    cost_grow_gluc[:] = 0.0

    
    if ((min(mycelia['cw_i'][idxs] - cost_grow_cw) < 0) and (max(grow_len > 0))):
        # new_grow_len = 0.5*grow_len * mycelia['cw_i'][idxs]/cost_grow_cw
        # if any(new_grow_len > 1e2):
        #     breakpoint()
        # cost_grow_cw = cost_grow_cw * new_grow_len/grow_len
        # grow_len = new_grow_len #if we change grow_len do we need to change dt also?
        
        not_enough_cw = np.where(mycelia['cw_i'][idxs] < cost_grow_cw)[0]
        try:
            grow_len[[not_enough_cw]] = grow_len[not_enough_cw]*mycelia['cw_i'][idxs[not_enough_cw]]/cost_grow_cw[not_enough_cw] #0.0
        except:
            breakpoint()
        # cost_grow_cw[not_enough_cw] = 0.0
        
         # The alternative is to set the grow_len to zero and wait until enough material
         # has been accumulated in the segment.
    rand_work = np.random.rand(*cost_grow_cw.shape)*cost_grow_cw
    diff_work = rand_work - cost_grow_cw
    # scale = np.exp(2*diff_work/cost_grow_cw)
    if any(grow_len > 1e2):
        breakpoint()
    # scale = 0.3    
    #scale = 1.0
    grow_len = grow_len*scale
    #cost_grow_cw = cost_grow_cw*scale
    #grow_len = grow_len *0.3
    #cost_grow_cw = cost_grow_cw *0.35

    #cost_grow_gluc = (mycelia['gluc_i'][idxs]/mycelia['seg_length'][idxs]) * (grow_len)
    #Here we are testing to see if the new tip volume will deplete all the material
    #from the preceding segment
    # Need to make extend_len shorter if cost is too much
    #if (min(mycelia['gluc_i'][idxs] - cost_grow_gluc) < 0) and (max(grow_len) > 0):
    #    # breakpoint()
    #    cost_grow_gluc = 0.9*np.min((cost_grow_gluc, mycelia['gluc_i'][idxs]), axis=0)
    #    grow_len = (mycelia['seg_length'][idxs]/mycelia['gluc_i'][idxs]) * cost_grow_gluc
    #    cost_grow_gluc = (mycelia['gluc_i'][idxs]/mycelia['seg_length'][idxs]) * (grow_len)
    #if(np.any(cost_grow_gluc > 1.0 )):
    #        breakpoint()

    #cost_grow_cw = (mycelia['cw_i'][idxs]/mycelia['seg_length'][idxs])  * (grow_len)
    #Here we are testing to see if the new tip volume will deplete all the material
    #from the preceding segment without regard to the amount of material required
    # to form the cell wall
    #if (min(mycelia['cw_i'][idxs] - cost_grow_cw) < 0) and (max(grow_len) > 0): 
    #    # breakpoint()
    #    cost_grow_cw = 0.9*np.min((cost_grow_cw, mycelia['cw_i'][idxs]), axis=0)
    #    grow_len = (mycelia['seg_length'][idxs]/mycelia['cw_i'][idxs]) * cost_grow_cw
    #    cost_grow_cw = (mycelia['cw_i'][idxs]/mycelia['seg_length'][idxs]) * (grow_len)

    if any(grow_len > 1e2):
        breakpoint()
    
    return grow_len, cost_grow_gluc, cost_grow_cw

# ----------------------------------------------------------------------------

def update_structure(mycelia, idxs, grow_len, cost_grow_gluc, cost_grow_cw, isCalibration):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idxs : int or array of ints
        Segment index or array of segment indicies.
    grow_len : double
        The amount in which a segment grows/extends.
    cost_grow_gluc : double
        The amount of glucose required to grow a length of grow_len.
    cost_grow_cw : double
        The amount of cell wall materials required to grow a length of grow_len.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    Purpose
    -------
    Determine updated endpoints of segments, updated length of segment,
    updated nutrients in the segment, and distance from nearest septa.
    """

    # New endpoint
    if (isCalibration == 0):
        mycelia['xy2'][idxs,0] += (grow_len*np.cos(mycelia['angle'][idxs])).flatten()
        mycelia['xy2'][idxs,1] += (grow_len*np.sin(mycelia['angle'][idxs])).flatten()
    else:
        mycelia['xy2'][idxs,0] += (grow_len*np.cos(0)).flatten()
        mycelia['xy2'][idxs,1] += (grow_len*np.sin(0)).flatten()
    

    # Update length and distance to nearest septa
    mycelia['seg_length'][idxs] += grow_len
    mycelia['seg_vol'][idxs] = mycelia['seg_length'][idxs] * params['cross_area']

    # Update concentration of nutrients due cost of growth
    mycelia['gluc_i'][idxs] -= cost_grow_gluc
    #if(np.shape(mycelia['cw_i'][idxs]) != np.shape(cost_grow_cw)):
    #    breakpoint()
    mycelia['cw_i'][idxs] -= cost_grow_cw
        

    # Update distance between tip and nearest septa
    mycelia['dist_to_septa'][idxs] += grow_len

    return mycelia

# ----------------------------------------------------------------------------

# def septa_formation(mycelia, num_total_segs):
#     """
#     Parameters
#     ----------
#     mycelia : dictionary
#         Stores structural information of mycelia colony for all hyphal segments.
#     num_total_segs : int
#         CCurrent total number of segments in the mycelium.

#     Returns
#     -------
#     mycelia : dictionary
#         Updated structural information of mycelia colony for all hyphal segments.

#     Purpose
#     -------
#     Denotes where a new septa forms.
#     """

#     print('septa forming')
#     # breakpoint()

#     # Indicices of segments whose dist_to_septa is long enough to introduce new septa
#     septa_need = np.where(mycelia['dist_to_septa']> 3*params['sl']*params['septa_len'])[0]
#     if any(mycelia['dist_to_septa'][septa_need] <= 3*params['sl']*params['septa_len']):
#         breakpoint()
#     #septa_need = np.where(mycelia['dist_to_septa']> params['sl']*params['septa_len'])[0]
#     # breakpoint()
#     # Find the branches that need new septa
#     branch_ids = mycelia['branch_id'][septa_need]
#     print('mycelia[dist_to_septa][septa_need] : ' , mycelia['dist_to_septa'][septa_need])

#     for idx, branch in enumerate(branch_ids):
#         print('branch : ', branch)
#         print('idx : ', idx)
#         # Find IDs of segments on that branch
#         segs_on_branch = np.where(mycelia['branch_id'][:num_total_segs] == branch)[0]
        
#         # If the branch already has a septa
#         if any(mycelia['septa_loc'][segs_on_branch]):
#             # print('IF')
#             # breakpoint()
#             # Find which segment the last septa was on
#             last_septa = max(np.where(mycelia['septa_loc'][segs_on_branch])[0])

#             # New segment septa will have seg index N larger than last septa seg index
#             new_septa = last_septa + params['septa_len']
            
#             if new_septa >= len(segs_on_branch):
#                 breakpoint()
#             print('branch = ', branch,'length(segs_on_branch) = ', len(segs_on_branch), ', new_septa = ',new_septa)

#             # Indicate location of septa
#             mycelia['septa_loc'][segs_on_branch[new_septa]] = True

#             # Allow the segment behind it to branch
#             mycelia['can_branch'][segs_on_branch[new_septa-1]] = True
            
#             # print('mycelia[dist_to_septa][septa_need] : ' , mycelia['dist_to_septa'][septa_need])
            
#             mycelia['dist_to_septa'][septa_need[idx]] -= params['sl']*params['septa_len']
#             if any(mycelia['dist_to_septa'][septa_need] < 0):
#                 breakpoint()
            

#         else:
#             # print('ELSE')
#             if len(segs_on_branch) < 2:
#                 breakpoint()
#                 print('There is no other segments in this single-segment-branch branch. There is no need to set branching segments.')
#                 #mycelia['septa_loc'][segs_on_branch[0]] = True
#             else:
#                 print('segs_on_branch[params[septa_len]]:', segs_on_branch[params['septa_len']])
#                 # Find the index of the segment that is N segments away from the start of the branch
#                 # if any(mycelia['dist_to_septa'][septa_need] - params['sl']*params['septa_len'] < 2*params['sl']*params['septa_len']):
#                 #     breakpoint()
#                 mycelia['septa_loc'][segs_on_branch[params['septa_len']]] = True
    
#                 # Allow the segment behind it to branch
#                 mycelia['can_branch'][segs_on_branch[params['septa_len']-1]] = True
#                 # if any(mycelia['dist_to_septa'][septa_need] - params['sl']*params['septa_len'] < 2*params['sl']*params['septa_len']):
#                 #     breakpoint()
#                 # print('mycelia[dist_to_septa][septa_need] : ' , mycelia['dist_to_septa'][septa_need])
#                 mycelia['dist_to_septa'][septa_need[idx]] -= params['sl']*params['septa_len']
            

#     # Update distance between tip and new septa
#     #mycelia['dist_to_septa'][septa_need] -= params['sl']*params['septa_len']
#     # if min(mycelia['dist_to_septa'][septa_need]) < 0:
#     #     breakpoint()

#     return mycelia


def septa_formation(mycelia, num_total_segs):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        CCurrent total number of segments in the mycelium.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    Purpose
    -------
    Denotes where a new septa forms.
    """

    # print('septa forming')
    # breakpoint()
    
    # Indicices of segments whose dist_to_septa is long enough to introduce new septa
    nn = 3
    septa_need = np.where(mycelia['dist_to_septa'][:num_total_segs] > nn*params['sl']*params['septa_len'])[0]
    if any(mycelia['dist_to_septa'][septa_need] <= nn*params['sl']*params['septa_len']):
        breakpoint()
    #septa_need = np.where(mycelia['dist_to_septa']> params['sl']*params['septa_len'])[0]
    # breakpoint()
    # Find the branches that need new septa
    branch_ids = mycelia['branch_id'][septa_need]
    # print('mycelia[dist_to_septa][septa_need] : ' , mycelia['dist_to_septa'][septa_need])

    for idx, branch in enumerate(branch_ids):
        if (mycelia['is_tip'][septa_need[idx]] == False):
            continue
        # print('branch : ', branch)
        # print('idx : ', idx)
        # Find IDs of segments on that branch
        segs_on_branch = np.where(mycelia['branch_id'][:num_total_segs] == branch)[0]
        
        # If the branch already has a septa
        if any(mycelia['septa_loc'][segs_on_branch]):
            # print('IF')
            # breakpoint()
            # Find which segment the last septa was on
            last_septa = max(np.where(mycelia['septa_loc'][segs_on_branch])[0])

            # New segment septa will have seg index N larger than last septa seg index
            # N = septa_len = 1 currently
            new_septa = last_septa + params['septa_len']
            
            if new_septa >= len(segs_on_branch):
                breakpoint()
            # print('branch = ', branch,'length(segs_on_branch) = ', len(segs_on_branch), ', new_septa = ',new_septa)

            # Indicate location of septa
            mycelia['septa_loc'][segs_on_branch[new_septa]] = True

            # Allow the segment behind it to branch
            mycelia['can_branch'][segs_on_branch[new_septa-1]] = True
            
            # print('mycelia[dist_to_septa][septa_need] : ' , mycelia['dist_to_septa'][septa_need])
            
            mycelia['dist_to_septa'][septa_need[idx]] -= params['sl']*params['septa_len']
            if (mycelia['dist_to_septa'][septa_need[idx]] < 0):
                breakpoint()
            

        else:
            # print('ELSE')
            if len(segs_on_branch) < 2:
                #breakpoint()
                print('There is no other segments in this single-segment-branch branch. There is no need to set branching segments.')
                #mycelia['septa_loc'][segs_on_branch[0]] = True
            else:
                # print('segs_on_branch[params[septa_len]]:', segs_on_branch[params['septa_len']])
                # Find the index of the segment that is N segments away from the start of the branch
                # if any(mycelia['dist_to_septa'][septa_need] - params['sl']*params['septa_len'] < 2*params['sl']*params['septa_len']):
                #     breakpoint()
                mycelia['septa_loc'][segs_on_branch[params['septa_len']]] = True
    
                # Allow the segment behind it to branch
                mycelia['can_branch'][segs_on_branch[params['septa_len']-1]] = True
                # if any(mycelia['dist_to_septa'][septa_need] - params['sl']*params['septa_len'] < 2*params['sl']*params['septa_len']):
                #     breakpoint()
                # print('mycelia[dist_to_septa][septa_need] : ' , mycelia['dist_to_septa'][septa_need])
                mycelia['dist_to_septa'][septa_need[idx]] -= params['sl']*params['septa_len']
                # if mycelia['dist_to_septa'][septa_need[idx]] > 60:
                #     breakpoint()
            

    # Update distance between tip and new septa
    #mycelia['dist_to_septa'][septa_need] -= params['sl']*params['septa_len']
    # if min(mycelia['dist_to_septa'][septa_need]) < 0:
    #     breakpoint()

    return mycelia

# ----------------------------------------------------------------------------

def split_segment(mycelia, num_total_segs, x_vals, y_vals, isCalibration, dist2Tip_new):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    x_vals : list/array
        The x-values of the external grid.
    y_vals : list/array
        The y-values of the external grid.

    Returns
    -------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.

    Purpose
    -------
    Splits a segment into twoIf a segment is long enough.
    The structural information for each segment is updated.
    """

    # Locations of spliting tips
    nn = 3
    tips_to_split = np.where(mycelia['seg_length'][:num_total_segs] > (nn-1)*params['sl'])[0]
    if any(mycelia['seg_length'][tips_to_split] < (nn-1)*params['sl']):
        breakpoint()
    new_tips = np.arange(len(tips_to_split)) + num_total_segs
    # print('segment(s) spliting, new tips', new_tips)
    # breakpoint()
    # branch_ids = mycelia['branch_id'][tips_to_split]
    # print('branch(s) undergoing segment splitting : ', branch_ids)

    # Record branch and segment id
    mycelia['branch_id'][new_tips] = mycelia['branch_id'][tips_to_split]
    mycelia['seg_id'][new_tips] = mycelia['seg_id'][tips_to_split]+1

    # Update neighbor list
    for idx, tip in enumerate(tips_to_split):
        mycelia['nbr_idxs'][tip].append(new_tips[idx])
        mycelia['nbr_idxs'][new_tips[idx]] = [tip]
    mycelia['nbr_num'][tips_to_split] += 1
    mycelia['nbr_num'][new_tips] += 1

    # Increase number of segments on branches that split
    num_total_segs += len(tips_to_split)

    # Store current lengths of splitting tips
    current_lens = mycelia['seg_length'][tips_to_split]

    # Redefine endpoint for segment behind the new tip
    if (isCalibration == 0):
        mycelia['xy2'][tips_to_split,0] = mycelia['xy1'][tips_to_split,0] + (params['sl']*np.cos(mycelia['angle'][tips_to_split])).flatten()
        mycelia['xy2'][tips_to_split,1] = mycelia['xy1'][tips_to_split,1] + (params['sl']*np.sin(mycelia['angle'][tips_to_split])).flatten()
    else:
        mycelia['xy2'][tips_to_split,0] = mycelia['xy1'][tips_to_split,0] + (params['sl']*np.cos(0)).flatten()
        mycelia['xy2'][tips_to_split,1] = mycelia['xy1'][tips_to_split,1] + (params['sl']*np.sin(0)).flatten()

    # Redefine length of segment behind new tip
    mycelia['seg_length'][tips_to_split] = params['sl']

    # Starting endpoint for new tip
    mycelia['xy1'][new_tips,:] = mycelia['xy2'][tips_to_split,:]

    # Ending endpoint for new tip with modified angle
    if (isCalibration == 0):
        mycelia['angle'][new_tips] = mycelia['angle'][tips_to_split] + np.random.normal(0, params['angle_sd'], np.shape(mycelia['angle'][tips_to_split]))
        mycelia['xy2'][new_tips,0] = mycelia['xy1'][new_tips,0] + ((current_lens - params['sl'])*np.cos(mycelia['angle'][new_tips])).flatten()
        mycelia['xy2'][new_tips,1] = mycelia['xy1'][new_tips,1] + ((current_lens - params['sl'])*np.sin(mycelia['angle'][new_tips])).flatten()
    else:
        mycelia['angle'][new_tips] = mycelia['angle'][tips_to_split] + 0#np.random.normal(0, 0, np.shape(mycelia['angle'][tips_to_split]))
        mycelia['xy2'][new_tips,0] = mycelia['xy1'][new_tips,0] + ((current_lens - params['sl'])*np.cos(mycelia['angle'][new_tips])).flatten()
        mycelia['xy2'][new_tips,1] = mycelia['xy1'][new_tips,1] + ((current_lens - params['sl'])*np.sin(mycelia['angle'][new_tips])).flatten()

    # Set length of new tip
    mycelia['seg_length'][new_tips] = current_lens - params['sl']

    # Designate new tip as a tip
    mycelia['is_tip'][tips_to_split] = False
    mycelia['is_tip'][new_tips] = True

    # Update distance to septa information
    mycelia['dist_to_septa'][new_tips] = mycelia['dist_to_septa'][tips_to_split]
    mycelia['dist_to_septa'][tips_to_split] = 0

    # Map to external grid
    for tip_idx in new_tips:
        mycelia = map_to_grid(mycelia, tip_idx, num_total_segs, x_vals, y_vals)

    # Update nutrient distribution
    # Percent of new growth compared to total
    perct_new_size = (current_lens - params['sl'])/current_lens

    # Store concentrations from originating segments
    gluc_new = mycelia['gluc_i'][tips_to_split]
    cw_new = mycelia['cw_i'][tips_to_split]

    # Update new & originating segments
    mycelia['gluc_i'][new_tips] = perct_new_size*gluc_new
    mycelia['cw_i'][new_tips] = perct_new_size*cw_new
    mycelia['gluc_i'][tips_to_split] = (1-perct_new_size)*gluc_new
    mycelia['cw_i'][tips_to_split] = (1-perct_new_size)*cw_new

    # Update distance to tip
    if dist2Tip_new == 1:
        dtt = nf.distance_to_tip_new(mycelia, num_total_segs)
    else:
        dtt = nf.distance_to_tip(mycelia, num_total_segs)
    # breakpoint()

    return mycelia, num_total_segs, dtt

# ----------------------------------------------------------------------------

# Extension - Main function for elongation
def extension(mycelia, num_total_segs, dtt, x_vals, y_vals, 
              isCalibration, dist2Tip_new, fungal_fusion,
              chance_to_fuse):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.
    x_vals : list/array
        The x-values of the external grid.
    y_vals : list/array
        The y-values of the external grid.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.

    Purpose
    -------
    Driver function for elongation - determines which tips elongate, updates
    the structural information and checks for septation and segment splitting.
    """

    tip_idxs = (np.where(mycelia['is_tip'])[0]).tolist()
    # Get the number density of segments around each segment 
    ndensity = [len(i) for i in mycelia['share_e'][:num_total_segs]]
    # Find those segments that have greater than half of their grid filled by other segments
    # But apply density filtering only after initial tips have segmented
    if 3 not in tip_idxs:
        #fidx = np.where(ndensity < np.int_(3))
        fidx = np.where(ndensity <= np.int64(params['dy']/params['hy_diam']*0.2))
        # Don't allow segments in high density regions to grow
        tip_idxs = list(set(tip_idxs) & set(fidx[0]))

    # Check to see if the list is empty, and if so, return
    if not tip_idxs:
        return mycelia, num_total_segs, dtt

    use_original = 0
    if(use_original == 1):
        extend_len = params['dt_i']*michaelis_menten(params['kg1_wall'],
                                                   params['Kg2_wall'],
                                                   params['yield_g']*params['mw_cw']*mycelia['cw_i'][tip_idxs])
    else:   
        #K = params['Kg2_wall'] * mycelia['hyphal_vol']
        dLdt = michaelis_menten(params['kg1_wall'],
                        params['Kg2_wall'],
                        mycelia['cw_i'][tip_idxs])
        #extend_len = params['dt']*dLdt
        extend_len = params['dt_i']*params['kg1_wall']*np.ones(np.shape([tip_idxs])).T
    
    if any(extend_len > 1e2):
        breakpoint()
    # Cost to grow predetermined amount or shorten if cost is too much
    extend_len, cost_grow_gluc, cost_grow_cw = cost_of_growth(mycelia, np.array(tip_idxs), extend_len)
    if(np.shape(mycelia['cw_i'][tip_idxs]) != np.shape(cost_grow_cw)):
        breakpoint()
    if any(extend_len > 1e2):
        breakpoint()
    # If nutrient to extend, update tip
    if max(extend_len) > 0:

        # New endpoint update and concentration update
        # print(tip_idxs, extend_len, cost_grow_gluc, cost_grow_cw)
        mycelia = update_structure(mycelia, tip_idxs, extend_len, cost_grow_gluc, cost_grow_cw, isCalibration)
        # if(np.any(np.isnan(mycelia['cw_i']))):
        #     xx = np.where(np.isnan(mycelia['cw_i']))
        #     mycelia['cw_i'][xx] = 0.0
            
##############################################################################
##############################################################################
        # Check for fusion
        if (fungal_fusion == 1):
            for idx in tip_idxs:
                mycelia = anastomosis(mycelia, idx, num_total_segs, chance_to_fuse)
            if(np.any(np.isnan(mycelia['cw_i']))):
                breakpoint()
##############################################################################
##############################################################################

        # Check if septa forms (i.e. when tip compartment is 3x length of a compartment)
        nn = 3
        #if max(mycelia['dist_to_septa']) > 3*params['sl']*params['septa_len']:
        if max(mycelia['dist_to_septa']) > nn*params['sl']*params['septa_len']:
            mycelia = septa_formation(mycelia, num_total_segs)
            # print('Septa formation from 3x length')
        # if(np.any(np.isnan(mycelia['cw_i']))):
        #     breakpoint()
        # Check if any tip segment splits
        if max(mycelia['seg_length'][tip_idxs] > (nn-1)*params['sl']):
            mycelia, num_total_segs, dtt = split_segment(mycelia, num_total_segs, x_vals, y_vals, isCalibration, dist2Tip_new)
            # print('Septa formation from 2x length')
        # if(np.any(np.isnan(mycelia['cw_i']))):
        #     breakpoint()
    return mycelia, num_total_segs, dtt

# ----------------------------------------------------------------------------

# Branching - Main function for new branches
def branching(mycelia, num_total_segs, dtt, x_vals, y_vals, 
              isCalibration, dist2Tip_new, fungal_fusion, restrictBranching,
              chance_to_fuse):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.
    x_vals : list/array
        The x-values of the external grid.
    y_vals : list/array
        The y-values of the external grid.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.
    num_total_segs : int
        Current total number of segments in the mycelium.
    dtt : array
        Contains the distance each segment is away from the nearest tip segment.

    Purpose
    -------
    Driver function for branching - determines which segments branch, updates
    the structural information.
    """
    reached_max_branches = False
    # Get the number density of segments around each segment 
    ndensity = [len(i) for i in mycelia['share_e'][:num_total_segs]]
    # Find those segments that have greater than half of their grid filled by other segments
    fidx = np.where(ndensity > np.int64(params['sl']/params['hy_diam']*0.2))
    #fidx = np.where(ndensity[4:] > np.int_(2))
    # Don't allow segments in high density regions to branch
    mycelia['can_branch'][fidx] = False
    use_original = 0
    if (restrictBranching == 0):
        potential_branch_idxs = (np.where(mycelia['can_branch'])[0])
    else:
        tmp_potential_branch_idxs = np.where(mycelia['can_branch'])[0]
        restricting = np.where(dtt[tmp_potential_branch_idxs]<=restrictBranching)[0]
        potential_branch_idxs = (tmp_potential_branch_idxs[restricting])
        # if np.max(num_total_segs > 8):
        #     breakpoint()

    if not np.any(potential_branch_idxs):
        return reached_max_branches, mycelia, num_total_segs, dtt

    if(use_original == 1):
        rand_vals = np.random.uniform(0, 1, (len(potential_branch_idxs),1))
        # print('rand_vals : ', rand_vals)
        # print('prob : ', params['branch_rate']*mycelia['cw_i'][potential_branch_idxs])
        true_branch_ids = potential_branch_idxs[np.where((rand_vals - params['branch_rate']*mycelia['cw_i'][potential_branch_idxs]) < 0)[0]]
        
    else:
        advection_vel_cw = params['advection_vel_cw']
        K_cw = advection_vel_cw*params['dt_i']
        alpha_cw = michaelis_menten(1, K_cw, 
                                mycelia['cw_i'][:num_total_segs])
        nutrient_scaling = 1.0 #rich media
        nutrient_scaling = 0.1 #minimal media
        prob = alpha_cw * nutrient_scaling
        #true_idx = np.where((prob - rand_vals).flatten() > 0)
        
        #print('potential_branch_idxs = ', potential_branch_idxs)
        branch_len = params['dt_i']*michaelis_menten(params['kg1_wall'],
                                                       params['Kg2_wall'],
                                                       mycelia['cw_i'][potential_branch_idxs])
        len0 = params['kg1_wall']*params['dt_i']
        branch_len0 = np.ones((len(potential_branch_idxs),1))*len0
        branch_len, cost_branch_gluc, cost_branch_cw = cost_of_growth(mycelia, np.array(potential_branch_idxs), branch_len0)
        cost_multiple = 4.0#0.5
        # prob = 1.0 - np.exp(-(mycelia['cw_i'][potential_branch_idxs] - cost_multiple*cost_branch_cw)/(cost_multiple*cost_branch_cw))
        #prob = np.exp((branch_len-1*branch_len0)/branch_len0)

        
        #prob = michaelis_menten(1,params['Kg2_wall'],mycelia['cw_i'][potential_branch_idxs])
        rand_vals = np.random.uniform(0, 1, (len(potential_branch_idxs),1))
        #rand_vals = np.ones((len(potential_branch_idxs),1))* 0.5
        #true_idx = np.where((prob - rand_vals).flatten() > 0)
        true_idx = np.where(mycelia['dist_from_center'][potential_branch_idxs] >= 0.5*max(mycelia['dist_from_center'][:num_total_segs][0]))[0]

        # true_idx = np.where((prob - rand_vals).flatten() > 0)
        true_branch_ids = potential_branch_idxs[true_idx]
        branch_len =  branch_len[true_idx]
        cost_branch_cw =  cost_branch_cw[true_idx]
        cost_branch_gluc =  cost_branch_gluc[true_idx]
    

        
    if any(true_branch_ids):
        if(use_original ==1):
            branch_len = params['dt_i']*michaelis_menten(params['kg1_wall'],
                                                   params['Kg2_wall'],
                                                   params['yield_g']*params['mw_cw']*mycelia['cw_i'][true_branch_ids])

            branch_len, cost_branch_gluc, cost_branch_cw = cost_of_growth(mycelia, np.array(true_branch_ids), branch_len)
        
        # If nutrient to extend, update tip
        if max(branch_len) > 0:

            # Locations of new branch tips
            new_tips = np.arange(len(true_branch_ids)) + num_total_segs
            if np.max(new_tips) > np.shape(mycelia['cw_i'])[0]:
                reached_max_branches = True;
                return reached_max_branches, mycelia, num_total_segs, dtt
            
            # print('new branch(es):', new_tips)
            if(np.shape(mycelia['cw_i'][new_tips]) != np.shape(cost_branch_cw)):
                breakpoint()

            # Record branch and segment id
            num_branches = max(mycelia['branch_id'])+1
            mycelia['branch_id'][new_tips] = (np.arange(len(true_branch_ids)) + num_branches).reshape(-1,1)
            mycelia['seg_id'][new_tips] = np.zeros((len(true_branch_ids),1))

            # Update neighbor list
            for idx, tip in enumerate(true_branch_ids):
                mycelia['nbr_idxs'][tip].append(new_tips[idx])
                mycelia['nbr_idxs'][new_tips[idx]] = [tip]
            mycelia['nbr_num'][true_branch_ids] += 1
            mycelia['nbr_num'][new_tips] += 1

            # Update total number of segments
            num_total_segs += len(true_branch_ids)

            # Determine position of the new branches
            mycelia['xy1'][new_tips,:] = 0.5*(mycelia['xy1'][true_branch_ids,:] + mycelia['xy2'][true_branch_ids,:])

            # This will be adjusted in the update_structure function below
            mycelia['xy2'][new_tips,:] = mycelia['xy1'][new_tips,:]

            # New endpoint
            angle_sign = np.sign(np.random.uniform(-1,1,np.shape(mycelia['angle'][true_branch_ids])))
            mycelia['angle'][new_tips] = mycelia['angle'][true_branch_ids] + angle_sign*np.random.normal(params['branch_mean'], params['branch_sd'], np.shape(mycelia['angle'][true_branch_ids]))
            zero_cost = np.zeros(np.shape(cost_branch_cw))
            mycelia = update_structure(mycelia, new_tips, branch_len, zero_cost, zero_cost, isCalibration)

            # breakpoint()
            # Designate new tip as a tip and set originating banch to can't branch
            mycelia['can_branch'][true_branch_ids] = False
            mycelia['is_tip'][new_tips] = True

            # Map to external grid
            for tip_idx in new_tips:
                mycelia = map_to_grid(mycelia, tip_idx, num_total_segs, x_vals, y_vals)

            # Update nutrient distribution
            # Percent of new growth compared to total
            perct_new_size = branch_len/(branch_len + mycelia['seg_length'][true_branch_ids])

            # Store concentrations from originating segments
            gluc_new = mycelia['gluc_i'][true_branch_ids]
            cw_new = mycelia['cw_i'][true_branch_ids]

            # Update new & originating segments
            mycelia['gluc_i'][new_tips] = perct_new_size*gluc_new
            mycelia['cw_i'][new_tips] = perct_new_size*cw_new
            mycelia['gluc_i'][true_branch_ids] = (1-perct_new_size)*gluc_new
            mycelia['cw_i'][true_branch_ids] = (1-perct_new_size)*cw_new
            
            if fungal_fusion == 1:
                for idx in new_tips:
                    mycelia = anastomosis(mycelia, idx, num_total_segs, chance_to_fuse)

            # Update distance to tip
            if dist2Tip_new == 1:
                dtt = nf.distance_to_tip_new(mycelia, num_total_segs)
            else:
                dtt = nf.distance_to_tip(mycelia, num_total_segs)
                    
            # breakpoint()
            if(np.any( mycelia['gluc_i'][:num_total_segs]< 0)):
                breakpoint()

    return reached_max_branches, mycelia, num_total_segs, dtt


# ----------------------------------------------------------------------------
# ANASTOMOSIS (FUSION) FUNCTIONS
# ----------------------------------------------------------------------------

# def anastomosis(mycelia, idx, num_total_segs):
#     """
#     Parameters
#     ----------
#     mycelia : dictionary
#         Stores structural information of mycelia colony for all hyphal segments.
#     idx : int
#         Segment index.
#     num_total_segs : int
#         Current total number of segments in the mycelium.

#     Returns
#     -------
#     mycelia : dictionary
#         Updated structural information of mycelia colony for all hyphal segments.

#     Purpose
#     -------
#     Checks if segment idx fuses with another segment and updates accordingly
#     """

#     # Endpoints of current hyphal segment
#     xy1_c = mycelia['xy1'][idx,:]
#     xy2_c = mycelia['xy2'][idx,:]

#     # Branch id for the current segment
#     branch_idx = mycelia['branch_id'][idx]

#     # Segments on the same branch
#     segs_on_branch = np.where(mycelia['branch_id'][:num_total_segs] == branch_idx)[0]

#     # Neighbors of current segment
#     seg_nbrs = mycelia['nbr_idxs'][idx]

#     # Combined list of neighbors & segments on branch
#     skip_ids = np.unique(np.concatenate((segs_on_branch, seg_nbrs)))

#     # Define the zone to look for interactions in
#     minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['sl'])

#     # Initialize fusion flags
#     seg_fuse = -1

#     # Loop through segments to check for intersections
#     for other_idx in range(num_total_segs):
#         if not mycelia['nbr_idxs'][other_idx]:
#             continue
#         if (mycelia['branch_id'][other_idx])[0] == -1:
#             continue
#         # Don't need to check other segments on the same branch or neighboring segments
#         if other_idx not in skip_ids:

#             # Name the endpoints of other segments
#             xy1_o = mycelia['xy1'][other_idx,:]
#             xy2_o = mycelia['xy2'][other_idx,:]

#             # Check if any enpoint inside box
#             if check_if_in_box(mycelia, xy1_o, xy2_o, minx, maxx, miny, maxy):

#                 # Check if intersection point in the segments
#                 found_intxn, x_intxn, y_intxn = get_seg_intxn(xy1_c,xy2_c,
#                                                               xy1_o,xy2_o)

#                 # breakpoint()

#                 # Update info if intersection is found
#                 if found_intxn:

#                     print('fusion occuring: {} (branch {} seg {}) fusing to {} (branch {} seg {})'.format(
#                         idx, mycelia['branch_id'][idx][0], mycelia['seg_id'][idx][0],
#                         other_idx, mycelia['branch_id'][other_idx][0], mycelia['seg_id'][other_idx][0]))
#                     print(' seg length before fusion: ', mycelia['seg_length'][idx])

#                     # breakpoint()
#                     # Update the endoints
#                     mycelia['xy2'][idx,0] = x_intxn
#                     mycelia['xy2'][idx,1] = y_intxn

#                     # Reset the length of the segment
#                     new_seg_len = calc_dist(mycelia['xy1'][idx,:], mycelia['xy2'][idx,:])


#                     # If the segment is long enough (satisfies CFL), proceed normally
#                     if new_seg_len >= params['dt']*params['vel_wall']:
#                         mycelia['seg_length'][idx] = new_seg_len
#                         short_multi_seg_fuse = False
#                         mycelia['is_tip'][idx] = False
#                         mycelia['can_branch'][idx] = False
                        
#                         if (mycelia['branch_id'][idx] == -1) and (mycelia['seg_id'][idx] > -1):
#                             breakpoint()
                    
#                     # If the segment is too short -> stability issues (fails CFL)
#                     else:
#                         mycelia['seg_length'][idx] = new_seg_len
#                         # If the segment is not the first on the branch, merge it with the previous seg on that branch
#                         if mycelia['seg_id'][idx] > 0:
#                             print('Fusion formed from a multi-segment branch')
#                             branch_idx = mycelia['branch_id'][idx]
#                             seg_idx = mycelia['seg_id'][idx]
#                             #breakpoint()
#                             # Find ID of previous seg on that branch
#                             segs_on_branch = np.where(mycelia['branch_id']==branch_idx)[0]
#                             where_is_idx = np.where(segs_on_branch == idx)[0]
#                             prev_idx = segs_on_branch[where_is_idx-1][0]
#                             # Redefine the endpoint of that segment to be the intersection point
#                             mycelia['xy2'][prev_idx,0] = x_intxn
#                             mycelia['xy2'][prev_idx,1] = y_intxn
#                             # Redefine the nbrs for the previous segment
#                             # Find where idx is listed as a neighbor of prev_idx, 
#                             #   and we will delete that information.
#                             find_idx_on_nbr_idxs = (np.where(mycelia['nbr_idxs'][prev_idx]==idx)[0])[0]
#                             A = mycelia['nbr_idxs'][prev_idx].pop(find_idx_on_nbr_idxs)
#                             # Redistribute the existing cw_i and gluc_i to the previous segment (?)
#                             # STILL NEED CONFIRMATION IF THIS IS OKAY.....
#                             mycelia['cw_i'][prev_idx] += (mycelia['cw_i'][idx])#/len(prev_idx))
#                             mycelia['gluc_i'][prev_idx] += (mycelia['gluc_i'][idx])#/len(prev_idx))
                            
#                             # Null out info for the current segment
#                             mycelia['nbr_idxs'][idx].clear()
#                             mycelia['branch_id'][idx] = -1
#                             mycelia['seg_id'][idx] = -1
#                             mycelia['xy1'][idx,:] = 0
#                             mycelia['xy2'][idx,:] = 0
#                             mycelia['angle'][idx] = 0
#                             mycelia['dist_to_septa'][idx] = 0
#                             mycelia['xy_e_idx'][idx,:] = 0
#                             mycelia['share_e'][idx] = None
#                             mycelia['cw_i'][idx] = 0
#                             mycelia['gluc_i'][idx] = 0
#                             mycelia['can_branch'][idx] = False
#                             mycelia['is_tip'][idx] = False
#                             mycelia['septa_loc'][idx] = 0
#                             mycelia['nbr_num'][idx] = 0
                            
#                             if (mycelia['branch_id'][idx] == -1) and (mycelia['seg_id'][idx] > -1):
#                                 breakpoint()
                            
#                             idx = prev_idx
#                             mycelia['nbr_num'][idx] -= 1
#                             new_seg_len = calc_dist(mycelia['xy1'][idx,:], mycelia['xy2'][idx,:])
#                             if new_seg_len < params['dt']*params['vel_wall']:
#                                 breakpoint()
#                             mycelia['seg_length'][idx] = new_seg_len
#                             print('new seg length after intra-multi-seg branch fusion : ', mycelia['seg_length'][idx])
#                             short_multi_seg_fuse = True
                            
#                         # The segment is first on the branch, just remove it???
#                         else:
#                             print('Fusion formed from a single segment branch')
#                             # breakpoint()
#                             nbr_idx = mycelia['nbr_idxs'][idx]

#                             # Add the cw & gluc in the seg to the neighboring segments
#                             mycelia['cw_i'][nbr_idx] += (mycelia['cw_i'][idx]/len(nbr_idx))
#                             mycelia['gluc_i'][nbr_idx] += (mycelia['gluc_i'][idx]/len(nbr_idx))

#                             # Remove this value as a neighbor
#                             # for nbr in nbr_idx:
#                             #     mycelia['nbr_idxs'][nbr].remove(idx)

#                             # Redefine all dictionary values
#                             #mycelia['branch_id'][idx] = -1
#                             #mycelia['seg_id'][idx] = -1
#                             #mycelia['xy1'][idx,:] = 0
#                             #mycelia['xy2'][idx,:] = 0
#                             #mycelia['angle'][idx] = 0
#                             mycelia['dist_to_septa'][idx] = 0
#                             #mycelia['xy_e_idx'][idx,:] = 0
#                             #mycelia['share_e'][idx] = None
#                             mycelia['cw_i'][idx] = 0
#                             mycelia['gluc_i'][idx] = 0
#                             mycelia['can_branch'][idx] = False
#                             mycelia['is_tip'][idx] = False
#                             mycelia['septa_loc'][idx] = 0
#                             #mycelia['nbr_idxs'][idx] = None
#                             #mycelia['nbr_num'][idx] = 0
#                             short_multi_seg_fuse = False
#                             # # Reset num of total segs
#                             # num_total_segs -= 1
#                             #breakpoint()
#                             print('new seg length after single-branch fusion : ', mycelia['seg_length'][idx])
#                             # if new_seg_len < params['dt']*params['vel_wall']:
#                                 # breakpoint()
                            
#                             if (mycelia['branch_id'][idx] == -1) and (mycelia['seg_id'][idx] > -1):
#                                 breakpoint()


#                     # Save neighbor information
#                     mycelia = set_anastomosis_nbrs(mycelia, idx, other_idx, seg_fuse, short_multi_seg_fuse)
                    
#                     # Rename the endpoints of the current hyphae segment
#                     xy2_c = mycelia['xy2'][idx,:]

#                     # Define zone to look for interections in
#                     minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['sl'])

#                     # Update fusion neighbor marker
#                     seg_fuse = other_idx

#     return mycelia

# ----------------------------------------------------------------------------

def anastomosis(mycelia, idx, num_total_segs, chance_to_fuse):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idx : int
        Segment index.
    num_total_segs : int
        Current total number of segments in the mycelium.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    Purpose
    -------
    Checks if segment idx fuses with another segment and updates accordingly
    """

    # Endpoints of current hyphal segment
    xy1_c = mycelia['xy1'][idx,:]
    xy2_c = mycelia['xy2'][idx,:]

    # Branch id for the current segment
    branch_idx = mycelia['branch_id'][idx]

    # Segments on the same branch
    segs_on_branch = np.where(mycelia['branch_id'][:num_total_segs] == branch_idx)[0]

    # Neighbors of current segment
    seg_nbrs = mycelia['nbr_idxs'][idx]

    # Combined list of neighbors & segments on branch
    skip_ids = np.unique(np.concatenate((segs_on_branch, seg_nbrs)))

    # Define the zone to look for interactions in
    # minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['sl'])
    minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['kg1_wall']*params['dt_i'])

    # Initialize fusion flags
    seg_fuse = -1
    
    # We will keep track how many times CFL-failed segment has been created.
    # Why you ask? Well, because it is possible that after we established one CFL-failed segment,
    # it has to be cut and re-established again. In addition, the data structure manipulation is
    # much more complicated than the intra-branch fusion case...
    formation_of_CFLFail_seg = 0

    # Loop through segments to check for intersections
    for other_idx in range(num_total_segs):
        target_idx = other_idx
        # Here we use 'if not' to check if the nbr_idxs is empty for 'target_idx'.
        
        # If target_idx has no neighbor, then it is a deleted segment 
        # (not a bypassed segment that joins two branches but fails CFL).
        ######################################################################
        if not mycelia['nbr_idxs'][target_idx]:
            continue
        if mycelia['branch_id'][target_idx] == -1 and mycelia['bypass'][target_idx] == False:
            continue
        ######################################################################
        
        # Don't need to check other segments on the same branch or neighboring segments
        if target_idx not in skip_ids:

            # Name the endpoints of other segments
            xy1_o = mycelia['xy1'][target_idx,:]
            xy2_o = mycelia['xy2'][target_idx,:]

            # Check if any enpoint inside box
            if check_if_in_box(mycelia, xy1_o, xy2_o, minx, maxx, miny, maxy):

                # Check if intersection point in the segments
                found_intxn, x_intxn, y_intxn = get_seg_intxn(xy1_c,xy2_c,
                                                              xy1_o,xy2_o)
                # If the intersection occursr with a bypassed segment, switch the intersection
                # to its neighbor
                

                # breakpoint()

                # Update info if intersection is found
                if found_intxn:
                    if target_idx in mycelia['nbr_idxs'][idx]:
                        # print('Why is the segment searching its existing neighbor to find intersection?')
                        # print('Something is wrong')
                        continue
                        # breakpoint()
                   
                    prob = np.random.uniform(0, 1, 1)
                    if (prob > chance_to_fuse):
                        # print('Intersection found but fails probability check!')
                        continue
                    # else:
                         # print('Intersection found')
                    # If the intersection is with a bypassed segment, the intersection
                    # is re-establish with a neighbor of the bypassed segment.
                    if mycelia['bypass'][target_idx] == True:
                        # continue
                        # print('Fusion triggered with a bypassed segment.')
                        nbr_idx = mycelia['nbr_idxs'][target_idx]
                        # Now we search neighbors of the bypassed segment to reestablish connection.
                        # We will choose one that doesn't belong to the same branch as idx and is not already a neighbor of idx
                        for i in range(len(nbr_idx)):  
                            if mycelia['branch_id'][nbr_idx[i]] != mycelia['branch_id'][idx] and nbr_idx[i] not in mycelia['nbr_idxs'][idx]:
                                target_idx = nbr_idx[i]        
                                break
                        # If such search fails to return a target_idx that is different from other_idx, it means that 
                        # idx and other_idx are probably already bridged by a bypassed segment, hence treating as immediate neighbors.
                        # In such case, we will skip testing this other_idx further.
                        if (target_idx == other_idx or mycelia['bypass'][target_idx]==True or mycelia['branch_id'][target_idx]==-1):
                            # breakpoint()
                            continue
                        x_intxn = (mycelia['xy1'][target_idx, 0] + mycelia['xy2'][target_idx, 0])/2.0
                        y_intxn = (mycelia['xy1'][target_idx, 1] + mycelia['xy2'][target_idx, 1])/2.0  
                    # else:
                        # print('Start fusing with a non-bypassed segment.')
                        # print('fusion occuring: {} (branch {} seg {}) fusing to {} (branch {} seg {})'.format(
                            # idx, mycelia['branch_id'][idx][0], mycelia['seg_id'][idx][0],
                            # target_idx, mycelia['branch_id'][target_idx][0], mycelia['seg_id'][target_idx][0]))
                        # print(' seg length before fusion: ', mycelia['seg_length'][idx])
                        # if mycelia['branch_id'][idx] == -1:
                        #     breakpoint()

                    # breakpoint()
                    # Update the endoints
                    mycelia['xy2'][idx,0] = x_intxn
                    mycelia['xy2'][idx,1] = y_intxn
                        
                    # Reset the length of the segment
                    new_seg_len = calc_dist(mycelia['xy1'][idx,:], mycelia['xy2'][idx,:])
                    short_multi_seg_fuse = False

                    print('Fusing in anastomosis')
                    # If the segment is long enough (satisfies CFL), proceed normally
                    if new_seg_len >= params['dt_i']*params['kg1_wall']:
                        mycelia['seg_length'][idx] = new_seg_len
                        mycelia['is_tip'][idx] = False
                        mycelia['can_branch'][idx] = True
                        print('Fusion between segments of two branches resulting a CFL-satisfying segment:', new_seg_len)
                        mycelia = set_anastomosis_nbrs(mycelia, idx, target_idx, seg_fuse, short_multi_seg_fuse)
                        # Rename the endpoints of the current hyphae segment
                        xy2_c = mycelia['xy2'][idx,:]
                        if (np.isnan(np.sum(xy2_c))):
                            breakpoint()
    
                        # Define zone to look for interections in
                        # minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['sl'])
                        minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['kg1_wall']*params['dt_i'])
    
                        # Update fusion neighbor marker
                        seg_fuse = target_idx
                        # print('seg_fuse : ', seg_fuse)
                    # If the segment is too short -> stability issues (fails CFL)
                    # else:
                    #     # If the segment is not the first on the branch, merge it with the previous seg on that branch
                    #     if mycelia['seg_id'][idx] > 0:
                    #         # print('CFL-failing tip is combined with its previous segment to satisfy CFL before fusion.')
                    #         branch_idx = mycelia['branch_id'][idx]
                    #         seg_idx = mycelia['seg_id'][idx]
                    #         #breakpoint()
                    #         # Find ID of previous seg on that branch
                    #         segs_on_branch = np.where(mycelia['branch_id']==branch_idx)[0]
                    #         where_is_idx = np.where(segs_on_branch == idx)[0]
                    #         prev_idx = segs_on_branch[where_is_idx-1][0]
                    #         # breakpoint()
                    #         # Redefine the endpoint of that segment to be the intersection point
                    #         mycelia['xy2'][prev_idx,0] = x_intxn
                    #         mycelia['xy2'][prev_idx,1] = y_intxn
                    #         # Redefine the nbrs for the previous segment
                    #         # Find where idx is listed as a neighbor of prev_idx, 
                    #         #   and we will delete that information.
                    #         # find_idx_on_nbr_idxs = (np.where(mycelia['nbr_idxs'][prev_idx]==idx)[0])[0]
                    #         # AA = np.where(mycelia['nbr_idxs'][prev_idx] == find_idx_on_nbr_idxs)
                    #         # breakpoint()
                    #         # A = mycelia['nbr_idxs'][prev_idx].pop(find_idx_on_nbr_idxs)
                    #         # mycelia['nbr_num'][prev_idx] -= 1
                    #         if mycelia['nbr_num'][prev_idx] < 1:
                    #             breakpoint()
                    #         # elif mycelia['nbr_num'][prev_idx] > 1:
                    #             # breakpoint()
                    #         # Redistribute the existing cw_i and gluc_i to the previous segment (?)
                    #         # STILL NEED CONFIRMATION IF THIS IS OKAY.....
                    #         mycelia['cw_i'][prev_idx] += (mycelia['cw_i'][idx])#/len(prev_idx))
                    #         mycelia['gluc_i'][prev_idx] += (mycelia['gluc_i'][idx])#/len(prev_idx))
                    #         mycelia['treha_i'][prev_idx] += (mycelia['treha_i'][idx])
                            
                    #         # Here we record the resulting length that forced the intra-branch fusion.
                    #         # Note that the deleted segment should have branch_id and seg_id set to -1
                    #         # to avoid being included in the calculation.
                    #         new_seg_len = calc_dist(mycelia['xy1'][idx,:], mycelia['xy2'][idx,:])
                    #         mycelia['seg_length'][idx] = new_seg_len
                            
                    #         # Null out info for the current segment
                    #         mycelia['nbr_idxs'][idx].clear()
                    #         mycelia['branch_id'][idx] = -1
                    #         mycelia['seg_id'][idx] = -1
                    #         mycelia['xy1'][idx,:] = 0
                    #         mycelia['xy2'][idx,:] = 0
                    #         mycelia['angle'][idx] = 0
                    #         mycelia['dist_to_septa'][idx] = 0
                    #         # remember to double check if dist_to_septa calculation successfully
                    #         # avoid calculation with the deleted segments
                    #         mycelia['xy_e_idx'][idx,:] = 0
                    #         mycelia['share_e'][idx] = None
                    #         mycelia['cw_i'][idx] = 0
                    #         mycelia['gluc_i'][idx] = 0
                    #         mycelia['treha_i'][idx] = 0
                    #         mycelia['can_branch'][idx] = False
                    #         mycelia['is_tip'][idx] = False
                    #         mycelia['septa_loc'][idx] = 0
                    #         mycelia['nbr_num'][idx] = 0
                            
                    #         mycelia = set_anastomosis_nbrs(mycelia, idx, target_idx, seg_fuse, short_multi_seg_fuse)
                    #         if idx in mycelia['nbr_idxs'][prev_idx]:
                    #             mycelia['nbr_idxs'][prev_idx].remove(idx)
                    #             mycelia['nbr_num'][prev_idx] -=1
                    #             if mycelia['nbr_num'][prev_idx] < 1:
                    #                 print('A fused segment idx has no prev_idx (for multi-seg branch fusionn)??? What is going on?')
                    #                 breakpoint()
                    #         else:
                    #             print('The deleted segment was not even recorded as a neighbor')
                    #             print('of its previous segment before deletion? Something is wrong')
                    #             breakpoint()   
                            
                    #         prev_idx_seg_len = calc_dist(mycelia['xy1'][prev_idx,:], mycelia['xy2'][prev_idx,:])
                    #         # if prev_idx_seg_len < params['dt']*params['vel_wall']:
                    #         #     breakpoint()
                    #         idx = prev_idx
                    #         new_seg_len = calc_dist(mycelia['xy1'][idx,:], mycelia['xy2'][idx,:])
                    #         mycelia['seg_length'][idx] = new_seg_len
                    #         # print('new seg length after intra-multi-seg branch fusion : ', mycelia['seg_length'][idx])
                    #         if new_seg_len < params['dt']*params['vel_wall']:
                    #             # breakpoint()
                    #             print('After fusing the tip with its previous segment,')
                    #             print('the new segment length still fail CFL? This is because the tip finds an intersection')
                    #             print('that intersect the previous segment of the tip!')
                    #             # breakpoint()
                    #         short_multi_seg_fuse = True
                    #         # Save neighbor information
                    #         mycelia = set_anastomosis_nbrs(mycelia, idx, target_idx, seg_fuse, short_multi_seg_fuse)
                    #         # Rename the endpoints of the current hyphae segment
                    #         xy2_c = mycelia['xy2'][idx,:]
                    #         if (np.isnan(np.sum(xy2_c))):
                    #             breakpoint()
        
                    #         # Define zone to look for interections in
                    #         # minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['sl'])
                    #         minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['kg1_wall']*params['dt'])
        
                    #         # Update fusion neighbor marker
                    #         seg_fuse = target_idx
                    #         # print('seg_fuse : ', seg_fuse)
                            
                    #     # The segment is first on the branch, we set the two segments it is suppopsed to
                    #     # join as direct neighbor.
                    #     else:
                    #         continue
                    #         print('CFL-failing fusion not allowed! (FOR NOW)')
                    #         # print('Fusion formed from a single segment branch with CFL-failing tip.')
                    #         #breakpoint()
                    #         # print('For now, we will remove such segment from existence.')
                    #         # breakpoint()
                    #         nbr_idx = mycelia['nbr_idxs'][idx]
                    #         if len(nbr_idx) > 2:
                    #             print('The segment we are about to set as "bypassed" have additional')
                    #             print('neighbors (>2). These additional neighbors must be addressed appropriately,')
                    #             print('i.e., these neighbor may already exist before the current segment')
                    #             print('fails the CFL condition')
                    #             # breakpoint()
                    #             nbr_idx_to_keep = nbr_idx[0]
                    #             for i in range(len(mycelia['nbr_idxs'][idx])):
                    #                 if mycelia['nbr_idxs'][idx][i]==nbr_idx_to_keep:
                    #                     continue
                    #                 else:
                    #                     if (idx in mycelia['nbr_idxs'][mycelia['nbr_idxs'][idx][i]]):   
                    #                         mycelia['nbr_idxs'][mycelia['nbr_idxs'][idx][i]].remove(idx)
                    #                     if nbr_idx_to_keep not in mycelia['nbr_idxs'][mycelia['nbr_idxs'][idx][i]]:
                    #                         # breakpoint()
                    #                         mycelia['nbr_idxs'][mycelia['nbr_idxs'][idx][i]].append(nbr_idx_to_keep)
                    #                         mycelia['xy2'][mycelia['nbr_idxs'][idx][i]] = (mycelia['xy1'][nbr_idx_to_keep]+mycelia['xy2'][nbr_idx_to_keep])/2
                    #                         mycelia['nbr_idxs'][nbr_idx_to_keep].append(mycelia['nbr_idxs'][idx][i])
                    #                         mycelia['nbr_num'][nbr_idx_to_keep]+=1
                    #                     # But what if this "bypassed" segment already had a branch from it?
                    #                     # How do we identify if such branch originates from this bypassed segment?
                    #         nbr_idx_to_keep = nbr_idx[0]
                    #         if idx in mycelia['nbr_idxs'][nbr_idx_to_keep]:
                    #             mycelia['nbr_idxs'][nbr_idx_to_keep].remove(idx)
                    #             mycelia['nbr_num'][nbr_idx_to_keep]-=1
                                
                    #         else:
                    #             if (mycelia['branch_id'][idx]>-1):  
                    #                 print('The bypassed segment was not even recorded as a neighbor')
                    #                 print('of its previous segment before bypassing? Something is wrong')
                    #                 breakpoint()   
                            
                                    
                    #         # Even though segment idx will be bypassed, we add the target_idx as a neighbor
                    #         # so when another segment intersects this bypassed segment, it has two different
                    #         # segments to re-establish a new connection.
                    #         if formation_of_CFLFail_seg == 0:
                    #             mycelia['nbr_idxs'][idx].append(target_idx)
                    #         elif formation_of_CFLFail_seg > 0:
                    #             mycelia['nbr_idxs'][idx].remove(seg_fuse)
                    #             mycelia['nbr_idxs'][idx].append(target_idx)
                    #         mycelia['bypass'][idx] = True
                    #         # if mycelia['bypass'][idx] == True:
                    #         #     breakpoint()

                    #         # Add the cw & gluc in the seg to the neighboring segments
                    #         mycelia['cw_i'][nbr_idx] += (mycelia['cw_i'][idx]/len(nbr_idx))
                    #         mycelia['gluc_i'][nbr_idx] += (mycelia['gluc_i'][idx]/len(nbr_idx)) 
                    #         mycelia['treha_i'][nbr_idx] += (mycelia['treha_i'][idx]/len(nbr_idx))                                  

                    #         # Here we record the resulting length which forced the bypass condition.
                    #         new_seg_len = calc_dist(mycelia['xy1'][idx,:], mycelia['xy2'][idx,:])
                    #         mycelia['seg_length'][idx] = new_seg_len
                    #         # if new_seg_len < params['dt']*params['vel_wall']
                            
                    #         # Redefine all dictionary values
                    #         # mycelia['bypass'][idx] = True
                    #         mycelia['branch_id'][idx] = -1
                    #         mycelia['seg_id'][idx] = -1
                    #         # mycelia['xy1'][idx,:] = 0
                    #         # mycelia['xy2'][idx,:] = 0
                    #         mycelia['angle'][idx] = 0
                    #         mycelia['dist_to_septa'][idx] = 0
                    #         mycelia['xy_e_idx'][idx,:] = 0
                    #         mycelia['share_e'][idx] = None
                    #         mycelia['cw_i'][idx] = 0
                    #         mycelia['gluc_i'][idx] = 0
                    #         mycelia['treha_i'][idx] = 0
                    #         mycelia['can_branch'][idx] = False
                    #         mycelia['is_tip'][idx] = False
                    #         mycelia['septa_loc'][idx] = 0
                    #         #mycelia['nbr_idxs'][idx].append(target_idx)
                    #         mycelia['nbr_num'][idx] = 0
                    #         short_multi_seg_fuse = False
                            
                    #         # if (len(mycelia['nbr_idxs'][idx])>1):
                    #         #     breakpoint()
                            
                            
                    #         # # Reset num of total segs
                    #         # num_total_segs -= 1
                    #         #breakpoint()
                    #         # Save neighbor information
                    #         # Then remove the neighbor information between the CFL-failing segment with target_idx
                    #         mycelia = set_anastomosis_nbrs_for_CFLfail(mycelia, idx, target_idx, seg_fuse, formation_of_CFLFail_seg, nbr_idx_to_keep)
                    #         # First we set the two segments joined by CFL-failing segment to be neighbors.
                    #         mycelia = set_anastomosis_nbrs_for_segs_joined_by_CFLfail(mycelia, nbr_idx[0], target_idx, seg_fuse)
                            
                    #         xy2_c = mycelia['xy2'][idx,:]
                    #         if (np.isnan(np.sum(xy2_c))):
                    #             breakpoint()
        
                    #         # Define zone to look for interections in
                    #         # minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['sl'])
                    #         minx, maxx, miny, maxy = get_box(xy1_c, xy2_c, params['kg1_wall']*params['dt'])
        
                    #         # Update fusion neighbor marker
                    #         seg_fuse = target_idx
                    #         # print('seg_fuse : ', seg_fuse)
                            
                    #         formation_of_CFLFail_seg += 1
                    #         # print('formaton_of_CFLFail_seg : ', formation_of_CFLFail_seg)
                else:
                    if idx in mycelia['nbr_idxs'][target_idx]:
                        if target_idx not in mycelia['nbr_idxs'][idx]:
                            mycelia['nbr_idxs'][target_idx].remove(idx)
                    if target_idx in mycelia['nbr_idxs'][idx]:
                        if idx not in mycelia['nbr_idxs'][target_idx]:
                            mycelia['nbr_idxs'][idx].remove(target_idx)


    return mycelia

# ----------------------------------------------------------------------------

def get_box(xy1, xy2, box_size):
    """
    Parameters
    ----------
    xy1 : list of doubles
        x- and y-coordinates of starting point.
    xy2 : list of doubles
        x- and y-coordinates of ending point.
    box_size : double
        Half the width/height of the search zone

    Returns
    -------
    minx : double
        Lower bound x-coordinate for search zone.
    maxx : double
        Upper bound x-coordinate for search zone.
    miny : double
        Lower bound y-coordinate for search zone.
    maxy : double
        Upper bound y-coordinate for search zone.
    """

    # Define zone to look for interections in
    minx = min(xy1[0], xy2[0]) - box_size
    maxx = max(xy1[0], xy2[0]) + box_size
    # maxx = min(xy1[0], xy2[0]) + box_size This looks like a typo - should be max(...)?
    miny = min(xy1[1], xy2[1]) - box_size
    maxy = max(xy1[1], xy2[1]) + box_size

    return minx, maxx, miny, maxy

# ----------------------------------------------------------------------------

def check_if_in_box(mycelia, xy1_o, xy2_o, minx, maxx, miny, maxy):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    xy1_o : list of doubles
        The x- and y-coordinates of the starting point.
    xy2_o : list of doubles
        The x- and y-coordinates of the ending point.
    minx : double
        Lower bound x-coordinate for search zone.
    maxx : double
        Upper bound x-coordinate for search zone.
    miny : double
        Lower bound y-coordinate for search zone.
    maxy : double
        Upper bound y-coordinate for search zone.

    Returns
    -------
    result : bool
        True if the segment is in search zone, otherwise False.
    """

    result = False

    # Check if either endpoint in the box
    if ((xy1_o[0] > minx and xy1_o[0] < maxx and xy1_o[1] > miny and xy1_o[1] < maxy) or
        (xy2_o[0] > minx and xy2_o[0] < maxx and xy2_o[1] > miny and xy2_o[1] < maxy)):
        result = True

    return result

# ----------------------------------------------------------------------------

def get_seg_intxn(xy1_c, xy2_c, xy1_o, xy2_o):
    """
    Parameters
    ----------
    xy1_c : list of doubles
        The x- and y-coordinates of the current segment's starting point.
    xy2_c : list of doubles
        The x- and y-coordinates of the current segment's ending point.
    xy1_o : list of doubles
        The x- and y-coordinates of the other segment's starting point.
    xy2_o : list of doubles
        The x- and y-coordinates of the other segment's ending point.

    Returns
    -------
    found_intersection : bool
        True if intersection occured, False otherwise.
    x_intersect : double
        x-coordinate of intersection point or 0 (if no intersection).
    y_intersect : double
        y-coordinate of intersection point or 0 (if no intersection).

    Description of method
    ---------------------
    Source : https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """

    x1, y1 = xy1_c
    x2, y2 = xy2_c
    x3, y3 = xy1_o
    x4, y4 = xy2_o

    t_numerator = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    u_numerator = -(x1-x2)*(y1-y3) + (y1-y2)*(x1-x3)
    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)

    if denominator == 0:
        found_intersection = False
        x_intersect = 0
        y_intersect = 0
    else:
        t = t_numerator/denominator
        u = u_numerator/denominator
        if (t >= 0) and (t <= 1) and (u >= 0) and (u <= 1):
            found_intersection = True
            x_intersect = x1 + t*(x2-x1)
            y_intersect = y1 + t*(y2-y1)
        else:
            found_intersection = False
            x_intersect = 0
            y_intersect = 0

    return found_intersection, x_intersect, y_intersect

# ----------------------------------------------------------------------------

def set_anastomosis_nbrs(mycelia, idx, other_idx, seg_fuse, short_multi_seg_fuse):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idx : int
        The index of the segment that is being tested for fusion.
    other_idx : int
        The index of the segment that idx fused with.
    seg_fuse : int
        The index of the segment that idx previously fused with.
    short_multi_seg_fuse: bool
        Indicate whether the intra-branch fusion due to short segment (condition 2 in anastomosis function)
        has occured. Because if an intra-branch fusion has occur after establishing a intersection with seg_fuse,
        the fused segment has already erased its neighbor information and the resulting new segment will not have
        seg_fuse as its existing neighbor.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    """
    
    # Save neighbor information
    if short_multi_seg_fuse == True:
        mycelia['nbr_idxs'][idx].append(other_idx)
        mycelia['nbr_idxs'][other_idx].append(idx)
        mycelia['nbr_num'][idx] += 1
        mycelia['nbr_num'][other_idx] += 1
        print('short_multi_seg_fuse == True')
    else:
        mycelia['nbr_idxs'][idx].append(other_idx)
        mycelia['nbr_idxs'][other_idx].append(idx)
        mycelia['nbr_num'][idx] += 1
        mycelia['nbr_num'][other_idx] += 1
        print('short_multi_seg_fuse == False')
        # mycelia['nbr_idxs'][other_idx].remove(idx)
        # mycelia['nbr_num'][other_idx] -= 1

    # Remove neighbor connection if needed
    if seg_fuse > 0 and short_multi_seg_fuse == False:
        # mycelia['nbr_idxs'][idx].remove(seg_fuse)
        # if idx not in mycelia['nbr_idxs'][seg_fuse]:
            # breakpoint()
        if mycelia['branch_id'][idx]>-1:
            mycelia['nbr_idxs'][seg_fuse].remove(idx)
            mycelia['nbr_idxs'][idx].remove(seg_fuse)
            # mycelia['nbr_num'][idx] -= 1
            mycelia['nbr_num'][seg_fuse] -= 1
            print('Remove neighbor connection if needed')        
    
        # mycelia['nbr_idxs'][seg_fuse].remove(idx)

    # Segment is no longer a tip
    mycelia['is_tip'][idx] = False

    # Segment cannot branch??
    mycelia['can_branch'][idx] = False
    mycelia['can_branch'][other_idx] = False

    return mycelia

# ----------------------------------------------------------------------------
def set_anastomosis_nbrs_for_segs_joined_by_CFLfail(mycelia, idx, other_idx, seg_fuse):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idx : int
        The index of the segment that is being tested for fusion.
    other_idx : int
        The index of the segment that idx fused with.
    seg_fuse : int
        The index of the segment that idx previously fused with.
    short_multi_seg_fuse: bool
        Indicate whether the intra-branch fusion due to short segment (condition 2 in anastomosis function)
        has occured. Because if an intra-branch fusion has occur after establishing a intersection with seg_fuse,
        the fused segment has already erased its neighbor information and the resulting new segment will not have
        seg_fuse as its existing neighbor.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    """
    # print('set anastomosis nbrs information between segments joined by CFL-failed segment')
    # Save neighbor information
    mycelia['nbr_idxs'][idx].append(other_idx)
    mycelia['nbr_idxs'][other_idx].append(idx)
    mycelia['nbr_num'][idx] += 1
    mycelia['nbr_num'][other_idx] += 1

    # Remove neighbor connection if needed
    if seg_fuse > 0:
        # where_idx_has_segfuse_as_nbr = np.where(mycelia['nbr_idxs'][idx]==seg_fuse)
        if seg_fuse in mycelia['nbr_idxs'][idx]:
        # if not where_idx_has_segfuse_as_nbr:
            mycelia['nbr_idxs'][idx].remove(seg_fuse)
            mycelia['nbr_idxs'][seg_fuse].remove(idx)
            mycelia['nbr_num'][idx] -= 1
            mycelia['nbr_num'][seg_fuse] -= 1
            # breakpoint()
            


    return mycelia

# ----------------------------------------------------------------------------

def set_anastomosis_nbrs_for_CFLfail(mycelia, idx, other_idx, seg_fuse, formation_of_CFLFail_seg, nbr_idx_to_keep):
    """
    Parameters
    ----------
    mycelia : dictionary
        Stores structural information of mycelia colony for all hyphal segments.
    idx : int
        The index of the segment that is being tested for fusion.
    other_idx : int
        The index of the segment that idx fused with.
    seg_fuse : int
        The index of the segment that idx previously fused with.
    short_multi_seg_fuse: bool
        Indicate whether the intra-branch fusion due to short segment (condition 2 in anastomosis function)
        has occured. Because if an intra-branch fusion has occur after establishing a intersection with seg_fuse,
        the fused segment has already erased its neighbor information and the resulting new segment will not have
        seg_fuse as its existing neighbor.

    Returns
    -------
    mycelia : dictionary
        Updated structural information of mycelia colony for all hyphal segments.

    """
    # print('set anastomosis nbrs information for CFL-failed segment')
    # Save neighbor information
    if other_idx not in mycelia['nbr_idxs'][idx]:
        mycelia['nbr_idxs'][idx].append(other_idx)
    # mycelia['nbr_idxs'][other_idx].append(idx)
    # mycelia['nbr_num'][idx] += 1
    # mycelia['nbr_num'][other_idx] += 1
    if idx in mycelia['nbr_idxs'][other_idx]:
        mycelia['nbr_idxs'][other_idx].remove(idx)
        mycelia['nbr_num'][other_idx] -= 1
    
    if other_idx == nbr_idx_to_keep:
        breakpoint()
    else:
        is_idx_in_nbr_of_other_idx = idx in mycelia['nbr_idxs'][other_idx]
        if is_idx_in_nbr_of_other_idx == True:
            mycelia['nbr_idxs'][other_idx].remove(idx)
            mycelia['nbr_num'][other_idx] -= 1
            

    # Remove neighbor connection if needed
    if seg_fuse != -1 and formation_of_CFLFail_seg > 0:
        # print('set anastomosis nbrs information for re-establishing CFL-failed segment')
        if seg_fuse in mycelia['nbr_idxs'][idx]:
            mycelia['nbr_idxs'][idx].remove(seg_fuse)
        # mycelia['nbr_idxs'][seg_fuse].remove(idx)
        # mycelia['nbr_num'][idx] -= 1
        # mycelia['nbr_num'][seg_fuse] -= 1
        if idx in mycelia['nbr_idxs'][seg_fuse]:
            mycelia['nbr_idxs'][seg_fuse].remove(idx)
            mycelia['nbr_num'][seg_fuse] -= 1
    elif seg_fuse != -1 and formation_of_CFLFail_seg == 0:
        mycelia['nbr_idxs'][idx].remove(seg_fuse)
        # mycelia['nbr_num'][idx] -= 1
        if idx in mycelia['nbr_idxs'][seg_fuse]:
            mycelia['nbr_idxs'][seg_fuse].remove(idx)
            mycelia['nbr_num'][seg_fuse] -= 1
        
    # elif seg_fuse == -1 and len(mycelia['nbr_idxs'][idx]) == 2:
    #     if nbr_idx_to_keep == mycelia['nbr_idxs'][idx][1]:
    #         breakpoint()
    #     mycelia['nbr_idxs'][idx].remove(mycelia['nbr_idxs'][idx][1])
        
    # if len(mycelia['nbr_idxs'][idx])>1:
    #     breakpoint()
    # elif seg_fuse < 0 and formation_of_CFLFail_seg == 0:
    #     mycelia['nbr_idxs'][other_idx].remove(idx)
    #     mycelia['nbr_num'][other_idx] -= 1
        
    # Segment is no longer a tip
    mycelia['is_tip'][idx] = False

    # Segment cannot branch??
    mycelia['can_branch'][idx] = False
    mycelia['can_branch'][other_idx] = False

    return mycelia

# ----------------------------------------------------------------------------
