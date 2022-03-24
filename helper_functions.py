#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:17:45 2020

@author: jolenebritton
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from matplotlib import collections  as mc
sns.set_style('white')
# sns.set_context("talk")
sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
import configparser
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap


# ----------------------------------------------------------------------------
# SET UP FUNCTIONS
# ----------------------------------------------------------------------------

def get_configs(config_filename):
    """
    Parameters
    ----------
    config_filename : str
        Filepath for the .ini config file containing parameters.

    Returns
    -------
    params_dict : dict
        A dictionary containing all parameters in a usable form.
    config : TYPE
        The config file that was loaded into this function.

    """
    config = configparser.ConfigParser()
    config.read(config_filename)
    # breakpoint()
    # Extract the sections
    discrete_params = config['DISCRETE PARAMS']
    growth_params = config['GROWTH PARAMS']
    nutrient_params = config['NUTRIENT PARAMS']

    # Generate extra values
    diam =  discrete_params.getfloat('hy_diam')
    sl =  discrete_params.getfloat('hy_compartment')
    cross_area = np.pi*(0.5*diam)**2
    init_vol_seg = sl*cross_area

    dy = discrete_params.getfloat('grid_len')
    dz = discrete_params.getfloat('grid_height')
    vol_grid = dy*dy*dz
    diff_e_gluc = nutrient_params.getfloat('diffusion_e_gluc')
    diff_i_gluc = nutrient_params.getfloat('diffusion_i_gluc')
    vel_wall = nutrient_params.getfloat('vel_wall')

    # dt = 0.99*(dy**2)/(4*diff_e_gluc)
    dt = 0.75*min((sl**2/diff_i_gluc),(sl/vel_wall))
    #dt = 0.5*0.75*min((sl**2/diff_i_gluc),(sl/vel_wall))

    up_state = nutrient_params['up_state']
    if up_state == 'repressed':
        Ku2 = nutrient_params.getfloat('Ku2a_gluc')*vol_grid
    else:
        Ku2 = nutrient_params.getfloat('Ku2b_gluc')*vol_grid

    # Save to a dictionary
    params_dict = {
        # SECTION 1: Discretization Parameters
        'dt' : dt,
        'final_time' : discrete_params.getfloat('final_time'),
        'plot_units_time' : discrete_params['plot_units_time'],

        'sl' : sl,
        'dy' :  dy,
        'vol_grid': vol_grid,
        'plot_units_space' : discrete_params['plot_units_space'],
        'init_segs_count' : discrete_params.getint('init_segs_count'),
        'environ_type' : discrete_params['environ_type'],
        'cross_area' : cross_area,
        'init_vol_seg' : init_vol_seg,
        'septa_len' : 1,
        'grid_scale_val' : discrete_params.getfloat('grid_scale_val'),
        'hy_diam' : discrete_params.getfloat('hy_diam'),

        # SECTION 2: Extension & Branching for Growth Parameters
        'angle_sd' : growth_params.getfloat('angle_sd')*(np.pi/180),
        'branch_mean' : growth_params.getfloat('branch_mean')*(np.pi/180),
        'branch_sd' : growth_params.getfloat('branch_sd')*(np.pi/180),
        'branch_cost' : growth_params.getfloat('branch_cost'),
        'branch_rate' : growth_params.getfloat('branch_rate'),
        'hy_density' :  growth_params.getfloat('hy_density'),
        'f_dw' :  growth_params.getfloat('f_dw'),
        'f_wall' :  growth_params.getfloat('f_wall'),
        'f_cw_cellwall' :  growth_params.getfloat('f_cw_cellwall'),

        # SECTION 3: Internal & External Nutrient Parameters
        'init_sub_e_dist' : nutrient_params['init_sub_e_dist'],
        'init_sub_e_gluc' : nutrient_params.getfloat('init_sub_e_gluc')*vol_grid,
        'init_sub_e_treha' : nutrient_params.getfloat('init_sub_e_treha')*vol_grid,
        'diffusion_e_gluc' : diff_e_gluc,

        'init_sub_i_gluc' : nutrient_params.getfloat('init_sub_i_gluc'),
        'diffusion_i_gluc' : nutrient_params.getfloat('diffusion_i_gluc'),
        'vel_gluc' : nutrient_params.getfloat('vel_gluc'),
        'vel_wall' : nutrient_params.getfloat('vel_wall'),

        # 'm_gluc' : nutrient_params.getfloat('m_gluc'),
        # 'rho' : nutrient_params.getfloat('rho')*vol_seg,

        'ku1_gluc' : nutrient_params.getfloat('ku1_gluc'),
        'Ku2_gluc' : Ku2,
        'yield_u' : nutrient_params.getfloat('yield_u'),

        #'kc1_gluc' : nutrient_params.getfloat('kc1_gluc'),
        #'Kc2_gluc' : nutrient_params.getfloat('Kc2_gluc'),
        'kc1_gluc' : nutrient_params.getfloat('ku1_gluc'),
        'Kc2_gluc' : Ku2,
        'yield_c' : nutrient_params.getfloat('yield_c'),

        'kg1_wall' : nutrient_params.getfloat('kg1_wall'),
        # 'Kg2_wall' : nutrient_params.getfloat('Kg2_wall')*vol_seg,
        'Kg2_wall' : nutrient_params.getfloat('Kg2_wall'),
        'yield_g' : nutrient_params.getfloat('yield_g'),
        'mw_cw' : nutrient_params.getfloat('mw_cw'),
        'mw_glucose' : nutrient_params.getfloat('mw_glucose'),

        'num_v' : nutrient_params.getfloat('num_v')
    }

    if not('yield_c_in_mmoles' in params_dict):
    	params_dict['yield_c_in_mmoles'] = params_dict['yield_c']*params_dict['mw_glucose']/params_dict['mw_cw']

    use_original = 0

    if(use_original != 1):
        # Max rate of moles of cell wall raw material used per time step:
        max_gms_cw_per_time = params_dict['kg1_wall']* np.pi*(params_dict['hy_diam']/2.0)**2.0 \
                        * params_dict['hy_density'] * params_dict['f_dw']* params_dict['f_wall'] * params_dict['f_cw_cellwall']
        max_moles_cw_per_time = max_gms_cw_per_time / params_dict['mw_cw']

        #Max rate of conversion of glucose to cell wall material is the maxrate of
        # grams of cell wall used per step/timestep / (gms cw produced per gms glucose used)/ (MW glucose)
        params_dict['yield_in_moles'] = params_dict['yield_c']*params_dict['mw_glucose']/params_dict['mw_cw']
        #params_dict['kc1_gluc'] =  max_moles_cw_per_time/yield_in_moles
        #params_dict['ku1_gluc'] = params_dict['kc1_gluc']
        
        params_dict['vel_wall'] = params_dict['kg1_wall']


    return params_dict, config

# ----------------------------------------------------------------------------

def get_filepath(params):
    """
    Parameters
    ----------
    params : dict
        Dictionary of parameters to be used in simulation.

    Returns
    -------
    folder_string : str
        Path for the folder where results will be stored.
    param_string : str
        Filename to be used to label stored results, contains parameter info.

    """
    # folder_string = "ihc={}_ext={}_dy={}_sl={}_dt={:.3}_ft={}".format(
    #                                                         params['init_segs_count'],
    #                                                         params['init_sub_e_dist'],
    #                                                         params['dy'],
    #                                                         params['sl'],
    #                                                         params['dt'],
    #                                                         params['final_time'])
    folder_string = "oldD2Tip_Fus_tipRe_brRate2e8_resBr4_noBkDiff_bkPatchy2_"
    # folder_string = "calibration"
    # file_string = "{}_b={:.3e}_ieg={}_deg={}_iig={:.3e}_dig={}_vw={}_kyu={},{:.3e},{}_kyc={:.3e},{:.3e},{}_kyg={},{:.3e},{}".format(
    #     folder_string,
    #     params['branch_rate'],
    #     params['init_sub_e_gluc'], params['diffusion_e_gluc'],
    #     params['init_sub_i_gluc'], params['diffusion_i_gluc'],
    #     # params['vel_gluc'],
    #     params['vel_wall'],
    #     params['ku1_gluc'], params['Ku2_gluc'], params['yield_u'],
    #     params['kc1_gluc'], params['Kc2_gluc'], params['yield_c'],
    #     params['kg1_wall'], params['Kg2_wall'], params['yield_g'])
    file_string = "oldD2Tip_Fus_tipRe_brRate2e8_resBr4_noBkDiff_bkPatchy2_"
    # file_string = "calibration"
    #file_string = "univSecr_metTransp_brRate5e8_NObkDiff_oldD2Tip_patchyExt_Fus_gridScaleVal80"
    return folder_string, file_string


# ----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------

def plot_fungus(mycelia, num_total_segs, curr_time, folder_string, param_string, params, run):
    """
    Parameters
    ----------
    hy : list
        List of class instances containing information about each hyphae segment.
    curr_time : double
        The current time of simulation, in days.
    param_string : str
        Used to create filename of saved plot.

    Returns
    -------
    None.

    Purpose
    -------
    Plot fungal mycelia network.
    Color of a segment corresponds to internal substrate concentration.

    """
    # cur_len = len(hy)
    idx_to_display = np.where(mycelia['branch_id'][:num_total_segs]>-1)[0]
    x1 = mycelia['xy1'][idx_to_display, 0].tolist()
    x2 = mycelia['xy2'][idx_to_display, 0].tolist()
    y1 = mycelia['xy1'][idx_to_display, 1].tolist()
    y2 = mycelia['xy2'][idx_to_display, 1].tolist()
    si = mycelia['cw_i'][idx_to_display].flatten()
    # x1 = mycelia['xy1'][:num_total_segs, 0].tolist()
    # x2 = mycelia['xy2'][:num_total_segs, 0].tolist()
    # y1 = mycelia['xy1'][:num_total_segs, 1].tolist()
    # y2 = mycelia['xy2'][:num_total_segs, 1].tolist()
    # si = mycelia['cw_i'][:num_total_segs].flatten()

    if any(si == 0.0):
        min_value = min(si[(si > 0)])
        si[np.where(si == 0.0)] = min_value /10
    if(np.any(np.isnan(si))):
        breakpoint()
    si = np.log10(si)


    segments = []
    for xi1, yi1, xi2, yi2 in zip(x1, y1, x2, y2):
        segments.append([(xi1, yi1), (xi2, yi2)])

    # Generated plot
    fig, ax = pl.subplots(dpi=600)

    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    bottom = cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    # Plot linesegments with coloring according to internal substrate conc.
    lc = mc.LineCollection(segments, array=si, cmap=cm.jet)#orange_blue)
    lc.set_linewidth(1)
    ax.add_collection(lc)

    # plt.scatter(x1,y1,s=0.1)

    # Colorbar
    ax.add_collection(lc)
    fc = fig.colorbar(lc)
    fc.set_label('Internal Cell Wall (for growth) Concentration')
    fc.outline.set_visible(False)

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time


    # Set labels, title, margins, etc.
    # ax.set_ylabel('dm')
    # ax.set_xlabel('dm')
    ax.set_ylabel('{}'.format(params['plot_units_space']))
    ax.set_xlabel('{}'.format(params['plot_units_space']))
    ax.set_title('Mycelia Network \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
    ax.axis('equal')
    ax.margins(0.1)
    # breakpoint()

    # Show the plot
    sns.despine()
    #plt.show()

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_t={:0.2f}_mycelia_cellwall_{}.png".format(param_string,
                                                                     run,
                                                                     param_string,
                                                                     curr_time,
                                                                     run)
    fig.savefig(fig_name)
    plt.close()

# ----------------------------------------------------------------------------

def plot_fungus_gluc(mycelia, num_total_segs, curr_time, folder_string, param_string, params, run):
    """
    Parameters
    ----------
    hy : list
        List of class instances containing information about each hyphae segment.
    curr_time : double
        The current time of simulation, in days.
    param_string : str
        Used to create filename of saved plot.

    Returns
    -------
    None.

    Purpose
    -------
    Plot fungal mycelia network.
    Color of a segment corresponds to internal substrate concentration.

    """
    # cur_len = len(hy)
    idx_to_display = np.where(mycelia['branch_id'][:num_total_segs]>-1)[0]
    x1 = mycelia['xy1'][idx_to_display, 0].tolist()
    x2 = mycelia['xy2'][idx_to_display, 0].tolist()
    y1 = mycelia['xy1'][idx_to_display, 1].tolist()
    y2 = mycelia['xy2'][idx_to_display, 1].tolist()
    si = mycelia['gluc_i'][idx_to_display].flatten()
    # x1 = mycelia['xy1'][:num_total_segs, 0].tolist()
    # x2 = mycelia['xy2'][:num_total_segs, 0].tolist()
    # y1 = mycelia['xy1'][:num_total_segs, 1].tolist()
    # y2 = mycelia['xy2'][:num_total_segs, 1].tolist()
    # si = mycelia['gluc_i'][:num_total_segs].flatten()

    if any(si ==0.0):
        min_value = min(si[(si > 0)])
        si[np.where(si == 0.0)] = min_value /10
    si = np.log10(si)

    segments = []
    for xi1, yi1, xi2, yi2 in zip(x1, y1, x2, y2):
        segments.append([(xi1, yi1), (xi2, yi2)])

    # Generated plot
    fig, ax = pl.subplots(dpi=600)

    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    bottom = cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    # Plot linesegments with coloring according to internal substrate conc.
    lc = mc.LineCollection(segments, array=si, cmap=cm.jet)#orange_blue)
    lc.set_linewidth(1)
    ax.add_collection(lc)

    # plt.scatter(x1,y1,s=0.1)

    # Colorbar
    ax.add_collection(lc)
    fc = fig.colorbar(lc)
    fc.set_label('Internal Glucose Concentration')
    fc.outline.set_visible(False)

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time


    # Set labels, title, margins, etc.
    # ax.set_ylabel('dm')
    # ax.set_xlabel('dm')
    ax.set_ylabel('{}'.format(params['plot_units_space']))
    ax.set_xlabel('{}'.format(params['plot_units_space']))
    ax.set_title('Mycelia Network \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
    ax.axis('equal')
    ax.margins(0.1)
    # breakpoint()

    # Show the plot
    sns.despine()
    #plt.show()

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_t={:0.2f}_mycelia_gluc_{}.png".format(param_string,
                                                                     run,
                                                                     param_string,
                                                                     curr_time,
                                                                     run)
    fig.savefig(fig_name)
    plt.close()

def plot_fungus_generic(mycelia, num_total_segs, curr_time, folder_string, param_string, params, run):
    """
    Parameters
    ----------
    hy : list
        List of class instances containing information about each hyphae segment.
    curr_time : double
        The current time of simulation, in days.
    param_string : str
        Used to create filename of saved plot.

    Returns
    -------
    None.

    Purpose
    -------
    Plot fungal mycelia network.
    Color of a segment corresponds to internal substrate concentration.

    """
    # cur_len = len(hy)

    x1 = mycelia['xy1'][:num_total_segs, 0].tolist()
    x2 = mycelia['xy2'][:num_total_segs, 0].tolist()
    y1 = mycelia['xy1'][:num_total_segs, 1].tolist()
    y2 = mycelia['xy2'][:num_total_segs, 1].tolist()
    ssi = mycelia['can_branch'][:num_total_segs].flatten()

    #si[np.where(si == 0.0)] = 1.0e-14
    si = np.zeros(ssi.shape)
    si[np.where(ssi == True)] = 500.0
    si[np.where(ssi == False)] = -500.0
    #breakpoint()

    segments = []
    for xi1, yi1, xi2, yi2 in zip(x1, y1, x2, y2):
        segments.append([(xi1, yi1), (xi2, yi2)])

    # Generated plot
    fig, ax = pl.subplots(dpi=600)

    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    bottom = cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    # Plot linesegments with coloring according to internal substrate conc.
    lc = mc.LineCollection(segments, array=si, cmap=cm.jet)#orange_blue)
    lc.set_linewidth(1)
    ax.add_collection(lc)

    # plt.scatter(x1,y1,s=0.1)

    # Colorbar
    ax.add_collection(lc)
    fc = fig.colorbar(lc)
    fc.set_label('Generic')
    fc.outline.set_visible(False)

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time


    # Set labels, title, margins, etc.
    # ax.set_ylabel('dm')
    # ax.set_xlabel('dm')
    ax.set_ylabel('{}'.format(params['plot_units_space']))
    ax.set_xlabel('{}'.format(params['plot_units_space']))
    ax.set_title('Mycelia Network \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
    ax.axis('equal')
    ax.margins(0.1)
    # breakpoint()

    # Show the plot
    sns.despine()
    #plt.show()

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_t={:0.2f}_mycelia_gluc_{}.png".format(param_string,
                                                                     run,
                                                                     param_string,
                                                                     curr_time,
                                                                     run)
    fig.savefig(fig_name)
    plt.close()
    
def plot_fungus_treha(mycelia, num_total_segs, curr_time, folder_string, param_string, params, run):
    """
    Parameters
    ----------
    hy : list
        List of class instances containing information about each hyphae segment.
    curr_time : double
        The current time of simulation, in days.
    param_string : str
        Used to create filename of saved plot.

    Returns
    -------
    None.

    Purpose
    -------
    Plot fungal mycelia network.
    Color of a segment corresponds to internal substrate concentration.

    """
    # cur_len = len(hy)
    idx_to_display = np.where(mycelia['branch_id'][:num_total_segs]>-1)[0]
    x1 = mycelia['xy1'][idx_to_display, 0].tolist()
    x2 = mycelia['xy2'][idx_to_display, 0].tolist()
    y1 = mycelia['xy1'][idx_to_display, 1].tolist()
    y2 = mycelia['xy2'][idx_to_display, 1].tolist()
    si = mycelia['treha_i'][idx_to_display].flatten()
    # x1 = mycelia['xy1'][:num_total_segs, 0].tolist()
    # x2 = mycelia['xy2'][:num_total_segs, 0].tolist()
    # y1 = mycelia['xy1'][:num_total_segs, 1].tolist()
    # y2 = mycelia['xy2'][:num_total_segs, 1].tolist()
    # si = mycelia['gluc_i'][:num_total_segs].flatten()

    # if any(si ==0.0):
        # min_value = min(si[(si > 0)])
        # si[np.where(si == 0.0)] = min_value /10
    # si = np.log10(si)

    segments = []
    for xi1, yi1, xi2, yi2 in zip(x1, y1, x2, y2):
        segments.append([(xi1, yi1), (xi2, yi2)])

    # Generated plot
    fig, ax = pl.subplots(dpi=600)

    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    bottom = cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    # Plot linesegments with coloring according to internal substrate conc.
    lc = mc.LineCollection(segments, array=si, cmap=cm.jet)#orange_blue)
    lc.set_linewidth(1)
    ax.add_collection(lc)

    # plt.scatter(x1,y1,s=0.1)

    # Colorbar
    ax.add_collection(lc)
    fc = fig.colorbar(lc)
    fc.set_label('Internal Trehalose Concentration')
    fc.outline.set_visible(False)

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time


    # Set labels, title, margins, etc.
    # ax.set_ylabel('dm')
    # ax.set_xlabel('dm')
    ax.set_ylabel('{}'.format(params['plot_units_space']))
    ax.set_xlabel('{}'.format(params['plot_units_space']))
    ax.set_title('Mycelia Network \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
    ax.axis('equal')
    ax.margins(0.1)
    # breakpoint()

    # Show the plot
    sns.despine()
    #plt.show()

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_t={:0.2f}_mycelia_treha_{}.png".format(param_string,
                                                                     run,
                                                                     param_string,
                                                                     curr_time,
                                                                     run)
    fig.savefig(fig_name)
    plt.close()

# ----------------------------------------------------------------------------

# def plot_externalsub(sub_e, yticks, yticklabels, curr_time, sub_e_max, plot_type, folder_string, param_string, params, run):
#     """
#     Parameters
#     ----------
#     sub_e : 2D numpy array
#         Matrix containing external nutrient concentration values at discritized grid points.
#     yticks : list
#         Helps determine how many labels appear of x- and y-axes.
#     yticklabels : list
#         Values to appear on the x- and y-axes.
#     curr_time : double
#         The current time of simulation, in days.
#     sub_e_max : double
#         Largest possible value for external substrate concentration.
#     param_string : str
#         Used to create filename of saved plot.

#     Returns
#     -------
#     None.

#     Purpose
#     -------
#     Plot external nutrient concentration.

#     """
#     fig, ax = pl.subplots(dpi=600)
#     # For the orange-blue color map
#     N = 1024#512#256
#     top = cm.get_cmap('Oranges_r', N)#128) # r means reversed version
#     bottom = cm.get_cmap('Blues', N)#128)# combine it all
#     newcolors = np.vstack((top(np.linspace(0, 1, N)),
#                            bottom(np.linspace(0, 1, N))))# create a new colormaps with a name of OrangeBlue
#     orange_blue = ListedColormap(newcolors, name='OrangeBlue')

#     # Convert units
#     if params['plot_units_time'] == 'days':
#         plot_time = curr_time / (60*60*24)
#     elif params['plot_units_time'] == 'hours':
#         plot_time = curr_time / (60*60)
#     elif params['plot_units_time'] == 'minutes':
#         plot_time = curr_time / 60
#     elif params['plot_units_time'] == 'seconds':
#         plot_time = curr_time

#     # Plot
#     if plot_type == 'Se':
#         ax = sns.heatmap(np.transpose(sub_e), cmap=orange_blue, vmin=0, vmax=sub_e_max, xticklabels=yticklabels, yticklabels=yticklabels)
#     elif plot_type == 'Ce':
#         ax = sns.heatmap(np.transpose(sub_e), cmap=orange_blue, vmin=0, xticklabels=yticklabels, yticklabels=yticklabels)
#     ax.set_yticks(yticks)
#     ax.set_xticks(yticks)
#     if plot_type == 'Se':
#         ax.collections[0].colorbar.set_label("External Nutrient Concentration")
#         ax.set_title('External Domain \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
#     elif plot_type == 'Ce':
#         ax.collections[0].colorbar.set_label("Chemical Inhibitor Concentration")
#         ax.set_title('Chemical Domain \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")

#     ax.set_ylabel('{}'.format(params['plot_units_space']))
#     ax.set_xlabel('{}'.format(params['plot_units_space']))
#     ax.invert_yaxis()
#     ax.invert_xaxis()
#     ax.axis('equal')
#     ax.margins(0.1)
#     plt.show()
#     fig_name = "Results/{}/Run{}/{}_t={:0.2f}_external{}_{}_gluc.png".format(param_string,
#                                                                         run,
#                                                                         param_string,
#                                                                         curr_time,
#                                                                         plot_type,
#                                                                         run)
#     fig = ax.get_figure()
#     fig.savefig(fig_name)
    
#-----------------------------------------------------------------------------

def plot_externalsub(sub_e, yticks, yticklabels, curr_time, sub_e_max, plot_type, folder_string, param_string, params, run):
    """
    Parameters
    ----------
    sub_e : 2D numpy array
        Matrix containing external nutrient concentration values at discritized grid points.
    yticks : list
        Helps determine how many labels appear of x- and y-axes.
    yticklabels : list
        Values to appear on the x- and y-axes.
    curr_time : double
        The current time of simulation, in days.
    sub_e_max : double
        Largest possible value for external substrate concentration.
    param_string : str
        Used to create filename of saved plot.

    Returns
    -------
    None.

    Purpose
    -------
    Plot external nutrient concentration.

    """

    # For the orange-blue color map
    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    bottom = cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time
    # breakpoint()
    # Plot
    if plot_type == 'Se':
        ax = sns.heatmap(np.transpose(sub_e), cmap=orange_blue, vmin=0, vmax=sub_e_max)#, xticklabels=yticklabels, yticklabels=yticklabels)
    # elif plot_type == 'Ce':
    #     ax = sns.heatmap(np.transpose(sub_e), cmap=orange_blue, vmin=0, xticklabels=yticklabels, yticklabels=yticklabels)
    # breakpoint()
    ax.set_yticks(yticks)
    ax.set_xticks(yticks)
    if plot_type == 'Se':
        ax.collections[0].colorbar.set_label("External Glucose Concentration")
        ax.set_title('External Domain \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
    elif plot_type == 'Ce':
        ax.collections[0].colorbar.set_label("Chemical Inhibitor Concentration")
        ax.set_title('Chemical Domain \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")

    ax.set_ylabel('{}'.format(params['plot_units_space']))
    ax.set_xlabel('{}'.format(params['plot_units_space']))
    ax.invert_yaxis()
    ax.axis('equal')
    ax.margins(0.1)
    # ax.set(yticklabels=[])
    # ax.set(xticklabels=[])
    # ax.invert_xaxis()
    #plt.show()
    fig_name = "Results/{}/Run{}/{}_t={:0.2f}_external_gluc_{}_{}.png".format(param_string,
                                                                        run,
                                                                        param_string,
                                                                        curr_time,
                                                                        plot_type,
                                                                        run)
    fig = ax.get_figure()
    fig.savefig(fig_name)
    plt.close()

def plot_externalsub_treha(sub_e, yticks, yticklabels, curr_time, sub_e_max, plot_type, folder_string, param_string, params, run):
    """
    Parameters
    ----------
    sub_e : 2D numpy array
        Matrix containing external nutrient concentration values at discritized grid points.
    yticks : list
        Helps determine how many labels appear of x- and y-axes.
    yticklabels : list
        Values to appear on the x- and y-axes.
    curr_time : double
        The current time of simulation, in days.
    sub_e_max : double
        Largest possible value for external substrate concentration.
    param_string : str
        Used to create filename of saved plot.

    Returns
    -------
    None.

    Purpose
    -------
    Plot external nutrient concentration.

    """

    # For the orange-blue color map
    top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    bottom = cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    orange_blue = ListedColormap(newcolors, name='OrangeBlue')

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time
    # breakpoint()
    # Plot
    if plot_type == 'Se':
        ax = sns.heatmap(np.transpose(sub_e), cmap=orange_blue, vmin=0, vmax=sub_e_max)#, xticklabels=yticklabels, yticklabels=yticklabels)
    # elif plot_type == 'Ce':
    #     ax = sns.heatmap(np.transpose(sub_e), cmap=orange_blue, vmin=0, xticklabels=yticklabels, yticklabels=yticklabels)
    # breakpoint()
    ax.set_yticks(yticks)
    ax.set_xticks(yticks)
    if plot_type == 'Se':
        ax.collections[0].colorbar.set_label("External Trehalose Concentration")
        ax.set_title('External Domain \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")
    elif plot_type == 'Ce':
        ax.collections[0].colorbar.set_label("Chemical Inhibitor Concentration")
        ax.set_title('Chemical Domain \nTime = {:0.2f} {}'.format(plot_time, params['plot_units_time']),fontweight="bold")

    ax.set_ylabel('{}'.format(params['plot_units_space']))
    ax.set_xlabel('{}'.format(params['plot_units_space']))
    ax.invert_yaxis()
    ax.axis('equal')
    ax.margins(0.1)
    # ax.set(yticklabels=[])
    # ax.set(xticklabels=[])
    # ax.invert_xaxis()
    #plt.show()
    fig_name = "Results/{}/Run{}/{}_t={:0.2f}_external{}_{}.png".format(param_string,
                                                                        run,
                                                                        param_string,
                                                                        curr_time,
                                                                        plot_type,
                                                                        run)
    fig = ax.get_figure()
    fig.savefig(fig_name)
    plt.close()
# ----------------------------------------------------------------------------

def plot_stat(count_times, count_stat, stat_type, folder_string, param_string, params, run):

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_times = count_times
    elif params['plot_units_time'] == 'hours':
        plot_times = 24*count_times
    elif params['plot_units_time'] == 'minutes':
        plot_times = 60*24*count_times
    elif params['plot_units_time'] == 'seconds':
        plot_times = 60*60*24*count_times

    fig, ax = plt.subplots()
    ax.plot(plot_times, count_stat)
    # ax.set_title(stat_type)
    ax.set_xlabel('Time ({})'.format(params['plot_units_time']))
    ax.set_ylabel(stat_type)
    sns.despine()
    plt.show()

    if stat_type == 'Num. of Branches':
        key_word = 'stat_b'
    elif stat_type == 'Num. of Tips':
        key_word = 'stat_t'
    elif stat_type == 'Branching Density':
        key_word = 'stat_d'
    elif stat_type == 'Radii of Mycelia ({})'.format(params['plot_units_space']):
        key_word = 'stat_r'

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_{}_{}.png".format(param_string,
                                                      run,
                                                      param_string,
                                                      key_word,
                                                      run)
    fig.savefig(fig_name)
    plt.close()

# ----------------------------------------------------------------------------

def plot_avg_treha_annulus(count_stat,max_count_stat, stat_type, folder_string, param_string, current_time, params, run):

    fig, ax = plt.subplots()
    xlabel = range(30,30*len(count_stat),30)
    # breakpoint()
    ax.plot(xlabel,count_stat[1:]/max_count_stat)
    # ax.set_title(stat_type)
    # ax.set_xlabel('Time ({})'.format(params['plot_units_time']))
    ax.set_ylabel(stat_type)
    sns.despine()
    #plt.show()

    if stat_type == 'Num. of Branches':
        key_word = 'stat_b'
    elif stat_type == 'Num. of Tips':
        key_word = 'stat_t'
    elif stat_type == 'Branching Density':
        key_word = 'stat_d'
    elif stat_type == 'Radii of Mycelia ({})'.format(params['plot_units_space']):
        key_word = 'stat_r'

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_{}_{}_avgTrehaAnnulus.png".format(param_string,
                                                      run,
                                                      param_string,
                                                      current_time,
                                                      run)
    fig.savefig(fig_name)
    plt.close()
    
def plot_max_treha_annulus(count_stat,max_count_stat, stat_type, folder_string, param_string, current_time, params, run):

    fig, ax = plt.subplots()
    xlabel = range(30,30*len(count_stat),30)
    # breakpoint()
    ax.plot(xlabel,count_stat[1:]/max_count_stat)
    # ax.set_title(stat_type)
    # ax.set_xlabel('Time ({})'.format(params['plot_units_time']))
    ax.set_ylabel(stat_type)
    sns.despine()
    #plt.show()

    if stat_type == 'Num. of Branches':
        key_word = 'stat_b'
    elif stat_type == 'Num. of Tips':
        key_word = 'stat_t'
    elif stat_type == 'Branching Density':
        key_word = 'stat_d'
    elif stat_type == 'Radii of Mycelia ({})'.format(params['plot_units_space']):
        key_word = 'stat_r'

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_{}_{}_maxTrehaAnnulus.png".format(param_string,
                                                      run,
                                                      param_string,
                                                      current_time,
                                                      run)
    fig.savefig(fig_name)
    plt.close()

def plot_min_treha_annulus(count_stat,max_count_stat, stat_type, folder_string, param_string, current_time, params, run):

    fig, ax = plt.subplots()
    xlabel = range(30,30*len(count_stat),30)
    # breakpoint()
    ax.plot(xlabel,count_stat[1:]/max_count_stat)
    # ax.set_title(stat_type)
    # ax.set_xlabel('Time ({})'.format(params['plot_units_time']))
    ax.set_ylabel(stat_type)
    sns.despine()
    #plt.show()

    if stat_type == 'Num. of Branches':
        key_word = 'stat_b'
    elif stat_type == 'Num. of Tips':
        key_word = 'stat_t'
    elif stat_type == 'Branching Density':
        key_word = 'stat_d'
    elif stat_type == 'Radii of Mycelia ({})'.format(params['plot_units_space']):
        key_word = 'stat_r'

    # Save the plot
    fig_name = "Results/{}/Run{}/{}_{}_{}_minTrehaAnnulus.png".format(param_string,
                                                      run,
                                                      param_string,
                                                      current_time,
                                                      run)
    fig.savefig(fig_name)
    plt.close()
    
# ----------------------------------------------------------------------------

def plot_errorbar_stat(count_times, avg_stat, std_stat, stat_type, folder_string, param_string, params, num_runs):

    # Convert units
    if params['plot_units_time'] == 'days':
        plot_times = count_times
    elif params['plot_units_time'] == 'hours':
        plot_times = 24*count_times
    elif params['plot_units_time'] == 'minutes':
        plot_times = 60*24*count_times
    elif params['plot_units_time'] == 'seconds':
        plot_times = 60*60*24*count_times

    fig, ax = plt.subplots()
    ax.errorbar(plot_times, avg_stat, std_stat)
    # ax.set_title(stat_type)
    ax.set_xlabel('Time ({})'.format(params['plot_units_time']))
    ax.set_ylabel(stat_type)
    sns.despine()
    #plt.show()

    if stat_type == 'Avg. Num. of Branches ({} Iterations)'.format(num_runs):
        key_word = 'avg_b'
    elif stat_type == 'Avg. Num. of Tips ({} Iterations)'.format(num_runs):
        key_word = 'avg_t'
    elif stat_type == 'Avg. Branching Density ({} Iterations)'.format(num_runs):
        key_word = 'avg_d'
    elif stat_type == 'Avg. Radii in {} ({} Iterations)'.format(params['plot_units_space'], num_runs):
        key_word = 'avg_r'

    # Save the plot
    fig_name = "Results/{}/Avg{}/{}_{}_avg{}.png".format(param_string,
                                                         num_runs,
                                                         param_string,
                                                         key_word,
                                                         num_runs)
    fig.savefig(fig_name)
    plt.close()

# ----------------------------------------------------------------------------

def plot_biomassdensity(radius_i, biomass_density, curr_time):
    """
    Parameters
    ----------
    radius_i : list
        List of smaller annuli radii value in which density is commputed.
    biomass_density : list
        Density of hyphae segment in an annulus with inner radii corresponding to radius_i.
    curr_time : double
        The current time of simulation, in days.

    Returns
    -------
    None.

    Purpose
    -------
    Plot biomass density at different distances from center of colony.
    For an annulus with inner radius r1 and outer radius r2,
        biomass density = (num. of segments in annulus) / (pi*(r2^2-r1^2))

    """
    fig1, ax1 = plt.subplots()
    ax1.plot(radius_i[1:len(radius_i)], biomass_density[1:len(radius_i)], marker='o')
    ax1.set_title('Biomass Density \nTime = {:.1f} Days'.format(curr_time), fontweight='bold')
    ax1.set_xlabel('Distance To Center (mm)')
    ax1.set_ylabel('Biomass Density')
    sns.despine()
    plt.show()


def plot_tipdensity(radius_i, tip_density, curr_time):
    """
    Parameters
    ----------
    radius_i : list
        List of smaller annuli radii value in which density is commputed.
    tip_density : list
        Density of hyphae tips in an annulus with inner radii corresponding to radius_i.
    curr_time : double
        The current time of simulation, in days.
    scale_val : double
        Parameter descrbing units used, scale_val=1 for mm or scale_val=1000 for µm.

    Returns
    -------
    None.

    Purpose
    -------
    Plot hyphal tip density at different distances from center of colony.
    For an annulus with inner radius r1 and outer radius r2,
        tip density = (num. of tips in annulus) / (pi*(r2^2-r1^2))

    """
    fig2, ax2 = plt.subplots()
    ax2.plot(radius_i[1:len(radius_i)], tip_density[1:len(radius_i)], marker='o')
    ax2.set_title('Hyphal Tip Density \nTime = {:.1f} Days'.format(curr_time), fontweight='bold')
    ax2.set_xlabel('Distance To Center (mm)')
    ax2.set_ylabel('Hyphal Tip Density')
    sns.despine()
    plt.show()

##############################################################################

def plot_hist(mycelia, curr_time,num_total_segs, param_string, params, run):
    
    if params['plot_units_time'] == 'days':
        plot_time = curr_time / (60*60*24)
    elif params['plot_units_time'] == 'hours':
        plot_time = curr_time / (60*60)
    elif params['plot_units_time'] == 'minutes':
        plot_time = curr_time / 60
    elif params['plot_units_time'] == 'seconds':
        plot_time = curr_time
    fig, ax = plt.subplots()
    # breakpoint()
    ax.hist(mycelia['dist_from_center'][:num_total_segs], range=[0, 2000], bins=100)
    fig_name = "Results/{}/Run{}/{}_{}_{}.png".format(param_string,
                                                      run,
                                                      curr_time,
                                                      param_string,                                                    
                                                      run)
    fig.savefig(fig_name)
    plt.close()
    
##############################################################################

def plot_density_annulus(density_per_unit_annulus, num_total_segs, param_string, params, run):
    
    fig, ax = plt.subplots()
    ax.plot(range(2000), density_per_unit_annulus)
    fig_name = "Results/{}/Run{}/{}_{}_density.png".format(param_string,
                                                      run,
                                                      param_string,                                                    
                                                      run)
    fig.savefig(fig_name)
    plt.close()
    
##############################################################################

def plot_treha_conc_annulus(avg_treha_annulus, num_total_segs, param_string, current_time, params, run):
    
    fig, ax = plt.subplots()
    ax.plot(range(2000), avg_treha_annulus)
    fig_name = "Results/{}/Run{}/{}_{}_{}_treha_conc.png".format(param_string,
                                                      run,
                                                      param_string,
                                                      current_time,
                                                      run)
    fig.savefig(fig_name)
    plt.close()
