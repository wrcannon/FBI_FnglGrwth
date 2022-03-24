Newest files are no longer located in the /WithTrehalose subfolder but are instead directly under /Bill1 branch (2/16/2022)

# Fungal Structure Code

## Before attempting to run the code:
- Go to the line "cwd_path ='/Users/libra/FBI_project/fungalGrowthModel_singleNutrient_py/0521'" in the driver_fungalGrowth_singleNutrient.py and change the path to where you  place the folder containing all files in this repository.
- Go to "def get_filepath(params):" in help_functions.py to change the name of folder string and file string.

## To Run the code:
- **Option 1:** Just run the driver_fungalGrowth_singleNutrient.py file 
    - The number of iterations of runs is determined by the variable ‘num_runs’ (set around line 269 - later will be an input if this file changes to a function)
    - If num_runs==1, then it will run just one iteration of a fungal mycelia growth
    - If num_runs>1, then it will run multiple iterations using the same set of parameters in parallel using the python Parallel function from the joblib package.
- **Option 2:** run the function (from within the py file listed in Option 1) using the command ‘driver_singleNutrient(1)’ to run one iteration of fungal mycelia growth.

## About the various files:
- **driver_fungalGrowth_singleNutrient.py:** contains the main driver file that executes the steps of diffusion in the external domain, elongation, branching, fusion, translocation, and uptake.
- **parameters.ini:** where all the parameters values are stored. If you want to change a parameter, it is most likely listed in this file.
- **helper_functions.py:** contains functions that do the following
    - convert parameters from the ini file to a usable form 
    - generates filenames used for saving data specific to simulation with given parameters
    - plotting funtions (the fungal structure, the external domain, various output stats)
- **setup_functions.py:** contains functions that do the following
    - set up the dictionary for storing info related to the mycelia structure
    - sets up the initial fungal structure and it’s properties
    - sets up the external grid shape and amount of nutrient in each cell
- **growth_functions.py:** contains functions that pertain to
    - elongation of hyphae at the tip
    - branching 
    - anastomosis (fusion) of hyphae
- **nutrient_functions.py:** contains functions that pertain to
    - translocation update
    - uptake of nutrients
## Particulars for branch Bill1:
- The code is set-up to run and get the same results as the main branch. Run this first to convince yourself.
- To run the version with updated Michaelis-Menten kinetics, in each of grow_functions.py, nutrient_functions.py and driver_fungalGrowth_singleNutrient.py
set the variable use_original = 0.
- •	The Michaelis Menten parameters still need to be tweaked. Also, I attempted a new approach to choose whether a branch should be created or not. It also does not work correctly yet. 
- Out of the box, this code produces minimal branching; ideally, we want it to produce branching something like the original.

## Metabolism:
- 

## Possible To Do - Soonish:
- *Translocation*
    - Different velocity coefficient for cell wall materials at septa (should slow down when passing through a septa)
    - Convert velocity of cell wall materials from a constant to a function that takes into account that the number of vissicles of cell wall material being transported in myosin-like filaments can become saturated. The likely effect is that segments closer to the tip may have higher concentrations of cell wall material than those further from the tip.
- *Branching*
    - Change the function for calculating the probability of branching. An exponential probability function is now available, but not sure if the results are any different.

## Possible To Do - Future:
- Apical branching: We may want to model Neurospora or another fungi instead.
- *Growth Direction*
    - Instead of pulling angles from a distribution, possibly have it grow towards higher levels of oxygen
    - Implement negative autotrophism (hyphal avoidance)?
- *Aging*
    - Different behavior for older hyphae near the center of the colony vs younger near the periphery - direction, growth rate, metabolism, etc.
    - Varying diameters of hyphae - older hyphae thicker than younger
- 3D
