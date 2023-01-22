

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.components.overland_flow import OverlandFlow
from landlab import RasterModelGrid
from landlab.components import SoilInfiltrationGreenAmpt
import time
import copy
import sys

# ks_vals_west = 1*10**-6, 3*10**-6, ## m s-1
# ks_vals_east = 9*10**-7, 3*10**-6,
# criticalStresses =[0.06,0.07]
# manning = [0.04, 0.08]

## Constants
densitySSL = 2500       # sediment density [kg / m^3]
densitywater = 1200     # water density [kg / m^3]
gravity = 9.82          # gravity constant [m / s^2]
criticalStress = 0.06   # critical shear stress [-]
mannings_n = 0.04       # Manning's roughness coeffcient [-]
ks = 1*10**-6           # Hydarulic conductivties [m / s]

## Rainfall vector  in mm/h for 30-minute storm.   100 YEAR return period
#starting_precip_mmhr_east =np.concatenate((np.ones((1,10))*47,np.ones((1,10))*100,np.ones((1,10))*19),axis=1).flatten()
starting_precip_mmhr_west = np.concatenate((np.ones((1,10))*30,np.ones((1,10))*76,np.ones((1,10))*17),axis=1).flatten()

## Convert mm/h to m/s
starting_precip_ms = starting_precip_mmhr_west * (2.777778 * 10 ** -7)  ## to m/s

## Load topographic data
profile = pd.read_csv(sys.path[0] + '\west.csv', header=None)
#profile = pd.read_csv(sys.path[0]+ '\east.csv', header=None)

## Create model grid
ncols = 3
mg = RasterModelGrid((profile.shape[0], ncols), 1.)
z = mg.add_zeros('node', 'topographic__elevation')
topo_rast = mg.node_vector_to_raster(z)
topo_rast[:] = np.tile(np.array(profile.loc[:,1].transpose()),(ncols,1)).T
z[:] = topo_rast.flatten()
initial_topo = copy.deepcopy(z)
slope = mg.calc_slope_at_node()
mg.at_node['topographic__slope'] = slope

## Boundry conditions
mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
mg.set_fixed_value_boundaries_at_grid_edges(False, True, False, False)

## Create field for model components
mg.add_zeros("surface_water__depth", at="node")
mg.add_zeros("surface_water__discharge", at="node")
mg.add_zeros("water_surface__gradient", at="node")
mg.add_zeros("shear_stress", at="node")

## Initiate components
of = OverlandFlow(mg, steep_slopes=True, mannings_n=mannings_n)
hydraulic_conductivity = mg.ones('node') * ks  ## ks is in units of [m/s]
d = mg.add_ones("soil_water_infiltration__depth", at="node", dtype=float)
d *= 0.03  ## meter
SI = SoilInfiltrationGreenAmpt(mg, hydraulic_conductivity=hydraulic_conductivity,
                               soil_type='silty clay',
                               initial_soil_moisture_content=0.05,
                               volume_fraction_coarse_fragments=0.1,
                               )

dt = 1  ## in sec# Setting the adaptive time step
storm_duration = dt * np.size(starting_precip_ms) * 60
model_run_time = storm_duration + (10*60)  # total simulation duration [sec]. Storm duartion + 10 minutes
hydrograph_time = []
discharge_at_outlet = []
waterdepth_at_outlet = []
RI = starting_precip_ms
rainfall_vec = []
shear_stress_outlet = []
sediment_flux_outlet = []
max_stress = np.zeros_like(mg.at_node['surface_water__depth'])
max_depth = np.zeros_like(mg.at_node['surface_water__depth'])
outlet_node = (83*3)+1
run_num = 1
cnti = 0
start_time = time.time()
elapsed_time = 1.0  # s # For each run

while elapsed_time < model_run_time:  # each run
    if elapsed_time < (
    storm_duration):  ## This is rain or not rain for one MINUTE so the next loop run within this minute
        RI = starting_precip_ms[cnti]
    else:
        RI = 0

    minutes_cnt = 0
    of._rainfall_intensity = RI
    while minutes_cnt <= 60:  ## run within one minute in steps of 1 s - this may change internally
        SI.run_one_step(dt=dt)
        of.run_one_step(dt=dt)

        q = mg.at_link['surface_water__discharge']
        q_nodes = of.discharge_mapper(q, convert_to_volume=True)
        mg['node']['surface_water__discharge'] = of.discharge_mapper(q, convert_to_volume=True)
        h = mg.at_node['surface_water__depth']
        mg['node']['water_surface__slope'] = (
                    of._water_surface_slope[mg.links_at_node] * mg.active_link_dirs_at_node).max(axis=1)
        mg.at_node['shear_stress'] = mg['node']['water_surface__slope'] * mg.at_node[
            'surface_water__depth'] * gravity * densitywater  # Kg m-1 s-2 or 1 N m-2
        shear_stress = mg.at_node['shear_stress']

        ## SAVE THE MAXIMAL SHEAR STRESS PER PIXEL
        max_stress[max_stress < shear_stress] = shear_stress[max_stress < shear_stress]
        max_depth[max_depth < h] = h[max_depth < h]

        hydrograph_time.append(elapsed_time + minutes_cnt)
        discharge_at_outlet.append(np.abs(q_nodes[outlet_node]))
        waterdepth_at_outlet.append(mg.at_node['surface_water__depth'][outlet_node])
        shear_stress_outlet.append(shear_stress[outlet_node])
        rainfall_vec.append(RI)

        # Counter - watch
        minutes_cnt += dt

        ## Updating elapsed_time
    elapsed_time += minutes_cnt
    cnti += 1



critical_gs = max_stress / ((densitySSL - densitywater) * gravity * criticalStress)
