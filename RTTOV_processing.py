#!/usr/bin/env python
# coding: utf-8

# In[2]:


import srfm
import srfm.main
import inspect
import importlib.metadata
import numpy as np
import matplotlib.pyplot as plt
"""If running the SRFM with a driver table, run this code."""
from srfm import *
import xarray as xr
from scipy.interpolate import interp1d
import os
import pandas as pd
import iris
import iris.quickplot as qplt
from scipy.optimize import least_squares
from scipy.constants import h, c, k


# In[ ]:


"""Here is the Planck function converting radiance to BT for given central wavenumber"""

c1 = 1.191042e-5 
c2 = 1.4387752 

def radiance_wn_cm_(T, wn):
    """
    Planck radiance for:
      T  : temperature (K)
      wn : wavenumber (cm^-1)

       Returns:
        mW m-2 sr-1 (cm-1)-1
    """
    wn_m = wn * 100.0

    # Planck rasurf = np.loadtxt('/network/group/aopp/eodg/RGG008_GRAINGER_IASIVOLC/knizek/MetOffice_intercomparison_dataset/surface_variables.datdiance per metre wavenumber
    B_m = (
        2 * h * c**2 * wn_m**3 /
        (np.exp((h * c * wn_m) / (k * T)) - 1)
    )

    # Convert:
    # W m^-2 sr^-1 m^-1 -> mW m^-2 sr^-1 (cm^-1)^-1
    B_cm = B_m *10**5

    return B_cm

"""Here is the band correction function which converts from BT to radiance using corrected BT. 
It returns radiance in mW m-2 sr-1 (cm-1)-1. """

#Band_Correction Function (using offset and scale factor coefficients)
def correction(a,b, BT, wn):

    corr_BT = a+(b*BT)
    rad = radiance_wn_cm_(corr_BT, wn)

    return rad


# ## Opening RTTOV datasets

# In[ ]:


#Converting clear sky scenario brightness temperatures for channel 7
cubes_0_ = iris.load("/home/m/moulin-garrigues/RTTOV/Full_data_7_08/combined_file_0.5_80_312.53_1_numden3.nc")
print(cubes_0_)
clear_rad_7_= cubes_0_[8].data
clear_BT_7_= cubes_0_[9].data

#Apply Planck Function
nu_7 =0.1147372612e+04
# Convert masked array to NumPy array
bt_7= np.asarray(clear_BT_7_.filled(np.nan), dtype=float)
    
a_7=0.8439648226E-01
b_7=0.9995983065E+00
trial_7 = correction(a_7,b_7, bt_7, nu_7)
print(trial_7)

sim_rad_7_ = cubes_0_[8].data
print(sim_rad_7_)

L_true = np.asarray(sim_rad_7_.filled(np.nan), dtype=float)

mask = np.isfinite(L_true) & np.isfinite(trial_7)

diff = trial_7[mask] - L_true[mask]

print("Mean bias:", diff.mean())
print("RMSE:", np.sqrt(np.mean(diff**2)))
print("Maximum abs difference:", np.max(np.abs(diff)))


# In[20]:


cubes_01 = iris.load("/home/m/moulin-garrigues/RTTOV/Full_data_3_08/combined_file_0.5_70_312.53_1_nobias.nc")
print(cubes_01)


# In[40]:


cubes_num = iris.load("/home/m/moulin-garrigues/RTTOV/Full_data_7_08/combined_file_0.5_80_339.53_1_numden3.nc")
print(cubes_num)

sim_bt_7= cubes_num[3].data
info_sim_bt_7 = cubes_num[3]
print(info_sim_bt_7)
print(sim_bt_7)

sim_bt_9= cubes_num[4].data
info_sim_bt_9 = cubes_num[4]
print(info_sim_bt_9)
print(sim_bt_9)



sim_bt_10= cubes_num[6].data
info_sim_bt_10 = cubes_num[6]
print(info_sim_bt_10)
print(sim_bt_10)


sim_bt_11= cubes_num[2].data
info_sim_bt_11 = cubes_num[2]
print(info_sim_bt_11)
print(sim_bt_11)


# ## 1. Converting Brightness Temperature to Radiance for RTTOV numdem3 ash datafiles

# In[ ]:


'''This cell converts RTTOV BT to radiance (radiance in mW) and inputs the computed RTTOV radiances in a csv file'''

def correction(a,b, BT, wn):

    corr_BT = a+(b*BT)
    rad = radiance_wn_cm_(corr_BT, wn)

    return rad
file= "combined_file_50.0_80_339.53_1_numden3"
input_file = f"/home/m/moulin-garrigues/RTTOV/Full_data_7_08/{file}.nc"
cubes_01 = iris.load(input_file)
info_sim_bt_7 = cubes_01[3] #metadata for 

info_sim_bt_9 = cubes_01[4]


info_sim_bt_10 = cubes_01[6]

info_sim_bt_11 = cubes_01[2]

# Extract cubes - extract brightness temperatures for each channel
BT_channels = {
    7: cubes_01[3].data, #channel 7
    9: cubes_01[4].data, #channel 9
    10: cubes_01[6].data, #channel 10
    11: cubes_01[2].data #channel 11
}

# Channel constants from RTTOV - central wavenumber/offset coefficient/scale factor coefficient for channels 7;9;10;11
channel_params = {
    7: {                        #channel 7
        "nu": 0.1147372612e+04,
        "a": 0.8439648226e-01,
        "b": 0.9995983065e+00
    },
    9: {                        #channel 9
        "nu": 0.9270688810e+03,
        "a": 0.1448128235e+00,
        "b": 0.9988828246e+00
    },
    10: {                       #channel 10
        "nu": 0.8368734588e+03,
        "a": 0.3209138358e-01,
        "b": 0.9994537860e+00
    },
    11: {                       #channel 11
        "nu": 0.7481686747e+03,
        "a": -0.5612228281e-01,
        "b": 0.9995570938e+00
    }
}
# Extract Coordinates from eg channel 10 dataset (same set of coordinates for all channels)
lat = cubes_01[10].coord('latitude').points
lon = cubes_01[10].coord('longitude').points

# Take time = 0
lon2d, lat2d = np.meshgrid(lon, lat)

# Start dataframe with coordinates
df = pd.DataFrame({
    "latitude": lat2d.ravel(),
    "longitude": lon2d.ravel()
})


for ch in [7, 9, 10, 11]:

    # brightness temperature
    Temp = np.asarray(
        BT_channels[ch][0].filled(np.nan),
        dtype=float
    )

    # get RTTOV constants for this channel
    nu = channel_params[ch]["nu"]
    a  = channel_params[ch]["a"]
    b  = channel_params[ch]["b"]

    Radiance = correction(a,b,Temp,nu)

    # add to dataframe
    df[f"BT_{ch}"] = Temp.ravel()
    df[f"Radiance_{ch}"] = Radiance.ravel()
    
# Save
df.to_csv(
    f"/home/m/moulin-garrigues/RTTOV/Full_Rad_3_08/Rad_{file}.csv", #Name of output file where the computed RTTOV radiances are stored
    index=False
)
with open(f"/home/m/moulin-garrigues/RTTOV/Full_Rad_3_08/Info_{file}.txt", "w") as f:
    f.write(f"Channel 7\n{cubes_01[3]}\n\n")
    f.write(f"Channel 9\n{cubes_01[4]}\n\n")
    f.write(f"Channel 10\n{cubes_01[6]}\n\n")
    f.write(f"Channel 11\n{cubes_01[2]}\n")
print(df.head())
print(df.shape)


# In[ ]:





# ## 2. Inputing RTTOV and SRFM radiances for various ash properties combinations into a single file per profile - Loop Over All Profiles 

# ### Looping over all profile (same as above cell only loop over all profiles)

# In[ ]:


"""This cell takes the converted RTTOV radiances from part 1. above, inputs them into a file with the corresponding SRFM radiances for corresponding ash property combination.
The cell loops over all profiles, and outputs one file for each profile. The output file contains for given profile all ash properties combinations, corresponding RTTOV radiance and SRFM radiance for each combination."""

# Loop over all profiles
rttov_file = "/home/m/moulin-garrigues/RTTOV/Full_Rad_3_08/"

#These are the first lat/long in RTTOV data corresponding to the 10 profiles in the SRFM MetOffice Intercomparison dataset
profile_locations = {
    "0": (50.078125, -29.882812),
    "1": (50.078125, -29.648438),
    "2": (50.078125, -29.414062),
    "3": (50.078125, -29.179688),
    "4": (50.078125, -28.945312),
    "5": (50.078125, -28.710938),
    "6": (50.078125, -28.476562),
    "7": (50.078125, -28.242188),
    "8": (50.078125, -28.007812),
    "9": (50.078125, -27.773438)
}

simulations = [
    {
        "file": "Rad_combined_file_0.5_70_312.53_1_numden3.csv",
        "mass_loading": 0.5,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },
    {
        "file": "Rad_combined_file_7.0_70_312.53_1_numden3.csv",
        "mass_loading": 7.0,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },
    {
        "file": "Rad_combined_file_20.0_70_312.53_1_numden3.csv",
        "mass_loading": 20.0,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },

    {
        "file": "Rad_combined_file_50.0_70_312.53_1_numden3.csv",
        "mass_loading": 50.0,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },

    {
        "file": "Rad_combined_file_0.5_80_339.53_1_numden3.csv",
        "mass_loading": 0.5,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    },
    {
        "file": "Rad_combined_file_7.0_80_339.53_1_numden3.csv",
        "mass_loading": 7.0,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    },
    {
        "file": "Rad_combined_file_20.0_80_339.53_1_numden3.csv",
        "mass_loading": 20.0,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    },

    {
        "file": "Rad_combined_file_50.0_80_339.53_1_numden3.csv",
        "mass_loading": 50.0,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    }]


for profile, (target_lat, target_lon) in profile_locations.items():

    print(f"Processing profile {profile}")

    # Load matching SRFM properties
    srfm_file = f"/home/m/moulin-garrigues/SRFM_RTTOV/Mes_resultats/Combined_prop_rad/Layer_thickness/matches_{profile}_thickness_312_339.csv"

    srfm = pd.read_csv(srfm_file)

    for sim in simulations:

        # Load RTTOV simulation
        rttov = pd.read_csv(rttov_file + sim["file"])

        # Add metadata
        rttov["mass_loading"] = sim["mass_loading"]
        rttov["r"] = sim["r"]
        rttov["thickness"] = sim["thickness"]
        rttov["alt_upp"] = sim["alt_upp"]


        # Select RTTOV pixel for this profile
        rttov_pixel = rttov[
            np.isclose(rttov["latitude"], target_lat) &
            np.isclose(rttov["longitude"], target_lon)
        ]


        if rttov_pixel.empty:
            print(f"No RTTOV pixel found for profile {profile}")
            continue


        rttov_pixel = rttov_pixel.iloc[0]


        # Match SRFM properties
        mask = (
            np.isclose(srfm["mass_loading"], sim["mass_loading"]) &
            np.isclose(srfm["r"], sim["r"]) &
            np.isclose(srfm["thickness"], sim["thickness"]) &
            np.isclose(srfm["alt_upp"], sim["alt_upp"])
        )


        # Insert RTTOV radiances
        srfm.loc[mask, "RTTOV_ch07"] = rttov_pixel["Radiance_7"]
        srfm.loc[mask, "RTTOV_ch09"] = rttov_pixel["Radiance_9"]
        srfm.loc[mask, "RTTOV_ch10"] = rttov_pixel["Radiance_10"]
        srfm.loc[mask, "RTTOV_ch11"] = rttov_pixel["Radiance_11"]


        # Store RTTOV location
        srfm.loc[mask, "RTTOV_latitude"] = rttov_pixel["latitude"]
        srfm.loc[mask, "RTTOV_longitude"] = rttov_pixel["longitude"]


    # Save in output file
    output = (f"/home/m/moulin-garrigues/RTTOV/Full_Rad_3_08/Merge_Ash_nobias/Merge_{profile}_nobias_numdem23.csv")

    srfm.to_csv(output, index=False)

    print(f"Saved profile {profile}: {output}")


# ### (same as above only without the loop) for a given profile

# In[ ]:


prof = 1
#Loop over all files for a given atm profile
rttov_file = "/home/m/moulin-garrigues/RTTOV/Full_Rad_3_08/"
srfm_file = f"/home/m/moulin-garrigues/SRFM_RTTOV/Mes_resultats/Combined_prop_rad/Layer_thickness/matches_{prof}_thickness_312_339.csv"

# Read file
srfm = pd.read_csv(srfm_file)

# Selected RTTOV pixel
target_lat = 50.078125  
target_lon = -29.882812  #Profile 0


simulations = [
    {
        "file": "Rad_combined_file_0.5_70_312.53_1_nobias.csv",
        "mass_loading": 0.5,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },
    {
        "file": "Rad_combined_file_7.0_70_312.53_1_nobias.csv",
        "mass_loading": 7.0,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },
    {
        "file": "Rad_combined_file_20.0_70_312.53_1_nobias.csv",
        "mass_loading": 20.0,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },

    {
        "file": "Rad_combined_file_50.0_70_312.53_1_nobias.csv",
        "mass_loading": 50.0,
        "r": 0.62,
        "thickness": 0.312,
        "alt_upp": 13.727,
    },

    {
        "file": "Rad_combined_file_0.5_80_339.53_1_nobias.csv",
        "mass_loading": 0.5,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    },
    {
        "file": "Rad_combined_file_7.0_80_339.53_1_nobias.csv",
        "mass_loading": 7.0,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    },
    {
        "file": "Rad_combined_file_20.0_80_339.53_1_nobias.csv",
        "mass_loading": 20.0,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    },

    {
        "file": "Rad_combined_file_50.0_80_339.53_1_nobias.csv",
        "mass_loading": 50.0,
        "r": 0.62,
        "thickness": 0.339,
        "alt_upp": 16.999,
    }]

for sim in simulations:

    rttov = pd.read_csv(rttov_file + sim["file"])

    rttov["mass_loading"] = sim["mass_loading"]
    rttov["r"] = sim["r"]
    rttov["thickness"] = sim["thickness"]
    rttov["alt_upp"] = sim["alt_upp"]

    rttov_pixel = rttov[
        np.isclose(rttov["latitude"], target_lat) &
        np.isclose(rttov["longitude"], target_lon)
    ]

    mask = (
        np.isclose(srfm["mass_loading"], sim["mass_loading"]) &
        np.isclose(srfm["r"], sim["r"]) &
        np.isclose(srfm["thickness"], sim["thickness"]) &
        np.isclose(srfm["alt_upp"], sim["alt_upp"])
    )

    srfm.loc[mask, "RTTOV_ch07"] = rttov_pixel.iloc[0]["Radiance_7"]
    srfm.loc[mask, "RTTOV_ch09"] = rttov_pixel.iloc[0]["Radiance_9"]
    srfm.loc[mask, "RTTOV_ch10"] = rttov_pixel.iloc[0]["Radiance_10"]
    srfm.loc[mask, "RTTOV_ch11"] = rttov_pixel.iloc[0]["Radiance_11"]

    
    # Store the latitude/longitude of the selected RTTOV pixel
    srfm.loc[mask, "RTTOV_latitude"] = rttov_pixel.iloc[0]["latitude"]
    srfm.loc[mask, "RTTOV_longitude"] = rttov_pixel.iloc[0]["longitude"]
    
    # Check the result
    print(srfm.loc[mask, [
        "mass_loading", "r", "thickness", "alt_upp",
        "ch07", "RTTOV_ch07",
        "ch09", "RTTOV_ch09",
        "ch10", "RTTOV_ch10",
        "ch11", "RTTOV_ch11"
    ]])
    
    # Save
    srfm.to_csv(
        f"/home/m/moulin-garrigues/RTTOV/Full_Rad_3_08/Merge_Ash_nobias/singleMerge_{prof}_nobias.csv",
        index=False
    )


# In[ ]:




