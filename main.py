import scipy.io as sio
import scipy.interpolate as interp
import scipy.optimize as sciop
import numpy as np
from Ray_Pile import ray_pile
import RayPileGraphics as RPGH
import ode_func as odef
c_const = 3e8
e_charge = 1.60e-19
e_mass = 9.11e-31
e0 = 8.85e-12
wp_const_mks = (e_charge**2)/(e_mass*e0) #
kB = 1.38e-23

def return_const(w_length):
    wavelength = w_length#*1e2
    wave_num = 2*np.pi/wavelength
    ang_freq = wave_num*c_const
    wp_const = (1.60e-19**2)/(8.85e-12*9.11e-31)
    n_c_val = (ang_freq**2)/wp_const
    return [wave_num,ang_freq,n_c_val]

laser_o = 0.351E-6
k_init,omega_laser,n_c = return_const(laser_o)


def ne_data_calc(coords_init,t):
    """
    interpolates values of electron density and partial derivatives
    coords_init: [x,y,z]
    t: time
    """
    x = coords_init[0]
    y = coords_init[1]
    z = coords_init[2]
    #should never bend away as the highest is 0.4 critical density
    slope = n_c*(0.4 - 0.1)/(200E-6)
    ne_val = slope*x + 0.1*n_c
    if ne_val >= 0:
        nedx_val = slope
        nedy_val = 0
        nedz_val = 0
        nedt_val = 0
    else:
        return [0,0,0,0,0]
    return [ne_val,nedx_val,nedy_val,nedz_val,nedt_val]

def te_data_calc(coords_init,t):
    """
    interpolates values of electron temperature and partial derivatives
    coords_init: [x,y,z]
    t: time
    """
    x = coords_init[0]
    y = coords_init[1]
    z = coords_init[2]

    freq = 2*np.pi/1e-8
    ne_val = 2e3*1.60e-19 #+ 5e9*np.cos(freq*y)
    nedx_val = 0
    nedy_val = 0#-freq*5e19*np.sin(freq*y)
    nedz_val = 0
    nedt_val = 0

    return [ne_val,nedx_val,nedy_val,nedz_val,nedt_val]


def branch_flagging(pileName,rayNumbers):
    origin = 0
    branched = False
    for rayNum in rayNumbers:
        for ind,i in enumerate(pileName.ray_absolute_path[rayNum]):
            if i%10e-6 < 8e-7 and i < 100e-6 and i > 25e-6:
                pileName.ray_flags[rayNum][ind] = 1
                branched = True
                
def branching_process(pileName,rayNumbers):
    for rayNum in rayNumbers:
        for ind,i in enumerate(pileName.ray_flags[rayNum]):
            if i==1:
                time_array = pileName.ray_time[rayNum][ind:]
                pos_array = pileName.ray_position[rayNum][:,ind]
                k_array = pileName.ray_wave_vector[rayNum][:,ind]
                n_now = ne_data_calc(pos_array,time_array[0])[0]
                t_now = te_data_calc(pos_array,time_array[0])[0]
                [wave_vec1,wave_vec2] = odef.plasmon_gen_k(k_array,pos_array,n_c,n_now,t_now,time_array[0])
                #plotPlasmaK_paper(k_array,pos_array,n_c,timeArray[0])
                pileName.create_plasma_ray(time_array,pos_array,wave_vec1,orig = rayNum)
                pileName.propagate_ray(len(pileName.ray_wave_vector)-1)
                pileName.create_plasma_ray(time_array,pos_array,wave_vec2,orig = rayNum)
                pileName.propagate_ray(len(pileName.ray_wave_vector)-1)




pile1 = ray_pile(ne_data_calc,te_data_calc)
timeArray = np.linspace(0,2e-11,5000)
locationArray = np.array([0,0,0])

k_array = np.array([k_init,0,0])
#k_plasma = [np.sqrt(initPlasmaK(omega_laser,locationArray,n_c,timeArray[0])),0,0]
polarization = np.array([0,1,0])
pile1.create_light_ray(timeArray,locationArray,k_array,polarization)
#pile1.create_plasma_ray(timeArray,locationArray,k_plasma)
pile1.propagate_ray(0)
#pile1.propagate_ray(1)
branch_flagging(pile1,[0])
branching_process(pile1,[0])
RPGH.graphTraj(pile1,list(range(len(pile1.ray_time))))
#RPGH.graphCompTraj(pile1,[0])
len_list = []
for i in pile1.ray_time:
    len_list.append(len(i))

for i in range(1,len(pile1.ray_time)):
    pile1.landau_dampen(i)

RPGH.graphTraj(pile1,list(range(len(pile1.ray_time))))

for i,listind in enumerate(pile1.ray_time[1:]):
    len_list[i] -= len(listind)
print(len_list)
