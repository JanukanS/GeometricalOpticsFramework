import scipy.io as sio
import scipy.interpolate as interp
import scipy.optimize as sciop
import numpy as np
from Ray_Pile import ray_pile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def graphTraj(pileName,rayNumbers,showPolarization=False):
    #graphs trajectory in 3 dimensions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    col_list = ['r','g','y']
    plt.axvline(x=100e-6/1e-6,color='pink')
    plt.xlim(left=-66.66,right=200)
    for val_ind,val in enumerate(rayNumbers):
        if pileName.ray_id[val][1] == 0:
            ax.plot(pileName.ray_position[val][0,:]/1e-6,pileName.ray_position[val][1,:]/1e-6,col_list[val_ind],label = 'Light ray')
            if showPolarization:
                ax.plot(pileName.ray_position[val][0,:]+1e-3*pileName.ray_polarization[val][0,:],\
                        pileName.ray_position[val][1,:]+1e-3*pileName.ray_polarization[val][1,:],\
                        pileName.ray_position[val][2,:]+1e-3*pileName.ray_polarization[val][2,:],'b',label = 'Polarization')
        if pileName.ray_id[val][1] == 1:
            ax.plot(pileName.ray_position[val][0,:]/1e-6,pileName.ray_position[val][1,:]/1e-6,'g-',label = 'Plasma wave ray')

    ax.set_xlabel('X axis (microns)')
    ax.set_ylabel('Y axis (microns)')
    #ax.set_zlabel('Z axis')
    ax.set_title('Ray Trajectories')
    #plt.legend()
    #plt.ylim(top = 150,bottom=-)
    plt.show()

def graphCompTraj(pileName,rayNumbers):
    #display individual components of trajectory, most effective one ray at a time
    col_list = ['r','g','y']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for val_ind,val in enumerate(rayNumbers):
        ax.plot(pileName.ray_time[val],pileName.ray_position[val][0,:],col_list[val_ind],label = 'x')
        #ax.plot(pileName.ray_time[val],pileName.ray_wave_vector[val][1,:],col_list[val_ind],label = 'y')
        #ax.plot(pileName.ray_time[val],pileName.ray_wave_vector[val][2,:],col_list[val_ind],label = 'z')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance')
    ax.set_title('Component Trajectory')
    plt.legend()
    plt.show()

def branch_flagging(pileName,rayNumbers):
    origin = 0
    branched = False
    for rayNum in rayNumbers:
        for ind,i in enumerate(pileName.ray_absolute_path[rayNum]):
            if i%10e-6 < 8e-7 and i < 100e-6 and i > 25e-6:
                pileName.ray_flags[rayNum][ind] = 1
                branched = True



def plotPlasmaK_paper(k_light,coords,crit_val,t):
    crit_now = ne_data_calc(coords,t)[0]
    te_now = te_data_calc(coords,t)[0]
    k0 = np.sqrt(k_light[0]**2+k_light[1]**2+k_light[2]**2)
    [kappa_val,ldeb_val] = calcKappa(k0,te_now,crit_now,crit_val)
    kx_domain_circle = [(0.5*k0*ldeb_val-kappa_val)/ldeb_val,(0.5*k0*ldeb_val+kappa_val)/ldeb_val]
    kx_domain_growth1 = [kx_domain_circle[0],0]
    kx_domain_growth2 = [k0,kx_domain_circle[1]]
    kx_1 = np.linspace(kx_domain_circle[0],kx_domain_circle[1])
    ky_1a = np.copy(kx_1)
    ky_1b = np.copy(kx_1)
    for ind,val in enumerate(kx_1):
        ky_1a[ind] = eq_wavematch(val,k0,te_now,crit_now,crit_val)
        ky_1b[ind] = -eq_wavematch(val,k0,te_now,crit_now,crit_val)
    kx_2 = np.linspace(kx_domain_growth1[0],kx_domain_growth1[1])
    ky_2a = np.copy(kx_2)
    ky_2b = np.copy(kx_2)
    for ind,val in enumerate(kx_2):
        ky_2a[ind] = eq_tbd_growth(val,k0,te_now,crit_now)
        ky_2b[ind] = -eq_tbd_growth(val,k0,te_now,crit_now)
    kx_3 = np.linspace(kx_domain_growth2[0],kx_domain_growth2[1])
    ky_3a = np.copy(kx_3)
    ky_3b = np.copy(kx_3)
    for ind,val in enumerate(kx_3):
        ky_3a[ind] = eq_tbd_growth(val,k0,te_now,crit_now)
        ky_3b[ind] = -eq_tbd_growth(val,k0,te_now,crit_now)
    fig = plt.figure()
    k0_x = np.linspace(0,k0)
    k0_y = 0*k0_x #???

    [k1_calc,k2_calc] = plasmon_gen_k(k_light,coords,crit_val,ne_data_calc(coords,t)[0],te_data_calc(coords,t)[0],t)
    k1_x_calc = [0,k1_calc[0]]
    k1_y_calc = [0,k1_calc[1]]
    k2_x_calc = [0,k2_calc[0]]
    k2_y_calc = [0,k2_calc[1]]
    ax = fig.add_subplot(111)

    ax.plot(kx_1,ky_1a,'b',kx_1,ky_1b,'b')
    ax.plot(kx_2,ky_2a,'r',kx_2,ky_2b,'r')
    ax.plot(kx_3,ky_3a,'r',kx_3,ky_3b,'r')
    ax.plot(k0_x,k0_y,'g')
    ax.plot(k1_x_calc,k1_y_calc,k2_x_calc,k2_y_calc)
    ax.plot()
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_title('Circle from Paper')
    plt.legend()
    plt.show()


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
graphTraj(pile1,list(range(len(pile1.ray_time))))
#graphCompTraj(pile1,[0])
len_list = []
for i in pile1.ray_time:
    len_list.append(len(i))

for i in range(1,len(pile1.ray_time)):
    pile1.landau_dampen(i)

graphTraj(pile1,list(range(len(pile1.ray_time))))

for i,listind in enumerate(pile1.ray_time[1:]):
    len_list[i] -= len(listind)
print(len_list)
