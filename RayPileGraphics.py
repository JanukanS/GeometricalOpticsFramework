import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
