import numpy as np
import scipy.optimize as sciop

class physical_value():
    def __init__(self,value,units):
        """
        defines a physical value
        value:float:defines the numerical value of the constant
        units:list/nparray of 7 floats:stores the dimensions of each SI unit
                                       in the format
                                       [Time(s),Length(m),Kilogram(kg),Temperature(K),Current(A),Amount(mol),Luminous Intensity(cd)]
        Incomplete as of August 2019
        """
        self.value = value
        self.units = units
    def scale_value(self,unit_scale):
        """
        changes to match new units
        unit_scale:list/nparray of 7 floats: how many of the new unit fit into old unit
                                             example, 100 cm in 1 m so unit_scale[1] = 1000
        """
        scale_factor = 1
        for unit_i in range(7):
            scale_factor *= unit_scale[unit_i]**self.units[unit_i]
        return self.value*scale_factor


ELECTRON_CHARGE = physical_value(1.60E-19,[1,0,0,0,1,0,0])
ELECTRON_MASS = physical_value(9.11E-31,[0,0,1,0,0,0,0])
VACUUM_PERMITTIVITY = physical_value(8.85E-12,[4,-3,-1,0,2,0,0])
LIGHT_SPEED = physical_value(3e8,[-1,1,0,0,0,0,0])
BOLTZMANN_CONSTANT = physical_value(1.38E-23,[-2,2,1,-1,0,0,0])

e_charge = ELECTRON_CHARGE.value
e_mass = ELECTRON_MASS.value
e0 = VACUUM_PERMITTIVITY.value
wp_const_mks = (e_charge**2)/(e_mass*e0) #
wp_const_cgs = (4*np.pi/9.11E-31)*(1.60E-19)**2 #(
thermal_coeff_mks = 3/9.11e-31#1.5*(1.60e-19)/(9.11E-31)
c_const_mks = 3E8
c_const_cgs = 3E10

def plasma_freq(ne):
    return np.sqrt(wp_const_mks*ne)


def dispersion_laser(k):
    return c_const_mks*k

def dispersion_light(k,ne):
    return np.sqrt((c_const_mks**2)*(k**2) + wp_const_mks*ne)
#dispersion_light_vec = np.vectorize(dispersion_light)

def k_light(w,ne):
    return np.sqrt(((w**2) - wp_const_mks*ne)/(c_const_mks**2))

def k_plasma(w,ne,te):
    return np.sqrt(((w**2) - wp_const_mks*ne)/(thermal_coeff_mks_te))

def dispersion_plasma(k,ne,te):
    if wp_const_mks*ne + thermal_coeff_mks*te*(k**2) <= 0:
        return 0
    else:
        return np.sqrt(wp_const_mks*ne + thermal_coeff_mks*te*(k**2))

def k_plasma(w,ne,te):
    return np.sqrt(((w**2) - wp_const_mks*ne)/(thermal_coeff_mks*te))

#set 1 odes are integrated over time
#set 2 odes are integrated over path

def ode_set1_time(t,y,conc_rat_data):
    '''
    the odes to solve, for GNU Scientific Library Wrapper (need to check for python callbacks)
    '''
    #physical coordinates
    x_coord = y[0]
    y_coord = y[1]
    z_coord = y[2]
    #wave vector
    kx = y[3]
    ky = y[4]
    kz = y[5]
    c_const = 3e8
    #omega value
    #concentration values
    conc = conc_rat_data[0]
    conc_dx = conc_rat_data[1]
    conc_dy = conc_rat_data[2]
    conc_dz = conc_rat_data[3]
    conc_dt = conc_rat_data[4]
    w = dispersion_light(np.sqrt(kx**2+ky**2+kz**2),conc)
    #print(w,,conc)
    #hamilton group velocity
    #print(conc*wp_const/(w**2),w)
    dxdt = (c_const**2)*kx/w
    dydt = (c_const**2)*ky/w
    dzdt = (c_const**2)*kz/w
    #hamilton dkdt
    dkxdt = -(wp_const_mks)*conc_dx/(2*w)#-(wp_const/w**2)*conc_dx/(2*disp)
    dkydt = -(wp_const_mks)*conc_dy/(2*w)#-(wp_const/w**2)*conc_dy/(2*disp)
    dkzdt = -(wp_const_mks)*conc_dz/(2*w)#-(wp_const/w**2)*conc_dz/(2*disp)
    #hamilton dwdt
    return [dxdt,dydt,dzdt,dkxdt,dkydt,dkzdt]

def ode_set1_s(s,y,conc_rat_data):

    #physical coordinates
    x_coord = y[0]
    y_coord = y[1]
    z_coord = y[2]
    #wave vector
    kx = y[3]
    ky = y[4]
    kz = y[5]
    w = y[6]

    #concentration values
    conc_rat = conc_rat_data[0]
    conc_dx = conc_rat_data[1]
    conc_dy = conc_rat_data[2]
    conc_dz = conc_rat_data[3]

    #hamilton group velocity
    c_const = 3E8
    dxds = 2*(c_const**2)*kx
    dyds = 2*(c_const**2)*ky
    dzds = 2*(c_const**2)*kz
    #hamilton dkdt
    dkxds = -(w**2)*conc_dx
    dkyds = -(w**2)*conc_dy
    dkzds = -(w**2)*conc_dz
    dwds = 0

    #hamilton dwdt
    return [dxds,dyds,dzds,dkxds,dkyds,dkzds,dwds];

def ode_set2(path_val,y,wave_vec,wave_vec_diff):
    #spatial ode for calculating polarizations
    polarize_x = y[0]
    polarize_y = y[1]
    polarize_z = y[2]
    k_x = wave_vec[0]
    k_y = wave_vec[1]
    k_z = wave_vec[2]

    k_x_diff = wave_vec_diff[0]
    k_y_diff = wave_vec_diff[1]
    k_z_diff = wave_vec_diff[2]
    d_product = polarize_x*k_x_diff + polarize_y*k_y_diff + polarize_z*k_z_diff


    polarize_dx = -k_x*d_product
    polarize_dy = -k_y*d_product
    polarize_dz = -k_z*d_product

    return [polarize_dx,polarize_dy,polarize_dz]

def ode_plasmawave_set1(time,y,conc_rat_data,e_temp_data):
    x_coord = y[0]
    y_coord = y[1]
    z_coord = y[2]
    #wave vector
    kx = y[3]
    ky = y[4]
    kz = y[5]
    k2 = kx**2 + ky**2 + kz**2
    #concentration values
    conc_rat = conc_rat_data[0]
    conc_dx = conc_rat_data[1]
    conc_dy = conc_rat_data[2]
    conc_dz = conc_rat_data[3]
    etemp = e_temp_data[0]
    etemp_dx = e_temp_data[1]
    etemp_dy = e_temp_data[2]
    etemp_dz = e_temp_data[3]
    disp = dispersion_plasma(np.sqrt(k2),conc_rat,etemp)#,np.sqrt(wp_const_mks*conc_rat + thermal_coeff_mks*k2*etemp)
    if disp == 0:
        print(x_coord,y_coord,z_coord,conc_rat,etemp,time)
    dxdt = kx*thermal_coeff_mks*etemp/disp
    dydt = ky*thermal_coeff_mks*etemp/disp
    dzdt = kz*thermal_coeff_mks*etemp/disp
        #print(dydt,ky,etemp,disp)
        #print(disp)
    dkxdt = -(wp_const_mks*conc_dx + thermal_coeff_mks*k2*etemp_dx)/(2*disp)
    dkydt = -(wp_const_mks*conc_dy + thermal_coeff_mks*k2*etemp_dy)/(2*disp)
    dkzdt = -(wp_const_mks*conc_dz + thermal_coeff_mks*k2*etemp_dz)/(2*disp)
    return [dxdt,dydt,dzdt,dkxdt,dkydt,dkzdt]

calc_lambda_debye = lambda Ne,Te:np.sqrt(e0*Te/(Ne*e_charge**2))

def calcKappa(k0,Te,Ne,Nc):
    lambda_debye = calc_lambda_debye(Ne,Te)
    omega_eq = (2/3)*(np.sqrt(Nc/Ne)-2)
    kappa_eq = np.sqrt(omega_eq/2-(k0*lambda_debye/2)**2)
    return [kappa_eq,lambda_debye]

def eq_wavematch(kx,k0,Te,Ne,Nc):
    lambda_debye = calc_lambda_debye(Ne,Te)#np.sqrt(e0*Te/(Ne*e_charge**2))
    omega_eq = (2/3)*(np.sqrt(Nc/Ne)-2)
    kappa_eq = np.sqrt(omega_eq/2-(k0*lambda_debye/2)**2)
    kx_term = (kx*lambda_debye - k0*lambda_debye/2)**2
    ky = np.sqrt(kappa_eq**2-kx_term)/lambda_debye
    return ky

def eq_tbd_growth(kx,k0,Te,Ne):
    lambda_debye = calc_lambda_debye(Ne,Te)
    kx_term = (kx*lambda_debye - k0*lambda_debye/2)**2
    k0_term = (k0*lambda_debye/2)**2
    ky = np.sqrt(kx_term-k0_term)/lambda_debye
    return ky

def min_func(kx_guess,k0_val,Te_val,Ne_val,Nc_val):
    p1 = eq_wavematch(kx_guess,k0_val,Te_val,Ne_val,Nc_val)
    p2 = eq_tbd_growth(kx_guess,k0_val,Te_val,Ne_val)
    return p1-p2

def plasmon_gen_k(k_light,coords,crit_val,crit_now,te_now,t):
    #crit_now = ne_data_calc(coords,t)[0]
    #te_now = te_data_calc(coords,t)[0]
    k0 = np.sqrt(k_light[0]**2+k_light[1]**2+k_light[2]**2)
    [kappa_val,ldeb_val] = calcKappa(k0,te_now,crit_now,crit_val)
    minfunc_spec = lambda x: min_func(x,k0,te_now,crit_now,crit_val)
    pos_kx_domain = [k0,(0.5*k0*ldeb_val+kappa_val)/ldeb_val]
    neg_kx_domain = [(0.5*k0*ldeb_val-kappa_val)/ldeb_val,0]
    pos_kx = np.abs(sciop.bisect(minfunc_spec,pos_kx_domain[0],pos_kx_domain[1]))
    pos_ky = eq_wavematch(pos_kx,k0,te_now,crit_now,crit_val)
    neg_kx = -np.abs(sciop.bisect(minfunc_spec,neg_kx_domain[0],neg_kx_domain[1]))
    neg_ky = -eq_wavematch(neg_kx,k0,te_now,crit_now,crit_val)
    #k1 = np.array([k0,0,0]) - k2
    k1 = np.array([pos_kx,pos_ky,0])
    k2 = np.array([neg_kx,neg_ky,0])

    return [k1,k2]
