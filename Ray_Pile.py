import numpy as np
import scipy.integrate as sc
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import pickle
#import testODESwig
import ode_func as odef
import multiprocessing as mp

class Profile:
    pass

class RectangularProfile(Profile):
    def __init__(self,originVector,dimensions,densityInterpolator,temperatureInterpolator):
        #Desc: Initializes a rectangular plasma profile
        #Inputs:
        #originVector: 3 element value of x,y,z assume float
        #dimensions: 3 element value of width(x),length(y) and height(z), assume positive float
        #densityInterpolator: function which can return electron density using coordinates as input
        #temperatureInterpolator: function which can return electron temperature using coordinates as input
        #Output: Rectangular Profile object
        self.xBounds = [originVector[0],originVector[0] + dimensions[0]]
        self.yBounds = [originVector[1],originVector[1] + dimensions[1]]
        self.zBounds = [originVector[2],originVector[0] + dimensions[0]]
        self.ne_interp = densityInterpolator
        self.te_interp = temperatureInterpolator

    def withinBounds(self,coord):
        #Desc: Checks whether if the specified coordinate is within the Profile
        #Inputs:
        #coord: 3 element value assume float
        #Output: True or False
        if self.xBounds[0] <= coord[0] and self.xBounds[1] >= coord[0]:
            if self.yBounds[0] <= coord[1] and self.yBounds[1] >= coord[1]:
                if self.zBounds[0] <= coord[2] and self.zBounds[1] >= coord[2]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False


class segmentBlock:
    def __init__(self,timeseries):
        self.ray_time = timeseries
        L = timeseries.shape(0)
        self.ray_flags = np.zeros([L])
        self.ray_position[id_ray] = np.zeros([3,L])
        self.ray_wave_vector[id_ray] = np.zeros([3,L])
        self.ray_frequency[id_ray] = np.zeros([L])
        self.ray_polarization[id_ray]  = np.zeros([3,L])
        self.ray_ne[id_ray] = np.zeros([L])
        self.ray_te[id_ray] = np.zeros([L])
        self.ray_absolute_path[id_ray] = np.zeros([L])
        self.ray_conditions[id_ray] = np.zeros([L])
    def refresh(self,timeseries):
        self.ray_time = timeseries
        L = timeseries.shape(0)
        self.ray_flags = np.zeros([L])
        self.ray_position[id_ray] = np.zeros([3,L])
        self.ray_wave_vector[id_ray] = np.zeros([3,L])
        self.ray_frequency[id_ray] = np.zeros([L])
        self.ray_polarization[id_ray]  = np.zeros([3,L])
        self.ray_ne[id_ray] = np.zeros([L])
        self.ray_te[id_ray] = np.zeros([L])
        self.ray_absolute_path[id_ray] = np.zeros([L])
        self.ray_conditions[id_ray] = np.zeros([L])

#automate backpropgation, a decreasing time step should inidicate backpropagation
class ray_pile:
    def __init__(self,ne_interpFunc,te_interpFunc):
        self.ray_id = [] #id,type,originator
        self.ray_children = [] #indices of children rays
        self.ray_flags = [] #flagged when certain criteria is met
        self.ray_time_system = [] # 0 for s, 1 for t

        self.ray_time = [] #stores values of time
        self.ray_position = [] #stores x,y,z location
        self.ray_wave_vector = [] #stores wave vector in x,y,z dimensions
        self.ray_frequency = [] #stores the angular frequency
        self.ray_w0 = [] #stores the inital angular frequency (c*k for light)
        #enabled just for use with C
        self.ne_data_matrix = None
        self.ne_data_origin = None
        self.ne_data_spacing = None
        #pulls electron density information as required in solver
        self.ne_interp_func = ne_interpFunc #stores the function to interpolate electron density
        self.te_interp_func = te_interpFunc #stores the function to interpolate electron temperature (actually energy)


        self.ray_ne = []; #stores electron density along the path of the ray
        self.ray_te = []; #stores electron temperature along the path of the ray

        self.ray_polarization = [] #stores x,y,z components of polarization along ray path
        self.ray_absolute_path = [] #parametrized distance, used for polarization calculations

        self.ray_conditions = [] #can store flags, currently not used in the code

    def enable_c(self,data_matrix,origin,spacing):
        '''
        stores data for c ode solver to be run
        ne_data_matrix: a 3d matrix of data in cylindrical coordinates
        origin: 3 member numpy array of the origin of matrix
        spacing: 3 member numpy array of spacing along each axis
        '''
        self.ne_data_matrix = data_matrix
        self.ne_data_origin = origin
        self.ne_data_spacing = spacing

    def ode_time_s_em(self,s,y):
        #passed into ode solver
        #should deprecate due to s time scale
        conc_rat_data = self.ne_interp_func(y[0:3],s)
        return odef.ode_set1_s(s,y,conc_rat_data)

    def ode_time_t_em(self,t,y):
        #passed into ode solver
        #pertains to solving the ray path of a light ray in plasma
        conc_rat_data = self.ne_interp_func(y[0:3],t)
        return odef.ode_set1_time(t,y,conc_rat_data)

    def ode_time_t_pl(self,t,y):
        #passed into ode solver
        #pertains to solving the ray path of a plasma ray
        conc_rat_data = self.ne_interp_func(y[0:3],t)
        te_data = self.te_interp_func(y[0:3],t)
        return odef.ode_plasmawave_set1(t,y,conc_rat_data,te_data)

    def ode_path(self,x,y,wave_vec,vec_diff):
        #passed into ode solver
        #peratins to solving the polarization along a light ray
        return odef.ode_set2(x,y,wave_vec,vec_diff)


    def create_ray(self,type,originator,time_system,time_span,position,ray_wave_vector,w0,frequency,polarization):
        """
        this function is better off not being called directly, it adds rays to the ray pile
        type: integer: specifies the type of ray, currently 0 is for light rays and 1 is for plasma rays
        originator:float: in the case of branching rays, this specifies the ray id it was created from, 0.5 if an original ray
        time_system: 0 for s (divide by 2 omega in the notes), 1 for normal, should be phased out of use
        time_span: nparray or list of floats: an array of all time values to calculate for
        position: nparray or list of 3 floats: contains the starting position of ray
        ray_wave_vector: nparray or list of 3 floats: contains the starting wave vector, equivalent to k
        w0: float: laser frequency, c*k, useful for determining critical densities
        frequency: float: initial angular frequency
        polarization: nparray or list of 3 floats: contains initial polarization vector
        """
        if len(self.ray_id) == 0: #use this if nothing in ray pile
            self.ray_id.append([0,type,0.5])
        else: #creates new ray in already existing set of rays
            self.ray_id.append([self.ray_id[-1][0]+1,type,originator])

        #create an array to store various properties
        self.ray_time.append(time_span)
        self.ray_time_system.append(time_system) #should deprecate
        self.ray_children.append([]) #list all children made from braching processes
        self.ray_flags.append(np.zeros(time_span.shape)) #flag certain time points using integers
        self.ray_position.append(np.zeros([3,time_span.shape[0]]))
        self.ray_position[-1][:,0] = position
        self.ray_wave_vector.append(np.zeros([3,time_span.shape[0]],dtype = np.double))
        self.ray_wave_vector[-1][:,0] = ray_wave_vector
        self.ray_frequency.append(np.zeros(time_span.shape))
        self.ray_frequency[-1][0] = frequency
        self.ray_w0.append(w0)
        self.ray_polarization.append(np.zeros([3,time_span.shape[0]]))
        self.ray_polarization[-1][:,0] = polarization
        self.ray_ne.append(np.zeros([time_span.shape[0]]))
        self.ray_te.append(np.zeros([time_span.shape[0]]))
        self.ray_absolute_path.append(np.zeros(time_span.shape,dtype = np.double))
        self.ray_conditions.append(np.zeros([4,time_span.shape[0]]))

    def create_light_ray(self,time_array,pos_array,wave_vec,polarization,orig = 0.5):
        """
        creates a light ray using create_ray
        time_array: list or nparray of floats: stores time values, same as in create_ray
        pos_array: list or nparray of 3 floats: stores initial position
        wave_vec: list or nparray of 3 floats: stores initial wave vector
        polarization: list or nparray of 3 floats: stores initial polarization
        orig: float: by default this implies creating an original ray
                     replace with integer ray_id if produced from branching
        """
        #find the initial electron density
        init_dens = self.ne_interp_func(pos_array,time_array[0])
        init_dens = init_dens[0]
        #find the magnitude of wave vector and calculate angular frequency and laser frequency
        k = np.sqrt(wave_vec[0]**2 + wave_vec[1]**2 + wave_vec[2]**2)
        w0 = odef.dispersion_laser(k)
        freq = odef.dispersion_light(k,init_dens)
        #create the ray
        self.create_ray(0,orig,1,time_array,pos_array,wave_vec,w0,freq,polarization)

    def create_plasma_ray(self,time_array,pos_array,wave_vec,orig = 0.5):
        """
        creates a plasma ray using create_ray
        time_array: list or nparray of floats: stores time values, same as in create_ray
        pos_array: list or nparray of 3 floats: stores initial position
        wave_vec: list or nparray of 3 floats: stores initial wave vector
        orig: float: by default this implies creating an original ray
                     replace with integer ray_id if produced from branching
        """
        #find initial electron density and temperature
        init_dens = self.ne_interp_func(pos_array,time_array[0]);
        init_dens = init_dens[0];
        init_temp = self.te_interp_func(pos_array,time_array[0]);
        init_temp = init_temp[0];
        #calculate k magnitude and find relevant frequencies
        k = np.sqrt(wave_vec[0]**2 + wave_vec[1]**2 + wave_vec[2]**2);
        w0 = odef.dispersion_plasma(k,init_dens,init_temp)
        freq = odef.dispersion_plasma(k,init_dens,init_temp)
        #[-1,-1,-1] for plasma polarization as plasma waves do not have polarization
        self.create_ray(1,orig,1,time_array,pos_array,wave_vec,w0,freq,[-1,-1,-1])

    def ode_neg(self,inputList):
        '''
        multiplies output of an ode rhs function by -1, used for back propagation
        '''
        outputList = np.array(inputList)
        return -1*outputList

    def prop_em_step1(self,id_ray,backprop = False):
        """
        calculates the path of the ray and its wave vector along the path of light rays
        id_ray: corresponds to index of the ray
        backprop: Boolean: default False for forward travel, if True time scale is reversed
        """

        #in the case of backpropagation, the time is flipped
        if backprop:
            time_bound = [self.ray_time[id_ray][-1],self.ray_time[id_ray][0]]
        else:
            time_bound = [self.ray_time[id_ray][0],self.ray_time[id_ray][-1]]

        #creates the initial values for ode solver, first 3 are for position, last 3 for wve vector
        init_con = np.zeros([6])
        init_con[:3] = self.ray_position[id_ray][:,0]
        init_con[3:] = self.ray_wave_vector[id_ray][:,0]
        #in the case of backpropagation, flip time
        if backprop:
            ''' might deprecate
            if self.ray_time_system[id_ray] == 0:
                back_time_s = lambda s,y: self.ode_time_s_em(time_bound[1]-(s-time_bound[0]),y)
                ray_sol = sc.solve_ivp(back_time_s,time_bound,init_con,t_eval=np.flip(self.ray_time[id_ray]),rtol = 1e-2)
            else:
            '''
            #generate a lambda function suitable for backwards propagation and solve
            back_time_t = lambda t,y: self.ode_neg(self.ode_time_t_em(time_bound[1]-(t-time_bound[0]),y))
            ray_sol = sc.solve_ivp(back_time_t,time_bound,init_con,t_eval=np.flip(self.ray_time[id_ray]),rtol = 1e-4)
        else:
            ''' might deprecate
            if self.ray_time_system[id_ray] == 0:
                ray_sol = sc.solve_ivp(self.ode_time_s_em,time_bound,init_con,t_eval=self.ray_time[id_ray],rtol = 1e-2,max_step = self.ray_time[id_ray][100]-self.ray_time[id_ray][0])
            else:
            '''
            #solve for properties of the forward travelling wave
            ray_sol = sc.solve_ivp(self.ode_time_t_em,time_bound,init_con,t_eval=self.ray_time[id_ray],rtol=1e-4)
        #store the values calculated
        self.ray_position[id_ray] = ray_sol.y[0:3,:]
        self.ray_wave_vector[id_ray] = ray_sol.y[3:6,:]
        #calculate and store values for electron density, electron temperature and ray_frequency
        for index,tval in enumerate(self.ray_time[id_ray]):
            self.ray_ne[id_ray][index] = self.ne_interp_func(self.ray_position[id_ray][:,index],tval)[0]
            #print(self.ne_interp_func(self.ray_position[id_ray][:,index],tval)[0])
            self.ray_te[id_ray][index] = self.te_interp_func(self.ray_position[id_ray][:,index],tval)[0]
            k = np.sqrt(sum(self.ray_wave_vector[id_ray][:,index]**2))
            self.ray_frequency[id_ray][index] = odef.dispersion_light(k,self.ray_ne[id_ray][index])


    def prop_em_step2(self,id_ray):
        """
        calculate the polarization along the path of a light ray
        id_ray:integer: index value of the ray to find polarization for
        """
        #parametrize the path distance
        diff = np.linalg.norm(np.abs(self.ray_position[id_ray][:,1:] - self.ray_position[id_ray][:,0:-1]),ord=2,axis = 0,keepdims=True)
        for pos_ind in range(1,self.ray_absolute_path[id_ray].shape[0]):
            self.ray_absolute_path[id_ray][pos_ind] = self.ray_absolute_path[id_ray][pos_ind-1] + diff[0,pos_ind-1]
        #path bounds specify the limits for the ode_solver
        path_bounds = [0,self.ray_absolute_path[id_ray][-1]]
        #calculate necessary values for the ode solver
        wave_vec_mag = np.linalg.norm(self.ray_wave_vector[id_ray], ord=2, axis=0, keepdims=True)
        unit_wave_vector = np.zeros(self.ray_wave_vector[id_ray].shape)
        unit_wave_diff = np.zeros(self.ray_wave_vector[id_ray].shape)
        for index in [0,1,2]:
            unit_wave_vector[index,:] = self.ray_wave_vector[id_ray][index,:]/wave_vec_mag
            unit_wave_diff[index,:] = np.gradient(unit_wave_vector[index,:],self.ray_absolute_path[id_ray])
            #print(unit_wave_diff)
        #generate interpolators for wave vector and wave vector gradient
        wave_vector_interp = interp.interp1d(self.ray_absolute_path[id_ray],unit_wave_vector)
        wave_vector_gradient_interp = interp.interp1d(self.ray_absolute_path[id_ray],unit_wave_diff)
        #create a lambda function to pass into ode solver
        path_ode = lambda x,y: odef.ode_set2(x,y,wave_vector_interp(x),wave_vector_gradient_interp(x))
        init_pol_con = self.ray_polarization[id_ray][:,0]
        #solve and store polarization
        ray_pol_sol = sc.solve_ivp(path_ode,path_bounds,init_pol_con,t_eval=self.ray_absolute_path[id_ray],rtol=5e-4,method='LSODA')
        self.ray_polarization[id_ray] = ray_pol_sol.y[0:3,:]

    def propagate_ray(self,id_ray,backprop=False):
        """
        call function to direct wave propagation by calling other functions
        set backprop = True if time is decreasing (back propagation)
        """
        if self.ray_id[id_ray][1] == 0: #process for light ray
            self.prop_em_step1(id_ray,backprop)
            self.prop_em_step2(id_ray)
        elif self.ray_id[id_ray][1] == 1: #process for plasma ray
            self.prop_pl_step1(id_ray,backprop)

    def prop_pl_step1(self,id_ray,backprop=False):
        """
        Plasma Ray Propagation
        id_ray: index of ray
        backprop: default false for forward propagation, set True for back propagation
        """
        # set initial conditions for ode solver, position then wave vector
        init_con = np.zeros([6])
        init_con[:3] = self.ray_position[id_ray][:,0]
        init_con[3:6] = self.ray_wave_vector[id_ray][:,0]
        if backprop: #back propagation, time will already be flipped
            time_bound = [self.ray_time[id_ray][-1],self.ray_time[id_ray][0]]
            back_plasma = lambda t,y: self.ode_neg(self.ode_time_t_pl(time_bound[1]-(t-time_bound[0]),y))
            ray_sol = sc.solve_ivp(self.ode_time_t_pl,time_bound,init_con,t_eval=np.flip(self.ray_time[id_ray]),rtol = 1e-4)
        else:
            time_bound = [self.ray_time[id_ray][0],self.ray_time[id_ray][-1]]
            ray_sol = sc.solve_ivp(self.ode_time_t_pl,time_bound,init_con,t_eval=self.ray_time[id_ray],rtol=1e-4)

        #stores values calculated
        self.ray_position[id_ray] = ray_sol.y[0:3,:]
        self.ray_wave_vector[id_ray] = ray_sol.y[3:6,:]

        #calculate values of electron density, temperature and frequency
        for index,tval in enumerate(self.ray_time[id_ray]):
            self.ray_ne[id_ray][index] = self.ne_interp_func(self.ray_position[id_ray][:,index],tval)[0]
            #print(self.ne_interp_func(self.ray_position[id_ray][:,index],tval)[0])
            self.ray_te[id_ray][index] = self.te_interp_func(self.ray_position[id_ray][:,index],tval)[0]
            k = np.sqrt(sum(self.ray_wave_vector[id_ray][:,index]**2))
            self.ray_frequency[id_ray][index] = odef.dispersion_plasma(k,self.ray_ne[id_ray][index],self.ray_te[id_ray][index])
        #self.ray_frequency[id_ray] = ray_sol.y[6,:]


    def back_propagate(self,id_ray):
        """
        run the propagation in reverse, useful for verification
        id_ray:integer: index of ray
        """
        #create new ray with flipped time then go through the correct propagation steps
        if self.ray_id[id_ray][1] == 0: #light ray
            self.create_light_ray(np.flip(self.ray_time[id_ray]),self.ray_position[id_ray][:,-1],self.ray_wave_vector[id_ray][:,-1],self.ray_polarization[id_ray][:,-1],orig = -1*id_ray)
            #self.create_ray(0,-1*id_ray,0,np.flip(self.ray_time[id_ray]),self.ray_position[id_ray][:,-1],-1*self.ray_wave_vector[id_ray][:,-1],self.ray_frequency[id_ray][-1],self.ray_polarization[id_ray][:,-1])
            self.prop_em_step1(len(self.ray_time)-1,backprop = True)
            self.prop_em_step2(len(self.ray_time)-1)
        elif self.ray_id[id_ray][1] == 1: #plasma ray
            self.create_plasma_ray(np.flip(self.ray_time[id_ray]),self.ray_position[id_ray][:,-1],self.ray_wave_vector[id_ray][:,-1],orig = -1*id_ray)
            self.prop_pl_step1(len(self.ray_time)-1,backprop = True)

    def propCondRay(self):
        """
        deprecated ray splitting
        """
        ray_pairs = []
        for ray_indice in range(len(self.ray_absolute_path)):
            ray_path = self.ray_absolute_path[ray_indice]
            path_indice = 0
            while ray_path[path_indice] < 8 and path_indice < len(ray_path)-1:
                path_indice += 1
            if path_indice < len(ray_path) and len(self.ray_children[ray_indice]) == 0:
                ray_pairs.append([ray_indice,path_indice])
        for index_pair in ray_pairs:
            ray_index = index_pair[0]
            path_index = index_pair[1]
            firstCoord = self.ray_position[ray_index][:,path_index]
            initialK = np.copy(self.ray_wave_vector[ray_index][:,path_index])
            initialP = np.copy(self.ray_polarization[ray_index][:,path_index])
            firstNewK = 0.5*initialK + 0.1*np.cross(initialK,initialP)
            secondNewK = 0.5*initialK - 0.1*np.cross(initialK,initialP)
            newKinit = np.sqrt(sum(firstNewK**2))
            self.create_ray(0,ray_index,0,np.linspace(0,3*1.6e-24,10000),firstCoord,firstNewK,3E8*newKinit,initialP)
            self.propagate_ray_em(-1)
            self.create_ray(0,ray_index,0,np.linspace(0,3*1.6e-24,10000),firstCoord,secondNewK,3E8*newKinit,initialP)
            self.propagate_ray_em(-1)
            self.ray_children[ray_indice] = [len(self.ray_children)-1,len(self.ray_children)-2]

    def proptime_light_c(self,ray_index):
        '''
        propagates light ray across time using the swig code
        ray_index: id of ray
        '''
        y_init = np.zeros([6])
        y_init[0:3] = self.ray_position[ray_index][:,0]
        y_init[3:6] = self.ray_wave_vector[ray_index][:,0]
        testODESwig.importIniDensity(self.ne_data_matrix,self.ne_data_origin,self.ne_data_spacing)
        timeStart = self.ray_time[ray_index][0]
        h_step = self.ray_time[ray_index][1] - timeStart
        y_out_arr = np.zeros([7],dtype=np.double)
        testODESwig.importIniConditions(timeStart,h_step/10,self.ray_w0[ray_index],self.ray_w0[ray_index],y_init)
        testODESwig.changeODESettings(h_step/10,1e-2,1e-2);
        for ind_ti,ti in enumerate(self.ray_time[ray_index]):
            testODESwig.step_ode_RTCyl(y_out_arr,ti)
            self.ray_position[ray_index][:,ind_ti] = np.copy(y_out_arr[1:4])
            self.ray_wave_vector[ray_index][:,ind_ti] = np.copy(y_out_arr[4:7])
        print('c time complete')

    def proppath_light_c(self,ray_index):
        '''
        propagates light ray across path using the swig code
        ray_index: id of ray
        '''
        y_init = np.copy(self.ray_polarization[ray_index][:,0])
        diff = np.linalg.norm(np.abs(self.ray_position[ray_index][:,1:] - self.ray_position[ray_index][:,0:-1]),ord=2,axis = 0,keepdims=True)
        for pos_ind in range(1,self.ray_absolute_path[ray_index].shape[0]):
            self.ray_absolute_path[ray_index][pos_ind] = self.ray_absolute_path[ray_index][pos_ind-1] + diff[0,pos_ind-1]
        y_out_arr = np.zeros([4],dtype=np.double)
        hStart = self.ray_absolute_path[ray_index][0]
        h_step = self.ray_absolute_path[ray_index][1] - hStart
        testODESwig.importRayTrail(self.ray_absolute_path[ray_index],self.ray_wave_vector[ray_index])
        testODESwig.set_pol_spline()
        testODESwig.importIniConditions(hStart,h_step,self.ray_w0[ray_index],self.ray_w0[ray_index],y_init)
        testODESwig.changeODESettings(h_step,1e-2,1e-2);
        for ind_s,si in enumerate(self.ray_absolute_path[ray_index][1:]):
                print('bef')
                testODESwig.step_ode_pol(y_out_arr,si)
                print('after')
                self.ray_polarization[ray_index][:,ind_s+1] = np.copy(y_out_arr[1:])
        #testODESwig.step_ode_pol(y_out_arr,para_x[1])
        testODESwig.free_pol_spline();
        print('c path complete')

    def ray_split(self,id_ray,time,proportion):
        """
        Split light ray into light and plasma
        id_ray: integer:index of ray
        time: float:time value to split ray at, must be in ray_pile.ray_time[id_ray]
        proportion: float: percentage of angular frequency going to light ray
        """
        #timeInd = self.ray_time[id_ray].index(time)
        #timeArray = self.ray_time[id_ray][timeInd:]
        #posArray = self.ray_pos[id_ray][:,timeInd]
        #ne = self.ray_ne[id_ray][timeInd]
        #te =self.ray_te[id_ray][timeInd]
        #w = self.ray_frequency(timeInd)
        #new_polarization =
        #new_kdir_1 =
        #new_kdir_2 =
        #new_w_1 =
        #new_w_2 = w - new_w_1
        #new_k_1 = new_kdir_1*ode_func.k_light(new_w_1,ne)
        #new_k_2 = new_kdir_2*ode_func.k_plasma(new_w_1,ne,te)
        #self.create_light_ray(timeArray,posArray,new_k_1,new_polarization,orig = id_ray)
        #self.create_plasma_ray(timeArray,posArray,new_k_2,orig = id_ray)
        pass

    def landau_dampen(self,id_ray):
        data_clipped = False
        i = 0
        while data_clipped == False and i < len(self.ray_time[id_ray]):
            lambda_debye = odef.calc_lambda_debye(self.ray_ne[id_ray][i],self.ray_te[id_ray][i])
            k_val = np.sqrt(np.sum(self.ray_wave_vector[id_ray][0,i]**2 + self.ray_wave_vector[id_ray][1,i]**2  + self.ray_wave_vector[id_ray][2,i]**2 ))
            if k_val*lambda_debye > 0.3:
                self.clip_data(id_ray,i)
                data_clipped = True
                print(k_val*lambda_debye)
            i += 1

    def clip_data(self,id_ray,data_ind):
        self.ray_time[id_ray] = self.ray_time[id_ray][:data_ind];
        self.ray_flags[id_ray]  = self.ray_flags[id_ray][:data_ind];
        self.ray_position[id_ray] = self.ray_position[id_ray][:,:data_ind];
        self.ray_wave_vector[id_ray] = self.ray_wave_vector[id_ray][:,:data_ind]
        self.ray_frequency[id_ray] = self.ray_frequency[id_ray][:data_ind]
        self.ray_polarization[id_ray] = self.ray_polarization[id_ray][:,:data_ind]
        self.ray_ne[id_ray] = self.ray_ne[id_ray][:data_ind]
        self.ray_te[id_ray] = self.ray_te[id_ray][:data_ind]
        self.ray_absolute_path[id_ray] = self.ray_absolute_path[id_ray][:data_ind]
        self.ray_conditions[id_ray] = self.ray_conditions[id_ray][:,:data_ind]

    def fuse_data(self,id_ray,segment):
        self.ray_time[id_ray] = np.concatenate(self.ray_time[id_ray],segment.ray_time[1:])
        self.ray_flags[id_ray]  = np.concatenate(self.ray_flags[id_ray],segment.ray_flags[1:])
        self.ray_position[id_ray] = np.concatenate(self.ray_position[id_ray],segment.ray_position[:,1:],axis = 1);
        self.ray_wave_vector[id_ray] = np.concatenate(self.ray_wave_vector[id_ray],segment.ray_wave_vector[:,1:], axis = 1);
        self.ray_frequency[id_ray] = np.concatenate(self.ray_frequency[id_ray],segment.ray_frequency[1:]);
        self.ray_polarization[id_ray] = np.concatenate(self.ray_polarization[id_ray],segment.ray_polarization[:,1:], axis = 1);
        self.ray_ne[id_ray] = np.concatenate(self.ray_ne[id_ray],segment.ray_ne[1:]);
        self.ray_te[id_ray] = np.concatenate(self.ray_te[id_ray],segment.ray_te[1:]);
        self.ray_absolute_path[id_ray] = np.concatenate(self.ray_absolute_path[id_ray],segment.ray_absolute_path[1:]);
        self.ray_conditions[id_ray] = np.concatenate(self.ray_conditions[id_ray],segment.ray_conditions[1:]);

def pickleSave(ray_pile_instance,filename):
    pickle.dump(ray_pile_instance,open(filename,"wb"))

def pickleLoad(filename):
    return pickle.load(open(filename,"rb"))

'''
    def save_data(self):
        #incomplete
        writer = RT.RTopenPMDWriter('','testSave.h5')
        fakeMom = np.array([0,0,0,0,0,0],dtype=np.float32)
        fakePos1 = np.array([-10,0,0,0,0,10],dtype=np.float32)
        fakePos2 = np.array([0,-10,0,0,10,0],dtype=np.float32)
        fakePos3 = np.array([0,0,-10,10,0,0],dtype=np.float32)
        for i in range(0,self.ray_position[0].shape[1],500):
            posX = np.zeros([len(self.ray_position)+6],dtype = np.float32)
            posY = np.zeros([len(self.ray_position)+6],dtype = np.float32)
            posZ = np.zeros([len(self.ray_position)+6],dtype = np.float32)
            momX = np.zeros([len(self.ray_position)+6],dtype = np.float32)
            momY = np.zeros([len(self.ray_position)+6],dtype = np.float32)
            momZ = np.zeros([len(self.ray_position)+6],dtype = np.float32)
            for m in range(len(self.ray_position)):
                posX[m] = np.float32(self.ray_position[m][0,i])
                posY[m] = np.float32(self.ray_position[m][1,i])
                posZ[m] = np.float32(self.ray_position[m][2,i])
                momX[m] = np.float32(self.ray_polarization[m][0,i])
                momY[m] = np.float32(self.ray_polarization[m][1,i])
                momZ[m] = np.float32(self.ray_polarization[m][2,i])
            posX[-7:-1] = fakePos1[:]
            posY[-7:-1] = fakePos2[:]
            posZ[-7:-1] = fakePos3[:]
            momX[-7:-1] = fakeMom
            momY[-7:-1] = fakeMom
            momZ[-7:-1] = fakeMom

            writer.writeParticleVector('Rays','position',posX.copy(),posY.copy(),posZ.copy(),i+10000)
            writer.writeParticleVector('Rays','momentum',momX.copy(),momY.copy(),momZ.copy(),i+10000)
            writer.writeScalarMesh('ratio',np.ones([2,2,2],dtype=np.float32), i+10000)
        del writer
'''
