"""
Holds class and methods for a model for langevin dynamics
"""
import ConfigParser
import os
import numpy as np
import shutil

######### MAIN CLASS #########
class langevin_model(object):
    def __init__(self, name):
        #initialize the model
        config = ConfigParser.SafeConfigParser(allow_no_value=True)
        config.read("%s.ini" % name)
        modelopts = {}
        parse_configs(modelopts, config)
        self.name = name
        self.setup_model(modelopts)
        self.iteration = modelopts["iteration"]
        
        self.number_parameters = len(self.params)
        
        ## below might be completley deprecated. Not useful now, but maybe useful in the future
        self.save_potentials_file = modelopts["potentials_file"]
        self.save_params_file = modelopts["model_params_file"]
        if self.save_potentials_file == None:
            self.save_potentials_file = "potentials"
        if self.save_params_file == None:
            self.save_params_file = "model_params"
        
        #self.save_model()
        
    def setup_model(self, modelopts):
        
        ##load the potential functions
        self.potentials = []           #list of potential function indices
        self.potentials_arguments = [] #list containing a list of each functions parameters
        self.fit_parameters = []       #list containing True or False depending on whether or not that function should be fitted
        self.number_fit_parameters = 0 #counts number of parameters to fit
        try: #if it finds the potentials file, load that information
            f = open(modelopts["potentials_file"],"r")
            for line in f: #read each line and append useful information
                stuff = line.split()
                self.potentials.append(int(stuff[0])) 
                if not int(stuff[0]) == 0: # only fit if it's not the zeroth index function - harmonic well trap
                    self.number_fit_parameters += 1
                    self.fit_parameters.append(True)
                else:
                    self.fit_parameters.append(False)
                stuff = stuff[1:]
                stuff = [float(i) for i in stuff]
                self.potentials_arguments.append(stuff)
                
        except: #if no potentials file specified, use a default harmonic well
            self.potentials = [0]
            self.potentials_arguments.append([])
            self.fit_parameters.append(False)
        self.potential_functions = []  #list of the actual callable potential funtions V(x) 
        self.force_functions = []      #list of the actual callable force functions f(x)
        
        for i in range(len(self.potentials)): #wrap up the potential and force functions and add them to the list in order
            self.potential_functions.append(choose_potential(self.potentials[i], self.potentials_arguments[i] ))
            self.force_functions.append(choose_force(self.potentials[i], self.potentials_arguments[i] ))
        
        #set the model params, default to 1 if none found
        try: 
            self.params = np.loadtxt(modelopts["model_params_file"])
        except:
            self.params = np.ones(np.shape(self.potential_functions)[0])
        
        ##set up fit_indices
        self.fit_indices = []          #list of force/potential indices that would be fitted
        for i in range(np.shape(self.fit_parameters)[0]):
            if self.fit_parameters[i]:
                self.fit_indices.append(i)
                
    def net_force(self, x):
        #returns the net force scaled by the params
        net_force = 0.0
        for i in range(self.number_parameters):
            net_force += self.params[i] * self.force_functions[i](x)
        return net_force
        
    def net_potential(self, x):
        #returns the net potential scaled by the params
        net_potential = 0.0
        for i in range(self.number_parameters):
            net_potential += self.params[i] * self.potential_functions[i](x)
        return net_potential
        
    def save_ini_file(self):
        #save the ini file with the current information
        name = self.name
        config = ConfigParser.SafeConfigParser(allow_no_value=True)
        config.add_section("model")
        
        config.set("model", "iteration", "%d" % self.iteration)
        config.set("model", "potentials_file", self.save_potentials_file)
        config.set("model", "model_params_file", self.save_params_file)

        if os.path.exists("%s.ini" % self.name):
            shutil.move("%s.ini" % self.name,"%s.1.ini" % self.name)

        with open("%s.ini" % self.name,"w") as cfgfile:
            config.write(cfgfile)
    
    def save_parameter_files(self):
        #save parameters files such as the potentials functions and model params
        np.savetxt("params", self.params)
        potentials_string = ""
        
        for i in range(len(self.potentials)):
            potentials_string += "%2d    " % self.potentials[i]
            for j in range(len(self.potentials_arguments)):
                potentials_string += "%10f " % j
                potentials_string += "\n"
        f = open("potentials", "w")
        f.write(potentials_string)
        f.close()

######### GENERAL USEFUL FUNCTIONS #########
def empty_modelopts():
    """Model options to check for"""
    opts = ["model_params_file", "iteration"] 
    modelopts = { opt:None for opt in opts }
    return modelopts

     
def parse_configs(modelopts, config):
    """ reads the modelopts and parses the data"""
    for item,value in config.items("model"):
        if value in [None,""]:
            pass
        else:
            print "  %-20s = %s" % (item,value)
            if item == "iteration":
                value = int(value)
            else:
                value = str(value)
                
        modelopts[item] = value


######### WRAPPER FUNCTIONS FOR CREATING THE POTENTIAL FUNCTIONS #########
  
def choose_potential(index, args):
    potential = {0:Harmonic_trap_V, 
                 1:Harmonic_V,
                 2:Double_well_V,
                 3:gaussian_well_V}
    
    def new_potential(x):
        return potential[index](x, *args)
    return new_potential

def choose_force(index, args):
    force = {0:Harmonic_trap_f,
             1:Harmonic_f,
             2:Double_well_f,
             3:gaussian_well_f}
    def new_force(x):
        return force[index](x, *args)
    return new_force

######### FORCES AND POTENTIAL FUNCTIONS DEFINED BELOW #########
def Harmonic_trap_V(x):
    return 0.05*(x**2)

def Harmonic_trap_f(x):
    return -0.1*x


def Harmonic_V(x, x0=0.0, spread=0.5):
    return spread*((x-x0)**2)

def Harmonic_f(x, x0=0.0, spread=0.5):
    return -1.*spread*2.0*(x-x0)


def Double_well_V(x):
    return (1.0/(5.0**4))*(x**2 - 5.0**2)**2

def Double_well_f(x):
    return -1.0*(1.9/(5.0**4))*2.0*(x**2 - 5.0**2)*(2.0*x)  
             
    
def gaussian_well_V(x,x0=1.0,sigma=0.5,depth=2.0):
    return -1.0*depth*np.exp(-1.0*(((x-x0)/sigma)**2))

def gaussian_well_f(x,x0=1.0,sigma=0.5,depth=2.0):
    return -2.0*depth*((x-x0)/(sigma**2))*np.exp(-1.0*(((x-x0)/sigma)**2))
          

