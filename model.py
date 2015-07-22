"""
Holds class and methods for a model for langevin dynamics
"""
import ConfigParser
import os
import numpy as np
class langevin_model(object):
    def __init__(self, name):
        
        config = ConfigParser.SafeConfigParser(allow_no_value=True)
        config.read("%s.ini" % name)
        modelopts = {}
        parse_configs(modelopts, config)
        self.setup_model(modelopts)
        self.iteration = modelopts["iteration"]
        
        
        
        self.save_potentials_file = modelopts["potentials_file"]
        self.save_params_file = modelopts["model_params_file"]
        if self.save_potentials_file == None:
            self.save_potentials_file = "potentials"
        if self.save_params_file == None:
            self.save_params_file = "model_params"
        
        self.save_model()
        
    def setup_model(self, modelopts):
        
        ##load the potential functions
        self.potentials = []
        self.potentials_arguments = []
        try:
            f = open(modelopts["potentials_file"],"r")
            for line in f:
                stuff = line.split()
                self.potentials.append(int(stuff[0]))
                stuff = stuff[1:]
                stuff = [float(i) for i in stuff]
                self.potentials_arguments.append(stuff)
        except:
            self.potentials = [0]
            self.potentials_arguments.append([])
        self.potential_functions = []
        self.force_functions = []
        
        for i in range(len(self.potentials)):
            self.potential_functions.append(choose_potential(self.potentials[i], self.potentials_arguments[i] ))
            self.force_functions.append(choose_force(self.potentials[i], self.potentials_arguments[i] ))
        
        #set the model params, default to 1 if none found
        try:
            self.params = np.loadtxt(modelopts["model_params_file"])
        except:
            self.params = np.ones(np.shape(self.potential_functions)[0])
        
        
    def net_force(self, x):
        net_force = 0.0
        for force in self.force_functions:
            net_force += force(x)
        return net_force
        
    def net_potential(self, x):
        net_potential = 0.0
        for potential in self.potential_functions:
            net_potential += potential(x)
        return net_potential
        
    def save_model(self):
        f = open(self.save_potentials_file,"w")
        for i in range(len(self.potentials)):
            f.write("%2d "%self.potentials[i])
            for j in self.potentials_arguments[i]:
                f.write(" %10f "%j)
            f.write("\n")
        f.close()
            
        np.savetxt(self.save_params_file, self.params)

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

##forces and potentials
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
    
def gaussian_well_V(x,x0=1.0,sigma=0.5,depth=-2.0):
    return -1.0*depth*np.exp(-1.0*(((x-x0)/sigma)**2))

def gaussian_well_f(x,x0=1.0,sigma=0.5,depth=-2.0):
    return -2.0*depth*((x-x0)/(sigma**2))*np.exp(-1.0*(((x-x0)/sigma)**2))
          
def empty_modelopts():
    """Model options to check for"""
    opts = ["model_params_file", "iteration"] 
    modelopts = { opt:None for opt in opts }
    return modelopts

     
def parse_configs(modelopts, config):
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
                
            
            
