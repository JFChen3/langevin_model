"""
Package containing functions for running a simulation
"""


import argparse
import langevin_model.model as model
import langevin_model.compute as compute
import os
import numpy as np

def run_sim(args):
    cwd = os.getcwd()
    new = model.langevin_model(args.name)
    
    x,v = compute.run_langevin(new, nsteps=args.steps)
    iterdir = "iteration_%d"%new.iteration
    
    try:
        os.mkdir(iterdir)
    except:
        pass
        
    np.savetxt("%s/position.dat"%iterdir, x)
    np.savetxt("%s/velocity.dat"%iterdir, v)
    os.chdir(iterdir)
    compute.plot_x_inverted(new,x)
    
    
    
##jacobian calculations:
def run_jac_procedure(args):
    target_feature = np.loadtxt(args.target)
    cwd = os.getcwd()
    new = model.langevin_model(args.name)
    iterdir = "%s/iteration_%d"%(cwd,new.iteration)
    newtondir = "%s/newton" % iterdir
    try:
        os.mkdir(newtondir)
    except:
        pass
        
    os.chdir(iterdir)
    x = np.loadtxt("position.dat")
    range_hist = (-1.0*args.histrange, 1.0*args.histrange)
    sim_feature, bincenters, slices, spacing = compute.plot_x_histogram(x, "iteration_%d"%new.iteration, nbins=args.nbins, histrange=range_hist) 
    Jacobian = compute.compute_jacobian(new,x, slices, sim_feature, args.nbins, spacing)        
    
    os.chdir(newtondir)
    np.savetxt("sim_feature.dat", sim_feature)
    np.savetxt("target_feature.dat", target_feature)
    np.savetxt("Jacobian.dat", Jacobian)
    compute.fit_jacobian()
    
def run_fit_procedure(args):   
    cwd = os.getcwd()
    new = model.langevin_model(args.name)

    cwd = os.getcwd()
    new = model.langevin_model(args.name)
    iterdir = "%s/iteration_%d"%(cwd,new.iteration)
    newtondir = "%s/newton" % iterdir
    
    os.chdir(newtondir)
    lambda_index = get_lambda_index(args.cutoff)
    param_changes = np.loadtxt("xp_%d.dat" % lambda_index)
    
    if np.max(param_changes) > args.scale:
        param_changes *= (args.scale/np.max(param_changes))
        
    count = 0
    for i in new.fit_indices:
        new.params[i] += param_changes[count]
        count += 1
    
    np.savetxt("params", new.params)

def run_next_step(args):
    cwd = os.getcwd()
    new = model.langevin_model(args.name)
    
    cwd = os.getcwd()
    new = model.langevin_model(args.name)
    iterdir = "%s/iteration_%d"%(cwd,new.iteration)
    newtondir = "%s/newton" % iterdir
    
    new.iteration += 1
    
    new.save_params_file = "%s/params" % newtondir
    
    if args.start:
        new.save_ini_file()
        
    
def get_lambda_index(cutoff):
    if cutoff == 0:
        return 0
    else:
        lvalues = np.loadtxt("lambdas.dat") 
        svf = np.loadtxt("singular_values.dat")
        index = 0
        max_search = np.shape(svf)[0]
        go = True
        i = 0 
        while (go and i < max_search):
            found_lambda, high, low = test_truncate(lvalues[i], svf, cutoff)
            if found_lambda:
                go = False
            i += 1
        open("Lambda_index.txt","w").write("%d"%(i-1))
        return i-1
        
def test_truncate(lam, svf, trunc):
    if svf[0] < lam:
        raise IOError("Larges singular value is smaller than the truncate value, consider changing")
    elif svf[np.shape(svf)[0]-1] > lam:
        high = svf[np.shape(svf)[0]-1]
        low = 0
    else:    
        for i in range(np.shape(svf)[0]-1):
            if svf[i] >= lam and svf[i+1] <= lam:
                low = svf[i+1]
                high = svf[i]
    if high >= trunc and low <=trunc:
        return True, high, low
    else:
        return False, high, low            
    
def get_args():
    ##parent parser
    parser = argparse.ArgumentParser(description="parent set of parameters", add_help=False)
    parser.add_argument("--name", default="simple", type=str)
    
    ##real parser
    par = argparse.ArgumentParser(description="Specify step sim, jac, fit, or next.")
    sub = par.add_subparsers(dest="step")
    
    sim_sub = sub.add_parser("sim", parents=[parser], help="For running a regular simulation")
    sim_sub.add_argument("--steps", default=1000000, type=int, help="Number of Langevin dynamics steps to take")
    
    jac_sub = sub.add_parser("jac", parents=[parser], help="for calculating a new fitted Jacobian")
    jac_sub.add_argument("--target", type=str, default="ideal_hist.dat", help="specify file for fitting the jacobian to")
    jac_sub.add_argument("--nbins", type=int, default=400)
    jac_sub.add_argument("--histrange", type=float, default=20.0)
    
    fit_sub = sub.add_parser("fit", parents=[parser], help="for taking a fitted simulation and calculating the new parameters")
    fit_sub.add_argument("--cutoff", default=0, type=float, help="specify the singular value cutoff default is 0, for all")
    fit_sub.add_argument("--scale", default = 0.2, type=float, help="specify the maximum step scale for the solution")
    
    next_sub = sub.add_parser("next", parents=[parser], help="for taking the next step in the simulation")
    next_sub.add_argument("--start", default=False, dest="start", action="store_true")
    args = par.parse_args()
    
    return args
    
if __name__ == "__main__":
    args = get_args()
    
    if args.step == "sim":
        print "starting simulation step..."
        run_sim(args)
    if args.step == "jac":
        print "starting Jacobian computation step..."
        run_jac_procedure(args)
    if args.step == "fit":
        print "Starting the fitting step..."
        run_fit_procedure(args)
    if args.step == "next":
        print "Starting the next step..."
        run_next_step(args)
    
