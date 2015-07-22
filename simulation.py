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
    
    
def get_args():
    ##parent parser
    parser = argparse.ArgumentParser(description="parent set of parameters", add_help=False)
    parser.add_argument("--name", default="simple", type=str)
    
    ##real parser
    par = argparse.ArgumentParser(description="Options for Jac_run_module. Use --cwd for analysis on not the current working directory")
    sub = par.add_subparsers(dest="step")
    
    sim_sub = sub.add_parser("sim", parents=[parser], help="For running a regular simulation")
    sim_sub.add_argument("--steps", default=1000000, type=int, help="Number of Langevin dynamics steps to take")
    
    args = par.parse_args()
    
    return args
    
if __name__ == "__main__":
    args = get_args()
    
    if args.step == "sim":
        run_sim(args)
    
    
