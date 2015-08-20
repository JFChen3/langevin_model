""" file for running analysis scripts on a finished set of data"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import langevin_model.compute as compute
import langevin_model.model as model

def run_energy_plotting(name, savedir, iters, plot_range, plot_step, axis, title):
    #plot the potnetial energy of several different iterations
    cwd = os.getcwd()
    new = model.langevin_model(name)
    plot_values = np.arange(plot_range[0], plot_range[1], plot_step)
    
    netdir = "%s/net_potential" % savedir
    paramdir = "%s/individual_potentials" % savedir
    if not os.path.isdir(netdir):
        os.mkdir(netdir)
    if not os.path.isdir(paramdir):
        os.mkdir(paramdir)
        
    individual_scaled_values = []
    individual_scaled_potentials = []
    label = []
    for j in range(len(new.params)):
            individual_scaled_values.append(plot_values)
            individual_scaled_potentials.append(new.potential_functions[j](plot_values))
            label.append("p-%d"%j)
    os.chdir(savedir)
    plot_simple(individual_scaled_values, individual_scaled_potentials, label, "all-potentials", "position", "potential energy", axis=axis, marker_type="", use_legend=False)    
    for i in iters:
        os.chdir(cwd)
        individual_scaled_values = []
        individual_scaled_potentials = []
        label = []
        new.params = np.loadtxt("iteration_%d/params"%i)
        potential_values = new.net_potential(plot_values)
        os.chdir(netdir)
        plot_one_simple(plot_values, potential_values, "net-potential_iteration-%d"%i, "position", "potential energy", axis=axis)
        os.chdir(paramdir)
        for j in range(len(new.params)):
            individual_scaled_values.append(plot_values)
            individual_scaled_potentials.append(new.params[j]*new.potential_functions[j](plot_values))
            label.append("p-%d"%j)
        plot_simple(individual_scaled_values, individual_scaled_potentials, label, "potentials_iteration-%d"%i, "position", "potential energy", axis=axis, marker_type="", use_legend=False)
    
    os.chdir(cwd)
        
def run_histogram(name, savedir, iters, reference_set, nbins, histrange, axis, title):
    #plot the histogram of several different iterations
    cwd = os.getcwd()
    
    all_hist = []
    all_bincenters = []
    label = []
    #load the reference data set, assumed binned at spacing of 0.1 from -20 to 20
    refhist = np.loadtxt(reference_set)
    centershist = np.arange(-19.95, 20.05, 0.1)
    all_hist.append(refhist)
    all_bincenters.append(centershist)
    label.append("FRET")
    
    for i in iters:
        print "Loading iteration %d" % i
        x = np.loadtxt("iteration_%d/position.dat" %i)
        hist, bincenters, slices = compute.hist_x_histogram(x, nbins=nbins, histrange=histrange)
        all_hist.append(hist)
        all_bincenters.append(bincenters)
        label.append("I-%d"%i)
    
    os.chdir(savedir)
    plot_simple(all_bincenters, all_hist, label, "Iter%d-%d_%s"%(np.min(iters), np.max(iters), title), "Position", "Probability", axis=axis) 

def plot_one_simple(x, y, title, xaxis_label, yaxis_label, axis=None):
    #general plot function for a single line with a title and axis labeled
    plt.figure()
    plt.plot(x, y, alpha=1, linewidth=3, linestyle="-", color="k")
    if not axis == None:
        plt.axis(axis)
        
    plt.xlabel(xaxis_label, fontsize=25)
    plt.ylabel(yaxis_label,fontsize=25)
    plt.title(title, fontsize=25)
    
    plt.savefig("%s.png"%title)
    
def plot_simple(x, y, label, title, xaxis_label, yaxis_label, axis=None, marker_type="o", use_legend=True):
    """plot_simple is for a simple x-y plot with the dots connected.  """    
    
    #specify generic color order for use
    colors = ["b","g","r","c","m","y"]
    linetype = ["-", "--",":"]
    
    plt.figure()
    maxvalue = 0.0
    maxcenter = 0.0
    mincenter = 0.0
    
    #plot it!
    for i in range(np.shape(label)[0]):
        if i == 0:
            plt.plot(x[i], y[i], alpha=1, linewidth=3, markersize=8, linestyle=linetype[0], color="k", label="%s"%label[i], marker=marker_type)
        else:
            plt.plot(x[i], y[i], alpha=0.6, linewidth=2, markersize=7, linestyle=linetype[(i%6)%3], color=colors[i%6], label="%s"%label[i], marker=marker_type)
        maxvalue = return_max(maxvalue, np.max(y[i]))
        maxcenter = return_max(maxcenter, np.max(x[i]))
        mincenter = return_min(mincenter, np.min(x[i]))
    
    #specify the axis size, labels, etc.
    if axis == None:
        plt.axis([int(mincenter*1.5)-1,int(maxcenter*1.5)+1, 0, maxvalue*1.2],fontsize=20) 
    else:
        plt.axis(axis)
    if use_legend:
        plt.legend()
    plt.xlabel(xaxis_label, fontsize=25)
    plt.ylabel(yaxis_label,fontsize=25)
    plt.title(title, fontsize=25)
    
    plt.savefig("%s.png"%title)
    
    #plt.show()
#useful functions for plotting methods
def return_max(cmax, nmax):
    if cmax < nmax:
        return nmax
    else:
        return cmax

def return_min(cmin, nmin):
    if cmin < nmin:
        return cmin
    else:
        return nmin
        
###Argument functions below###        
def sanitize_args(args):
    #clean the arguments
    cwd = os.getcwd()
    
    #make savedir if not found, go to it, and save the full path for savedir
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    os.chdir(args.savedir)
    args.savedir = os.getcwd()
    
    os.chdir(cwd)
        
    return args
    
def get_args():
    #load arguments from command line
    parser = argparse.ArgumentParser(description="parent set of parameters", add_help=False)
    parser.add_argument("--name", default="simple", type=str, help="specify name of model .ini file. Default='simple'")
    parser.add_argument("--savedir", default="%s/analysis"%os.getcwd(), type=str, help="specify save dir. Default='analysis'")
    parser.add_argument("--title", default="plot", type=str, help="specify save name and title for the plot. Default='plot'")
    parser.add_argument("--axis", nargs=4, type=float, default=None, help="specify the axis for the plot. Default=None")
    parser.add_argument("--range", nargs=2, type=int, default=[0,1], help="specify the range of iterations to use. Default=[0,1]")
    parser.add_argument("--iter_step", type=int, default=1, help="specify the increment for iterations to use. Default=1")
    parser.add_argument("--iters", nargs="+", type=int, default=None, help="specify the specific iterations to use in some irregular increment and range. Overrides --range and --iter_step flags. Default=None")
    ##real parser
    par = argparse.ArgumentParser(description="Specify hist")
    sub = par.add_subparsers(dest="step")
    
    hist_sub = sub.add_parser("hist", parents=[parser], help="for specifying histogramming an iteration range")
    hist_sub.add_argument("--reference_set", type=str, default="ideal_set.dat", help="specify the reference data set for the histogram. Default='ideal_set.dat'")
    hist_sub.add_argument("--nbins", type=int, default=400, help="specify the number of histogram bins to use. Default=400")
    hist_sub.add_argument("--histrange", nargs=2, type=float, default=[-20.0, 20.0], help="specify the range to histogram over. Default=[-20.0,20.0]")
    
    
    energy_sub = sub.add_parser("energy", parents=[parser], help="for specifying the plotting of potential energies")
    energy_sub.add_argument("--plot_range", nargs=2, type=float, default=[-20.0,20.0], help="specify the range to plot the potential over. Default=[-20.0,20.0]")
    energy_sub.add_argument("--plot_step", default=0.1, type=float, help="specify the spacing between plotted energy levels. Default=0.1")
    
    args = par.parse_args()
    args = sanitize_args(args)
    
    return args
    
if __name__ == "__main__":
    args = get_args()
    print "Starting histogram analysis"
    if args.iters == None:
        iters = np.arange(args.range[0], args.range[1]+1, args.iter_step)
    else:
        iters = args.iters
    if args.step == "hist":
        run_histogram(args.name, args.savedir, iters, args.reference_set, args.nbins, args.histrange, args.axis, args.title)
    elif args.step == "energy":
        run_energy_plotting(args.name, args.savedir, iters, args.plot_range, args.plot_step, args.axis, args.title)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
