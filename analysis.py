""" file for running analysis scripts on a finished set of data"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import langevin_model.compute as compute

def run_histogram(name, savedir, iters, reference_set, nbins, histrange, axis, title):
    cwd = os.getcwd()
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
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
    
def plot_simple(x, y, label, title, xaxis_label, yaxis_label, axis=None):
    """plot_simple is for a simple x-y plot with the dots connected.  """    
    
    #specify generic color order for use
    colors = ["k","b","g","r","c","m","y","b","g","r","c","m","y","b","g","r","c","m","y","b","g","r","c","m","y"]
    linetype = ["-", "--",":"]
    
    plt.figure()
    maxvalue = 0.0
    maxcenter = 0.0
    mincenter = 0.0
    
    #plot it!
    for i in range(np.shape(label)[0]):
        if i == 0:
            plt.plot(x[i], y[i], alpha=1, linewidth=3, markersize=8, linestyle=linetype[i/6], color=colors[i], label="%s"%label[i], marker="o")
        else:
            plt.plot(x[i], y[i], alpha=0.6, linewidth=2, markersize=7, linestyle=linetype[i/6], color=colors[i], label="%s"%label[i], marker="o")
        maxvalue = return_max(maxvalue, np.max(y[i]))
        maxcenter = return_max(maxcenter, np.max(x[i]))
        mincenter = return_min(mincenter, np.min(x[i]))
    
    #specify the axis size, labels, etc.
    if axis == None:
        plt.axis([int(mincenter*1.5)-1,int(maxcenter*1.5)+1, 0, maxvalue*1.2],fontsize=20) 
    else:
        plt.axis(axis)
    plt.legend()
    plt.xlabel(xaxis_label, fontsize=25)
    plt.ylabel(yaxis_label,fontsize=25)
    plt.title(title, fontsize=25)
    
    plt.savefig("%s.png"%title)
    
    #plt.show()

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

def get_args():
    parser = argparse.ArgumentParser(description="parent set of parameters", add_help=False)
    parser.add_argument("--name", default="simple", type=str)
    parser.add_argument("--savedir", default="%s/analysis"%os.getcwd(), type=str)
    parser.add_argument("--title", default="plot", type=str)
    ##real parser
    par = argparse.ArgumentParser(description="Specify hist")
    sub = par.add_subparsers(dest="step")
    
    hist_sub = sub.add_parser("hist", parents=[parser], help="for specifying histogramming an iteration range")
    hist_sub.add_argument("--range", nargs=2, type=int, default=[0,1])
    hist_sub.add_argument("--iter_step", type=int, default=1)
    hist_sub.add_argument("--iters", nargs="+", type=int, default=None)
    hist_sub.add_argument("--reference_set", type=str, default="ideal_set.dat")
    hist_sub.add_argument("--nbins", type=int, default=400)
    hist_sub.add_argument("--histrange", nargs=2, type=float, default=[-20.0, 20.0])
    hist_sub.add_argument("--axis", nargs=4, type=float, default=None)
    args = par.parse_args()
    
    return args
    
if __name__ == "__main__":
    args = get_args()
    if args.step == "hist":
        print "Starting histogram analysis"
        if args.iters == None:
            iters = np.arange(args.range[0], args.range[1]+1, args.iter_step)
        else:
            iters = args.iters
        run_histogram(args.name, args.savedir, iters, args.reference_set, args.nbins, args.histrange, args.axis, args.title)
        
