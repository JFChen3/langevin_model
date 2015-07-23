"""
Methods for working with a langevin dynamics model
"""
import numpy as np
import matplotlib.pyplot as plt

def run_langevin(model, nsteps=1000000, dt=0.01, kbT=1.0, mu=1.0, gamma=1.0):
    sigma = np.sqrt(2.*kbT*gamma/mu)

    x = np.zeros(nsteps,float)
    v = np.zeros(nsteps,float)

    xi = np.random.normal(0,1,nsteps)
    theta = np.random.normal(0,1,nsteps)

    f = model.net_force
    
    for i in xrange(nsteps - 1):
        A = 0.5*(dt**2)*(f(x[i]) - gamma*v[i]) + sigma*(dt**(3./2))*(0.5*xi[i] + np.sqrt(3./36.)*theta[i])
        try:
            x[i + 1] = x[i] + dt*v[i] + A
            v[i + 1] = v[i] + 0.5*dt*(f(x[i + 1]) + f(x[i])) - dt*gamma*v[i] + sigma*np.sqrt(dt)*xi[i] - gamma*A
        except:
            print x[i]
            print dt
            print v[i]
            print A


    return x, v
    

def plot_x_inverted(model, x):
    plt.figure()
    plt.plot(x)
    plt.savefig("trace.png")
    
    plt.figure()
    x -= np.mean(x)
    n,bins = np.histogram(x,bins=100)
    
    bin_avg = 0.5*(bins[1:] + bins[:-1])
    pmf = -np.log(n)
    pmf -= min(pmf)
    realstuff = model.net_potential(bin_avg)
    realstuff -= np.min(model.net_potential(bin_avg))
    plt.plot(bin_avg,pmf)
    plt.plot(bin_avg,realstuff,'g',lw=2)
    plt.savefig("energy_levels.png")
