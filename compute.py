"""
Methods for working with a langevin dynamics model
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def run_langevin(model, nsteps=1000000, dt=0.01, kbT=1.0, mu=1.0, gamma=1.0, initial_x=2.0):
    #runs a langevin dynamics at the specified temperatures and other factors
    sigma = np.sqrt(2.*kbT*gamma/mu)

    x = np.zeros(nsteps,float)
    x[0] = initial_x
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


def compute_jacobian(model, x, slices, sim_feature, nbins, spacing):
    #get all functions values for each frame:
    fenergy = np.zeros((np.shape(x)[0], model.number_fit_parameters))
    fenergy_index = 0
    jac_indices = slices - 1
    for i in model.fit_indices:
        fenergy[:, fenergy_index] = model.potential_functions[i](x)
        fenergy_index += 1

    #make jacobian
    Jacobian = np.zeros((nbins, model.number_fit_parameters))
    jac_count = np.zeros(nbins)
    for idx, bin_location in enumerate(jac_indices):
        Jacobian[bin_location, :] += fenergy[idx,:]
        jac_count[bin_location] += 1

    favg = np.sum(Jacobian, axis=0) / np.shape(x)[0]

    for i in range(nbins):
        if not jac_count[i] == 0:
            Jacobian[i,:] /= jac_count[i]
            Jacobian[i,:] -= favg
            Jacobian[i,:] *= sim_feature[i]

    
    return Jacobian*(-1.0)
    
        
def fit_jacobian():
    #loads in the expected files and outputs a set of fitted models at different truncations
    target_feature = np.loadtxt("target_feature.dat")
    sim_feature = np.loadtxt("sim_feature.dat")
    Jacobian = np.loadtxt("Jacobian.dat")
    J = Jacobian
           
    ## Normalize the target step. 
    df = target_feature - sim_feature
    print df
    u,s,v = np.linalg.svd(J)
    temp  = list(0.5*(s[:-1] + s[1:])) + [0.0]
    temp.reverse()
    Lambdas = np.array(temp)
    
    nrm_soln = []
    nrm_resd = []
    condition_number = []
    solutions = []
    for i in range(len(Lambdas)):
        S = np.zeros(J.shape) 
        Lambda = Lambdas[i] 
        s_use = s[s > Lambda]
        S[np.diag_indices(len(s_use))] = 1./s_use
        J_pinv = np.dot(v.T,np.dot(S.T,u.T))
        x_soln = np.dot(J_pinv,df)  ## particular

        residual = np.dot(J,x_soln) - df

        nrm_soln.append(np.linalg.norm(x_soln))
        nrm_resd.append(np.linalg.norm(residual))
        solutions.append(x_soln)

        J_use = np.dot(v.T,np.dot(S.T,u.T))     
        cond_num = np.linalg.norm(J_use)*np.linalg.norm(J_pinv)
        condition_number.append(cond_num)    
    #save
    for i in range(len(solutions)):
        np.savetxt("xp_%d.dat"%i, solutions[i])
    np.savetxt("singular_values.dat", s)
    np.savetxt("lambdas.dat", Lambdas)
    
    
##plotting functions    
def plot_x_histogram(x, title, nbins=400, histrange=(-20.0,20.0)):
    #plot a histogram of the position distribution. Also returns the histogram information and slices for analysis
    plt.figure()
    hist, bincenters, slices = hist_x_histogram(x, nbins=nbins, histrange=histrange)
    plt.plot(bincenters, hist, 'ok')
    plt.savefig("histogram_position.png")
    np.savetxt("%s.dat"%title, np.array([bincenters, hist]).transpose())
    
    return hist, bincenters, slices, (float(histrange[1]-histrange[0]) / float(nbins))
    
def hist_x_histogram(x, nbins=400, histrange=(-20.0,20.0)):
    #actually perform the histogramming and returns the histogram information
    hist, edges, slices = stats.binned_statistic(x, np.ones(np.shape(x)[0]), statistic="sum", range=[histrange], bins=nbins)
    hist = hist/(np.sum(hist) * (float(histrange[1]-histrange[0]) / float(nbins)))
    bincenters = 0.5*(edges[1:] + edges[:-1])
    
    return hist, bincenters, slices
         
def plot_x_inverted(model, x):
    #Plot the free energy of the system in x.
    plt.figure()
    plt.plot(x)
    plt.savefig("trace.png")
    
    plt.figure()
    #x -= np.mean(x)
    n,bins = np.histogram(x,bins=100)
    
    bin_avg = 0.5*(bins[1:] + bins[:-1])
    pmf = -np.log(n)
    pmf -= min(pmf)
    realstuff = model.net_potential(bin_avg)
    realstuff -= np.min(model.net_potential(bin_avg))
    plt.plot(bin_avg,pmf)
    plt.plot(bin_avg,realstuff,'g',lw=2)
    plt.savefig("energy_levels.png")
