'''
TO RUN:
>>> import generate_events from roman-pipeline
>>> file_path = 'path_to_file_storage_location'
>>> psbl = generate_events(gap_size = 110, number = 500000)
>>> psbl.to_pickle(file_path)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform
import pandas as pd
import random
import time
from scipy.stats import powerlaw
from scipy import stats
# import tdpy
import math
from scipy import interpolate


from pyLIMA import microlsimulator, microltoolbox

np.random.seed(42)

sources = np.loadtxt(r"sources.txt")
lens = np.loadtxt(r"lenses.txt")
print('Loaded source and lens parameters')
lens = lens[lens[:,10] > 10]

precision = pd.read_csv(r"roman_gbtds_noise_model.txt")
interp = interpolate.interp1d(precision['F146'], precision['sigma_photon'], kind = 'linear')

def get_theta_E(s,l):
    G = 6.6743e-11 
    pi = 3.141
    c = 299792458
    d_s = 10**((s[7] + 5)/5)
    d_l = 10**((l[7] + 5)/5) 
    d_ls = d_s - d_l
    solar_mass_to_kg = 1.989e+30
    parsec_to_meter = 3.086e16
    solar_mass = l[-1]
    theta_E = np.sqrt((4*G*solar_mass*solar_mass_to_kg)/(c**2) * (d_ls*parsec_to_meter/(d_l*parsec_to_meter*d_s*parsec_to_meter))) * (3600 * 180)/pi
    return theta_E

def get_rho(s, l):
    sigma = 5.670367e-8
    pi = 3.141
    lum = 10**s[4]
    temp = 10**s[5]
    d_s = 10**((s[7] + 5)/5)
    solar_lum_to_watt = 3.826e26
    parsec_to_meter = 3.086e16
    rho = ((np.sqrt(lum*solar_lum_to_watt/(4*pi*sigma*temp**4)) / (d_s*parsec_to_meter)) * ((3600 * 180) / pi)) / get_theta_E(s, l)
    return rho

def logs(s,l):
    pi = 3.141
    d_l = 10**((l[7] + 5)/5)
    au_to_parsec = 206300
    sem = semimajor_axis() 
    sep = (sem / au_to_parsec / d_l) * (3600 * 180) / pi / get_theta_E(s, l)
    return [np.log10(sep), sem]

def logq_exp(l, mass):
    return np.log10(mass/l[-1])

def t_E():
    return np.random.lognormal(3.1, 1, 1)[0]

def logq_stellar(l):
    return np.log10(l[9])

def al():
    alpha = np.random.uniform(-np.pi, np.pi) # theta_e
    return alpha

def semimajor_axis():
    temp = loguniform.rvs(0.3, 100)
    return temp
        
def eval_pdf(n = 5000):
    mass_cut = 1.5614e-5
    sm = 0.000285259
    frequencies = np.zeros(n)
    grid = np.logspace(np.log10(9.00819e-8), np.log10(0.0300273), n)
    num = len(grid[grid < mass_cut])
    frequencies[:num] = 2
    upper = grid[grid >= mass_cut]
    frequencies[num:] = 0.24*(upper/sm)**(-0.73)
    frequencies /= np.sum(frequencies)
    return grid, frequencies
  
def exoplanet_mass(pdf):
    return np.random.choice(pdf[0], p = pdf[1])

pdf = eval_pdf(1000000)

def create_event_gaps(gap_size, t_start=2457300, separation=72, long=838):
    test_event = microlsimulator.simulate_a_microlensing_event(name ='Test 1', 
                                                                    ra=270, dec=-30)    
    roman1 = microlsimulator.simulate_a_telescope('Roman1',test_event, t_start,t_start+separation,0.25, 'Space','W149',
                                                      uniform_sampling=True)
    roman2 = microlsimulator.simulate_a_telescope('Roman1',test_event,t_start+separation + gap_size, t_start+2*separation + gap_size,0.25, 'Space','W149',
                                                      uniform_sampling=True)
    roman3 = microlsimulator.simulate_a_telescope('Roman1',test_event,t_start+2*separation + 2*gap_size, t_start+3*separation + 2*gap_size,0.25, 'Space','W149',
                                                      uniform_sampling=True)
    roman4 = microlsimulator.simulate_a_telescope('Roman1',test_event,t_start+3*separation + 2*gap_size+long, t_start+4*separation + 2*gap_size+long,0.25, 'Space','W149',
                                                      uniform_sampling=True)
    roman5 = microlsimulator.simulate_a_telescope('Roman1',test_event,t_start+4*separation + 3*gap_size+long, t_start+5*separation + 3*gap_size+long,0.25, 'Space','W149',
                                                      uniform_sampling=True)
    roman6 = microlsimulator.simulate_a_telescope('Roman1',test_event,t_start+5*separation + 4*gap_size+long, t_start+6*separation + 4*gap_size+long,0.25, 'Space','W149',
                                                      uniform_sampling=True)

    roman_tot = microlsimulator.simulate_a_telescope('Roman1',test_event,t_start+10*separation, t_start+11*separation,0.25, 'Space','W149',
                                                      uniform_sampling=True)
    
    roman_tot.lightcurve_flux = np.r_[roman1.lightcurve_flux,roman2.lightcurve_flux,roman3.lightcurve_flux,roman4.lightcurve_flux, roman5.lightcurve_flux,roman6.lightcurve_flux]
    roman_tot.lightcurve_magnitude = np.r_[roman1.lightcurve_magnitude,roman2.lightcurve_magnitude,roman3.lightcurve_magnitude,roman4.lightcurve_magnitude, roman5.lightcurve_magnitude,roman6.lightcurve_magnitude]

    test_event.telescopes.append(roman_tot)
    return test_event

exp_masses = np.random.choice(pdf[0], p = pdf[1], size = 5000000)
te = np.random.lognormal(3.1, 1, 1000000)

def generate_light_curve(modeltype='PSPL', index = None, sources=None, lens=None, gap_size=110, t_start=2457300, separation=72, long=838):
    event = create_event_gaps(gap_size, t_start, separation, long)
    if modeltype == 'Red noise':
        model = microlsimulator.simulate_a_microlensing_model(event, model='PSPL')
    else:
        model = microlsimulator.simulate_a_microlensing_model(event, model=modeltype)
            
    parameters = microlsimulator.simulate_microlensing_model_parameters(model)
    parameters[0] = np.random.uniform(t_start, t_start+6*separation + 4*gap_size + long)
    parameters[1] = np.random.uniform(0,3)
    parameters[2] = te[index]
    if parameters[0] - parameters[2] > (t_start+3*separation + 2*gap_size) and parameters[0] + parameters[2] \
        < (t_start+3*separation + 2*gap_size + long):
        return [np.nan] * 15
    a = np.random.uniform(0,1)
    b = microltoolbox.magnitude_to_flux(np.random.uniform(15,19))
    flux_parameters = [b,a]
    parameters += flux_parameters
    params_list = []
    
    if modeltype == 'Red noise':
        parameters[0] = 0
        params_list = parameters[:3]+[np.nan]*7
    elif modeltype == 'PSPL':
        params_list = parameters[:3]+[np.nan]*7
        params_list[6] = lens[index][-1]
    elif modeltype == 'FSBL':
        parameters = parameters[:3]
        rho = get_rho(sources[index], lens[index])
        mass_ratio = logq_exp(lens[index], exp_masses[index])
        b = 1
        lgs = logs(sources[index],lens[index])
        host_mass = lens[index][-1]
        parameters = parameters + [rho, lgs[0],mass_ratio,al()]
        params_list = parameters

    params_list += flux_parameters
    pyLIMA_parameters = model.compute_pyLIMA_parameters(parameters)
    flux_model = model.compute_the_microlensing_model(event.telescopes[-1], pyLIMA_parameters)
    magnitude = microltoolbox.flux_to_magnitude(flux_model[0]) 
    w149_mag = event.telescopes[-1].lightcurve_magnitude
    w149_mag[:,1] = magnitude
    w149_mag[:, 2] = interp(w149_mag[:,1])
    w149_mag[:,1] = w149_mag[:,1] + np.random.randn(w149_mag[:,1].shape[0]) * w149_mag[:, 2]
    
    if modeltype == 'FSBL':
        model_s = microlsimulator.simulate_a_microlensing_model(event, 'FSPL')
        pyLIMA_parameters_s = model_s.compute_pyLIMA_parameters(parameters[:4] + flux_parameters)
        flux_model_s = model_s.compute_the_microlensing_model(event.telescopes[0], pyLIMA_parameters_s)
        metric = np.sum(((magnitude - microltoolbox.flux_to_magnitude(flux_model_s[0]))/w149_mag[:,2])**2)/len(w149_mag[:,0])
        params_list = params_list[:7] + [host_mass] + [b] + [metric] + [lgs[1]] + params_list[7:]
#     print(params_list)
    return [params_list] + [w149_mag]

def generate_events(gap_size, number=50000):
    psbl = pd.DataFrame(columns = ['t_0','u_0','t_E','rho', 'logs','logq','alpha','host_mass','binary_type','detectability','semimajor_axis',
                                        'fs_w149','g_w149','w149_mag','t1_time','index'])
    for i in range(0, number):
        t = True
        while t == True:
            try:
                t1 = time.time()
                ind = i
                lc = generate_light_curve('FSBL', index = ind, sources = sources, lens = lens,
                                          gap_size=gap_size, t_start=2457300, separation=72, long=838)
                length = len(psbl)
                t2 = time.time()
                # print(lc)
                psbl.loc[length] = lc[0] + lc[1:] + [t2-t1] + [ind]
                # print(psbl)
                t = False
            except Exception:
                time.sleep(0.0000001)
        if i % 1000 == 0:
            print(f"On light curve {i}") 
                
    return psbl

# psbl = generate_events(110, 500000)
# psbl.to_pickle('test')
