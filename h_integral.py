import numpy as np
import struct
import pdb
import matplotlib.pyplot as plt
import os
from input import *

rjup = 7.1492e9
kboltz = 1.38064852e-16
amu = 1.660539040e-24
g = 977
r0 = 1.79*rjup
m = 2.4*amu
m_water = 18*amu
Av = 6.02214076e23

opacity_path = os.environ['HOME'] + '/Desktop/PhD/DACE/1H2-16O__POKAZATEL_e2b/'
pressure_array = 10**np.array([-8, -7.66, -7.33, -7, -6.66, -6.33, -6, -5.66, -5.33, -5, -4.66, -4.33, -4, -3.66,
                               -3.33, -3, -2.66, -2.33, -2, -1.66, -1.33, -1, -0.66, -0.33, 0, 0.33, 0.66, 1.0])

temperature = 1000
scale_height = kboltz*temperature/m/g

res = 2     # resolution for the opacities
step_size = int(res/0.01)

min_wavenumber = int(1e4/wavelength_bins[-1])
max_wavenumber = int(1e4/wavelength_bins[0])

epsilon = 0.000001
p0 = 10 + epsilon   # add some tiny value to p0 to avoid infinities in integration



def load_opacity(temperature, pressure):

    temp_str = str(temperature).zfill(5)     # temperature as in opacity filename

    pressure_load = int(np.log10(pressure) * 100)
    if pressure_load < 0:
        pressure_str = 'n' + str(abs(pressure_load)).rjust(3, '0')  # pressure as in opacity filename
    else:
        pressure_str = 'p' + str(abs(pressure_load)).rjust(3, '0')

    filename = 'Out_00000_42000_'+temp_str+'_'+pressure_str+'.bin'

    data = []
    with open(opacity_path + filename, "rb") as f:
        byte = f.read(4)
        while byte:
            data.extend(struct.unpack("f", byte))
            byte = f.read(4)

    data = np.array(data[(min_wavenumber)*100:(max_wavenumber)*100:step_size])

    return data



def tau(temperature, pmin, p0):
    # Compute tau for all pressures

    pressure_array_pmin = pressure_array[np.where(pressure_array==pmin)[0][0]:] # remove everything below pmin

    opacity_line_length = int((max_wavenumber-min_wavenumber)/res)
    integrand_grid = np.zeros((len(pressure_array_pmin), opacity_line_length))  # This will be the integrand
                                                                                # we will integrate over pressure,
                                                                                # for one temperature, for all wavelengths
    
    # Load integrands for all pressures
    for i,p in enumerate(pressure_array_pmin):
        opacity = load_opacity(temperature, p)   # load opacity for this temperature
        opacity = opacity[1:]
        integrand_grid[i] = opacity/np.sqrt(np.log(p0/p))   # compute kappa/sqrt(ln(P0/P))

    kappa_grid = np.zeros((len(pressure_array_pmin), opacity_line_length))

    factor = np.sqrt(2*scale_height*r0)/(kboltz*temperature)

    # Integrate for each pressure p, from pmin to p
    for i, p in enumerate(pressure_array_pmin):

        pressure_sliced = pressure_array_pmin[:i+1]     # pass in pressure values and integrand values for all pressures
        integrand_grid_sliced = integrand_grid[:i+1]    # below p, above pmin

        integral_value = np.trapz(integrand_grid_sliced, pressure_sliced, axis=0)   # calculate integral using trapezoid approximation

        kappa_grid[i] = factor*integral_value

    tau_grid = kappa_grid*m_water     # convert opacity to cross-section

    return pressure_array_pmin, tau_grid

pressure_values, tau_values = tau(temperature, 1e-6, p0)



def h(temperature, pmin, p0):

    pressure_array_pmin = pressure_array[np.where(pressure_array==pmin)[0][0]:np.where(pressure_array==pressure_array[-1])[0][0]] # remove everything below pmin

    opacity_line_length = int((max_wavenumber-min_wavenumber)/res)
    integrand_grid = np.zeros((len(pressure_array_pmin), opacity_line_length))  # This will be the integrand
                                                                                # we will integrate over pressure,
                                                                                # for one temperature, for all wavelengths
    # Load integrands for all pressures
    for i, p in enumerate(pressure_array_pmin):
       
        integrand_grid[i] = (1 - np.exp(-tau_values[i]))/p * (r0 + scale_height*np.log(p0/p))

    h_grid = np.zeros((len(pressure_array_pmin), opacity_line_length))

    factor = scale_height/r0

    # Integrate for each pressure p, from pmin to p
    for i, p in enumerate(pressure_array_pmin):

        pressure_sliced = pressure_array_pmin[:i+1]     # pass in pressure values and integrand values for all pressures
        integrand_grid_sliced = integrand_grid[:i+1]    # below p, above pmin

        integral_value = np.trapz(integrand_grid_sliced, pressure_sliced, axis=0)   # calculate integral using trapezoid approximation

        h_grid[i] = factor*integral_value

    return pressure_array_pmin, h_grid

pressure_values, h_values = h(temperature, 1e-6, p0)




def init_plotting(x=20,y=20):
    plt.rcParams['figure.figsize'] = (20,15)
    plt.rcParams['font.size'] = 30
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 12
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 12
    plt.rcParams['ytick.minor.size'] = 8
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.labelcolor'] = 'k'
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams['xtick.color'] = 'k'
    plt.rcParams['ytick.color'] = 'k'
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    plt.rcParams['axes.labelpad'] = 15
init_plotting()



fig, ax = plt.subplots()




cmap = plt.get_cmap('rainbow')

norm = plt.Normalize(vmin=0, vmax=int((1e4/wavelength_bins[0]-1e4/wavelength_bins[-1])/2))

for i in range(0, int((1e4/wavelength_bins[0]-1e4/wavelength_bins[-1])/2), 100):

    x = np.log10(pressure_values)
    y = np.log10(h_values[:,i])


#   print(1.7e4/(1.7*res*i + 1e4))
    rgba = cmap(norm(i))
    plot = plt.plot(x, y, c=rgba, lw=3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_clim(int(1e4/wavelength_bins[-1]), int(1e4/wavelength_bins[0]))
cbar = plt.colorbar(sm)

#tick_locator = ticker.MaxNLocator(nbins=10)
#cbar.locator = tick_locator
#cbar.update_ticks()





#cbar.ax.set_yticklabels(['0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7'])



cbar.set_label('Wavelength ($\mu$m)')



ax.set_xlabel(r'log P (bar)')
ax.set_ylabel(r'log h')

fig.savefig('h_integral.png', bbox_inches='tight')
