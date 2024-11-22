# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:41:21 2024

@author: abedn
"""

#%% THE TIME RESOLUTION

#importing necessary libaries
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

omega = 50 #PHz Angular frequency window
dt = 2 * np.pi / omega #fs The time resolution
print(f'The time resolution dt = {dt:0.3f} fs')

#%% WIDTH OF THE TIME WINDOW

N = 2**16
dw = omega / N #PHz Angular frequency resolution
omega_array = np.arange(-omega/2, omega/2, dw) #Angular frequency array
T = 2 * np.pi / dw
print(f'The time window T = {T: 0.1f} fs')

#%% THE SPECTRA FIELD

c = 299.792458 #nm/fs speed of light
wl = 800 #nm
w_0 = 2 * np.pi* c / wl # carrier angular frequency
print(f'The  carrier anglar frequency w_0 = {w_0: 0.3f} PHz')
print(f'The anglar frequency resolution dw = {dw: 0.6f} PHz')

T_0 = 10 #fs
def spectra_field(t, w, w_c):
    amplitude = np.sqrt(np.pi) * t / np.sqrt(8 * np.log(2))
    exp_term = np.exp(-1 * (w - w_c)**2 * t**2 / (8 * np.log(2)))
    field_w = amplitude * exp_term
    return field_w
E_w = spectra_field(T_0, omega_array, w_0)

phase = np.angle(E_w)

#%% THE SPECTRA AMPLITUDE

plt.figure(1)
plt.plot(omega_array, np.abs(E_w), 'r-', label = 'Amplitude spectra field')
plt.xlim(1.8, 3)
plt.ylim(0, 8)
plt.xlabel('Angular frequency (PHZ)')
plt.ylabel('spectra amplitude (a.u)')
plt.legend()
plt.grid()
plt.show()

#%% THE PHASE

plt.figure(2)
plt.plot(omega_array, phase, 'b-', label = 'spectra phase')
plt.xlim(1.8, 3)
plt.ylim(-1, 1)
plt.xlabel('Angular frequency (PHZ)')
plt.ylabel('spectra phase(rad)')
plt.legend()
plt.grid()
plt.show()

#%% SPECTRAL INTENSITY
I_w = np.abs(E_w)**2

plt.figure(3)
plt.plot(omega_array, I_w/np.max(I_w), 'r-', label = 'Spectra Intensity')
plt.xlim(1.8, 3)
plt.xlabel('Angular frequency (PHZ)')
plt.ylabel('spectral intensity (a.u)')
plt.legend()
plt.grid()
plt.show()

#%% INVERSE FOURIER TRANSFORM
E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_w))) * (omega/(2 * np.pi))

t = np.arange(-T/2, T/2, dt)
#Function for the analytical field
def Electric_field_analytical(t, t0, w_c):
    first_exp = np.exp(-2 * np.log(2) * (t/t0)**2)
    second_exp = np.exp(1j * w_c * t)
    analytical = 0.5 * first_exp * second_exp
    return analytical
E_analytical = Electric_field_analytical(t, T_0, w_0)

#The plots of FT and analytical
plt.figure(4)
plt.plot(t, np.real(E_t), 'r-', label = 'Numerical Electric field')
plt.plot(t, np.real(E_analytical), 'b--', label = 'Analytical Electric field')
plt.xlim(-20, 20)
plt.xlabel('Time (fs)')
plt.ylabel('Electric field (a.u)')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

#%% TEMPORAL INTENSITY OF A TL PULSE

I_t = np.abs(E_t)**2

plt.figure(5)
plt.plot(t, I_t/np.max(I_t), 'r-', label = 'Temporal Intensity') #Normalizing
plt.xlim(-20, 20)
plt.ylim(0,1)
plt.xlabel('Time (fs)')
plt.ylabel('Temporal intensity (a.u)')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

#%% TEMPORAL DURATION OF TL PULSE - (HOME WORK 2)

# Calculate the value of the intensity at half its maximum
I_t_half = np.amax(I_t) / 2

# Find the index of the maximum intensity
i_max = np.argmax(I_t)

# Initialize indices for finding the left and right half-max points of FWHM
i_0 = i_max
i_1 = i_max

# Simultaneous getting the half-max points with some conditions
while I_t[i_0] > I_t_half or I_t[i_1] > I_t_half:
    if I_t[i_0] > I_t_half:
        i_0 -= 1
    if I_t[i_1] > I_t_half:
        i_1 += 1

# Calculate the FWHM using the indices for the half-max points
FWHM = t[i_1] - t[i_0]
print(f'The FWHM of the temporal intensity = {FWHM:.6f} fs')


#%% Alternate method - LINEAR INTERPOLATION 

#the FWHM calculation using linear interpolation
t_0 = t[i_0] + ((I_t_half - I_t[i_0]) / (I_t[i_0 + 1] - I_t[i_0])) * dt #the left
t_1 = t[i_1] - ((I_t_half - I_t[i_1]) / (I_t[i_1 - 1] - I_t[i_1])) * dt #the right

# the  FWHM using the interpolated times
FWHM_1 = t_1 - t_0
print(f'The FWHM of the intensity (temporal) = {FWHM_1:.6f} fs')


""" Note: there is a slit difference between the methods (Use first method and create function for it)"""
#See lecture class practice code for other continue analysis and jotting
#%% FUNNCTION TO CALCULATE FWHM (Personal training)

