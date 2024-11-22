# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:04:38 2024

@author: abedn
"""

#This script is for assignment 3 femtosecond python course. To understand the effect of GDD, TOD, FOD and QOD
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

c = 299.792458 #nm/fs
wl = 800 #nm
N = 2**16 #the time window
omega = 50 #PHz
dt = 2 * np.pi / omega #time resolution
dw = omega / N #angular frequency resolution
T = N * dt #Time window
tau_0 = 6 #fs

w_0 = 2 * np.pi * c / wl #The carrier angular frequency
t = np.arange(-T/2, T/2, dt)
w = np.arange(-omega/2, omega/2, dw)
dom = w - w_0

#print all necessary parameters
print(f'The time resolution dt is {dt: 0.3f} fs and the time window T is {T: 0.1f} fs')
print(f'The anglar frequency resolution dw is {dw: 0.6f} PHz and the carrier angular frequency w_0 is {w_0: 0.3f} PHz')

#%% Getting all the necessary functions
#The complex spectrum
def complex_spectrum(t, w, w_c): #t is the time FWHM, w_c is the carrier angular freq
    amplitude = np.sqrt(np.pi) * t / np.sqrt(8 * np.log(2))
    exp_term = np.exp(-(w - w_c)**2 * t**2 / (8 * np.log(2)))
    spectrum = amplitude * exp_term
    return spectrum
#The analytical temporal spectrum
def analytical_field(t, t0, w_c): # t is the time array and t0 is d time FWHM
    exp_term = np.exp(-2 * np.log(2) * (t/t0)**2) * np.exp(1j * w_c * t)
    temporal_field = 0.5 * exp_term
    return temporal_field
#The spectral phase
def transfer_function(phi0, GD, GDD, T, F, Q, N): #the first... fifth order dispersion term (N is time window)
  A =  np.ones(N)
  dw = omega / N
  w = np.arange(-omega/2, omega/2, dw)
  dom = w - w_0
  spectral_phase = phi0 + (GD * dom) + (GDD/2 * dom**2) + ((T/6) * dom**3) + ((F/24) * dom**4) + ((Q/120) * dom**5)
  H = A * np.exp(-1j * spectral_phase)
  return H

#Inverse transform 
def inverse_fourier(spectrum): #spectrum is the spectral field
    temporal_field = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(spectrum))) * omega / (2 * np.pi)
    return temporal_field
#The output E field
#the output E field is a product of the transfer funnction and the input spectral field

#%%some needed parameters for use
#A  function to get the time index
def time_index(t,t1,t2): #t1 is the min and t2 is the max
    t_min = np.argmin(np.abs(t - t1))
    t_max = np.argmin(np.abs(t - t2))
    t0 = np.argmin(np.abs(t - 0))
    return t_min, t_max, t0

    
#A  function to get the spectral index
def spectral_index(w, w1, w2, w0): #w1 is the min and w2 is the max, w is the ang. freq array and w0 is the carrier ang frq
    i_min = np.argmin(np.abs(w - w1))
    i_max = np.argmin(np.abs(w - w2))
    i0 = np.argmin(np.abs(w - w0))
    return i_min, i_max, i0

#The output E field
#the output E field is a product of the transfer funnction and the input spectral field
#to get the output spectral field and the phase  
def output_spectral_field(H,E_in, i_0): #E_in is the input spectral domain here
    output_field = H * E_in
    output_phase = np.unwrap(-np.angle(output_field))
    n = round(output_phase[i_0] / (2 * np.pi))
    phase_out = output_phase - n * 2 * np.pi
    return output_field, phase_out

#To get the relative GD (spectral domain)
def relative_GD_out(output_phase, i_0, w):
    dom = w -w_0
    GD_out = np.gradient(output_phase, dom)
    relGD_out = GD_out - GD_out[i_0]
    return relGD_out
#In the temporal domain the phase and instantaneous frequency
def temporal_phase(t_field, t_0):
    phase_out_t = np.unwrap(np.angle(t_field))
    n = round(phase_out_t[t_0] / (2 * np.pi))
    phase_out_t = phase_out_t - n * 2 * np.pi
    w_inst = np.gradient(phase_out_t, dt) #instantaneous frequency
    return phase_out_t, w_inst

#%%FUNCTION TO FIND THE FWHM
    
def FWHM(I_t, t):
    dt = 2 * np.pi / omega #time resolution

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
    #the FWHM calculation using linear interpolation
    t_0 = t[i_0] + ((I_t_half - I_t[i_0]) / (I_t[i_0 + 1] - I_t[i_0])) * dt #the left
    t_1 = t[i_1] - ((I_t_half - I_t[i_1]) / (I_t[i_1 - 1] - I_t[i_1])) * dt #the right

    # the  FWHM using the interpolated times
    FWHM = t_1 - t_0  
    return FWHM, t_0, t_1


#%% FOD 10000

#the input field and other parameters
E8_w_in = complex_spectrum(tau_0, w, w_0)
H8_w = transfer_function(0, 0, 0, 0, 1000, 0, 2**16)
i_min = spectral_index(w, 1, 4, w_0)[0]
i_max = spectral_index(w, 1, 4, w_0)[1]
i0 = spectral_index(w, 1, 4, w_0)[2]
t_min = time_index(t, -50, 50)[0]
t_max = time_index(t, -50, 50)[1]
t0 = time_index(t, -50, 50)[2]

#The output - PLOTTED PARAMETERS
E8_w_out = output_spectral_field(H8_w, E8_w_in, i0)[0] #spectra domain
output_phase8 = output_spectral_field(H8_w, E8_w_in, i0)[1]
relGD8_out = relative_GD_out(output_phase8, i0, w) #The relative GD
E8_t_out = inverse_fourier(E8_w_out) #temporal domain
phase8_out_t = temporal_phase(E8_t_out, t0)[0]
w8_inst = temporal_phase(E8_t_out, t0)[1] #The instantaneous frequency
#%% Plots
#The  spectral field and phase
plt.figure(20)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E8_w_out[i_min:i_max]), 'r-', label = 'Output Electric spectrum')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], output_phase8[i_min:i_max], 'b-', label = 'Output Phase')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#The intensity and relative GD

plt.figure(21)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E8_w_out[i_min:i_max])**2, 'r-', label = 'Output Intensity')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Intensity (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], relGD8_out[i_min:i_max], 'b-', label = 'Relative GD')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Relative GD (fs)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Temporal intensity and phase

plt.figure(22)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E8_t_out[t_min:t_max]), 'r--', label = 'Output temporal field')
plt.plot(t[t_min:t_max], np.real(E8_t_out[t_min:t_max]), 'r-')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], phase8_out_t[t_min:t_max], 'b-', label = 'Output temporal Phase')
plt.xlabel('Time (fs)')
plt.ylabel('Temporal phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#the intensity and the instantaneous frequency

plt.figure(23)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E8_t_out[t_min:t_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], w8_inst[t_min:t_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

temporal_dur = FWHM(np.abs(E8_t_out[t_min:t_max])**2, t)[0]
t0 = FWHM(np.abs(E8_t_out[t_min:t_max])**2, t)[1]
t1 = FWHM(np.abs(E8_t_out[t_min:t_max])**2, t)[2]

print(f'The pulse  duration (FWHM) from the I(t) curve of FOD = 1000 is {temporal_dur: 0.2f} fs ')

#%%FOD -1000


#the input field and other parameters
E9_w_in = complex_spectrum(tau_0, w, w_0)
H9_w = transfer_function(0, 0, 0, 0, -1000, 0, 2**16)
i_min = spectral_index(w, 1, 4, w_0)[0]
i_max = spectral_index(w, 1, 4, w_0)[1]
i0 = spectral_index(w, 1, 4, w_0)[2]
t_min = time_index(t, -50, 50)[0]
t_max = time_index(t, -50, 50)[1]
t0 = time_index(t, -50, 50)[2]

#The output - PLOTTED PARAMETERS
E9_w_out = output_spectral_field(H9_w, E9_w_in, i0)[0] #spectra domain
output_phase9 = output_spectral_field(H9_w, E9_w_in, i0)[1]
relGD9_out = relative_GD_out(output_phase9, i0, w) #The relative GD
E9_t_out = inverse_fourier(E9_w_out) #temporal domain
phase9_out_t = temporal_phase(E9_t_out, t0)[0]
w9_inst = temporal_phase(E9_t_out, t0)[1] #The instantaneous frequency
#%% Plots
#The  spectral field and phase
plt.figure(24)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E9_w_out[i_min:i_max]), 'r-', label = 'Output Electric spectrum')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], output_phase9[i_min:i_max], 'b-', label = 'Output Phase')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#The intensity and relative GD

plt.figure(25)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E9_w_out[i_min:i_max])**2, 'r-', label = 'Output Intensity')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Intensity (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], relGD9_out[i_min:i_max], 'b-', label = 'Relative GD')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Relative GD (fs)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Temporal intensity and phase

plt.figure(26)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E9_t_out[t_min:t_max]), 'r--', label = 'Output temporal field')
plt.plot(t[t_min:t_max], np.real(E9_t_out[t_min:t_max]), 'r-')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], phase9_out_t[t_min:t_max], 'b-', label = 'Output temporal Phase')
plt.xlabel('Time (fs)')
plt.ylabel('Temporal phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#the intensity and the instantaneous frequency

plt.figure(27)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E9_t_out[t_min:t_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], w9_inst[t_min:t_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

temporal_dur1 = FWHM(np.abs(E9_t_out[t_min:t_max])**2, t)[0]
print(f'The pulse  duration (FWHM) from the I(t) curve of FOD = -1000 is {temporal_dur1: 0.2f} fs ')