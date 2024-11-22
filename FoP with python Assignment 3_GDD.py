# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:53:44 2024

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

#To get the analytical pulse duration
def analytical_pulse_duration(tau0, GDD0):
    pulse_dur = tau0 * np.sqrt(1 + (4 * np.log(2) * GDD0 / tau0**2)**2)
    return pulse_dur                          
t1_a = analytical_pulse_duration(6, 50)    
t2_a = analytical_pulse_duration(6, -50)    
t3_a = analytical_pulse_duration(6, 1000)    
t4_a = analytical_pulse_duration(6, 8000)    
print(f'The analytical pulse duration for GDD of 50 is {t1_a:0.2f} fs')
print(f'The analytical pulse duration for GDD of -50 is {t2_a:0.2f} fs')
print(f'The analytical pulse duration for GDD of 1000 is {t3_a:0.2f} fs')
print(f'The analytical pulse duration for GDD of 8000 is {t4_a:0.2f} fs')

#the analytical duration from GD
def analytical_dur_GDD(tau0, GDD0):
    delta_w = 4 * np.log(2) / tau0
    Time_duraion = delta_w * GDD0
    return Time_duraion
tg1_a = analytical_dur_GDD(6, 50)
tg2_a = analytical_dur_GDD(6, 1000)
tg3_a = analytical_dur_GDD(6, 8000)

print(f'The pulse duration from GDD of 50 is {tg1_a:0.2f} fs')
print(f'The pulse duration from GDD of 1000 is {tg2_a:0.2f} fs')
print(f'The pulse duration from GDD of 8000 is {tg3_a:0.2f} fs')

#%% FOR GDD of 50
#the input field and other parameters
E1_w_in = complex_spectrum(tau_0, w, w_0)
H1_w = transfer_function(0, 0, 50, 0, 0, 0, N)
i_min = spectral_index(w, 1, 4, w_0)[0]
i_max = spectral_index(w, 1, 4, w_0)[1]
i0 = spectral_index(w, 1, 4, w_0)[2]
t_min = time_index(t, -50, 50)[0]
t_max = time_index(t, -50, 50)[1]
t0 = time_index(t, -50, 50)[2]

#The output - PLOTTED PARAMETERS
E1_w_out = output_spectral_field(H1_w, E1_w_in, i0)[0] #spectra domain
output_phase1 = output_spectral_field(H1_w, E1_w_in, i0)[1]
relGD1_out = relative_GD_out(output_phase1, i0, w) #The relative GD
E1_t_out = inverse_fourier(E1_w_out) #temporal domain
phase1_out_t = temporal_phase(E1_t_out, t0)[0]
w1_inst = temporal_phase(E1_t_out, t0)[1] #The instantaneous frequency
#%% Plots
#The  spectral field and phase
plt.figure(1)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E1_w_out[i_min:i_max]), 'r-', label = 'Output Electric spectrum')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], output_phase1[i_min:i_max], 'b-', label = 'Output Phase')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#The intensity and relative GD

plt.figure(2)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E1_w_out[i_min:i_max])**2, 'r-', label = 'Output Intensity')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Intensity (a.u)')
plt.legend()
plt.grid()


plt.subplot(212)
plt.plot(w[i_min:i_max], relGD1_out[i_min:i_max], 'b-', label = 'Relative GD')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Relative GD (fs)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Temporal intensity and phase

plt.figure(3)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E1_t_out[t_min:t_max]), 'r--', label = 'Output temporal field')
plt.plot(t[t_min:t_max], np.real(E1_t_out[t_min:t_max]), 'r-')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], phase1_out_t[t_min:t_max], 'b-', label = 'Output temporal Phase')
plt.xlabel('Time (fs)')
plt.ylabel('Temporal phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#the intensity and the instantaneous frequency

plt.figure(4)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E1_t_out[t_min:t_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], w1_inst[t_min:t_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

temporal_dur = FWHM(np.abs(E1_t_out[t_min:t_max])**2, t)[0]
# t0_dur = FWHM(np.abs(E1_t_out[t_min:t_max])**2, t)[1] #cal for the time dur for FWHM
# t1 = FWHM(np.abs(E1_t_out[t_min:t_max])**2, t)[2]

print(f'The pulse  duration (FWHM) from the I(t) curve of GDD = 50 is {temporal_dur: 0.2f} fs ')


#%% For GDD of -50

#the input field and other parameters
E2_w_in = complex_spectrum(tau_0, w, w_0)
H2_w = transfer_function(0, 0, -50, 0, 0, 0, N)


#The output - PLOTTED PARAMETERS
E2_w_out = output_spectral_field(H2_w, E2_w_in, i0)[0] #spectra domain
output_phase2 = output_spectral_field(H2_w, E2_w_in, i0)[1]
relGD2_out = relative_GD_out(output_phase2, i0, w) #The relative GD
E2_t_out = inverse_fourier(E2_w_out) #temporal domain
phase2_out_t = temporal_phase(E2_t_out, t0)[0]
w2_inst = temporal_phase(E2_t_out, t0)[1] #The instantaneous frequency

plt.figure(5)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E2_w_out[i_min:i_max]), 'r-', label = 'Output Electric spectrum')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], output_phase2[i_min:i_max], 'b-', label = 'Output Phase')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#The intensity and relative GD

plt.figure(6)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E2_w_out[i_min:i_max])**2, 'r-', label = 'Output Intensity')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Intensity (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], relGD2_out[i_min:i_max], 'b-', label = 'Relative GD')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Relative GD (fs)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Temporal intensity and phase

plt.figure(7)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E2_t_out[t_min:t_max]), 'r--', label = 'Output temporal field')
plt.plot(t[t_min:t_max], np.real(E2_t_out[t_min:t_max]), 'r-')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], phase2_out_t[t_min:t_max], 'b-', label = 'Output temporal Phase')
plt.xlabel('Time (fs)')
plt.ylabel('Temporal phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#the intensity and the instantaneous frequency

plt.figure(8)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E2_t_out[t_min:t_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t_min:t_max], w2_inst[t_min:t_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



temporal_dur1 = FWHM(np.abs(E2_t_out[t_min:t_max])**2, t)[0]
print(f'The pulse  duration (FWHM) from the I(t) curve of GDD = -50 is {temporal_dur1: 0.2f} fs ')


#%% FOR GDD 1000 and time interval -1000, 1000

#the input field and other parameters
E3_w_in = complex_spectrum(tau_0, w, w_0)
H3_w = transfer_function(0, 0, 1000, 0, 0, 0, N)
t3_min = time_index(t, -1000, 1000)[0] #new time index!!
t3_max = time_index(t, -1000, 1000)[1]
t30 = time_index(t, -1000, 1000)[2]

#The output - PLOTTED PARAMETERS
E3_w_out = output_spectral_field(H3_w, E3_w_in, i0)[0] #spectra domain
output_phase3 = output_spectral_field(H3_w, E3_w_in, i0)[1]
relGD3_out = relative_GD_out(output_phase3, i0, w) #The relative GD
E3_t_out = inverse_fourier(E3_w_out) #temporal domain
phase3_out_t = temporal_phase(E3_t_out, t30)[0]
w3_inst = temporal_phase(E3_t_out, t30)[1] #The instantaneous frequency

#the intensity and the instantaneous frequency

plt.figure(9)
plt.subplot(211)
plt.plot(t[t3_min:t3_max], np.abs(E3_t_out[t3_min:t3_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t3_min:t3_max], w3_inst[t3_min:t3_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

temporal_dur3 = FWHM(np.abs(E3_t_out[t3_min:t3_max])**2, t)[0]
print(f'The pulse  duration (FWHM) from the I(t) curve of GDD = 1000 is {temporal_dur3: 0.2f} fs ')
#%% FOR GDD 8000 and time interval -T/2, T/2

#the input field and other parameters
E4_w_in = complex_spectrum(tau_0, w, w_0)
H4_w = transfer_function(0, 0, 8000, 0, 0, 0, N)
t4_min = time_index(t, -T/2, T/2)[0] #new time index!!
t4_max = time_index(t, -T/2, T/2)[1]
t40 = time_index(t, -T/2, T/2)[2] #This is same and constant

#The output - PLOTTED PARAMETERS
E4_w_out = output_spectral_field(H4_w, E4_w_in, i0)[0] #spectra domain
output_phase4 = output_spectral_field(H4_w, E4_w_in, i0)[1]
relGD4_out = relative_GD_out(output_phase4, i0, w) #The relative GD
E4_t_out = inverse_fourier(E4_w_out) #temporal domain
phase4_out_t = temporal_phase(E4_t_out, t40)[0]
w4_inst = temporal_phase(E4_t_out, t40)[1] #The instantaneous frequency

#the intensity and the instantaneous frequency

plt.figure(10)
plt.subplot(211)
plt.plot(t[t4_min:t4_max], np.abs(E4_t_out[t4_min:t4_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t[t4_min:t4_max], w4_inst[t4_min:t4_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% FOR GDD 8000 and time interval -T/2, T/2 with increase time window


#A function to get the time array and new time indexes
def get_time_window_array(N):
    T = N * dt
    t = np.arange(-T/2, T/2, dt)
    dw = omega / N
    w = np.arange(-omega/2, omega/2, dw)
    return T, t, w #This return the time window and the time window array
T1 = get_time_window_array(2**18)[0]
t1 = get_time_window_array(2**18)[1]

t5_min = time_index(t1, -T1/2, T1/2)[0] #new time index!! 
t5_max = time_index(t1, -T1/2, T1/2)[1]
t50 = time_index(t1, -T1/2, T1/2)[2] #This here we use the new time window array!!!

#the input field and other parameters
E5_w_in = complex_spectrum(tau_0, get_time_window_array(2**18)[2], w_0)
H5_w = transfer_function(0, 0, 8000, 0, 0, 0, 2**18)

i00 = spectral_index(get_time_window_array(2**18)[2], 1, 4, w_0)[2] #Take note!!
#The output - PLOTTED PARAMETERS
E5_w_out = output_spectral_field(H5_w, E5_w_in, i00)[0] #spectra domain
output_phase5 = output_spectral_field(H5_w, E5_w_in, i00)[1]
relGD5_out = relative_GD_out(output_phase5, i00, get_time_window_array(2**18)[2]) #The relative GD
E5_t_out = inverse_fourier(E5_w_out) #temporal domain
phase5_out_t = temporal_phase(E5_t_out, t50)[0] ###################
w5_inst = temporal_phase(E5_t_out, t50)[1] #The instantaneous frequency

#the intensity and the instantaneous frequency

plt.figure(11)
plt.subplot(211)
plt.plot(t1[t5_min:t5_max], np.abs(E5_t_out[t5_min:t5_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(t1[t5_min:t5_max], w5_inst[t5_min:t5_max], 'b-', label = 'Instantaneous frequency')
plt.xlabel('Time (fs)')
plt.ylabel('Instantaneous frequency (PHz)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

temporal_dur5 = FWHM(np.abs(E5_t_out[t5_min:t5_max])**2, t1)[0]
t0 = FWHM(np.abs(E5_t_out[t5_min:t5_max])**2, t1)[1]
t1 = FWHM(np.abs(E5_t_out[t5_min:t5_max])**2, t1)[2]

print(f'The pulse  duration (FWHM) from the I(t) curve of GDD = 8000 is {temporal_dur5: 0.2f} fs ')