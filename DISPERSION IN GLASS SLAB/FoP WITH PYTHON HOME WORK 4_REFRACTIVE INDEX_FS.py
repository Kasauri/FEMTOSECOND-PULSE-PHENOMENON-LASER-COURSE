# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:16:13 2024

@author: abedn
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#Necessary parameter
N = 2**17
omega = 50 #PHz
dw = omega / N
w = np.arange(-omega/2, omega/2, dw) #angular frequency array
c = 299.792458 #nm/fs
T = 2 * np.pi* N / omega
dt = T/ N
wl = 800 #nm
w_0 = 2 * np.pi * c  / wl #central angular freq / carrier #PHz
tau_0 = 6 #fs
t = np.arange(-T/2, T/2, dt)

#Refractive index sellmieier equation for the different glass type from refractive index.info webpage
#Fused silica (FS)  = 1 range (0.21, 6.7)
# SF_10 = 2 range (0.38, 2.5)
#BK_7 = 3 range (0.3, 2.5)

def refindex(angular_freq, glass_type):
    if angular_freq == 0 :
       return 1 #um microns Return the wavelength as 1
    else:
        x = (2 * np.pi * c / angular_freq) * 1e-3 #convert to micrometer This is the wavelength
    
    if glass_type == 1 and  0.21 <=  x <= 6.7:
        n = (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5 #fused silica
    elif glass_type == 2 and  0.38 <=  x <= 2.5:
        n = (1+1.62153902/(1-0.0122241457/x**2)+0.256287842/(1-0.0595736775/x**2)+1.64447552/(1-147.468793/x**2))**.5 #SF_10
    elif glass_type == 3 and  0.3 <=  x < 2.5:
        n = (1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**.5 #BK_7    
    else:
        n = 1
    return n #casting that for any other the refractive index is n = 1
    

#%%Use the function to get the refractive index

# n_FS = refindex(w_0, 1)
# n_SF_10 = refindex(w_0, 2)
# n_BK_7 = refindex(w_0, 3)
# print(f'the refractive index of BK7 = {n_BK_7 :.4f} at 800nm') #testing the function
#%% Getting the refractive index at the different angular freq in the array
n_FS = np.array([refindex(i, 1) for i in w])
n_SF10 = np.array([refindex(i, 2) for i in w])
n_BK7 = np.array([refindex(i, 3) for i in w])

#%% The plots

plt.figure(1)
plt.subplot(211)
plt.plot(w, n_FS, 'r-', label = 'Fused silica')
plt.plot(w, n_SF10, 'g-', label = 'SF10')
plt.plot(w, n_BK7, 'b-', label = 'BK7')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Refractive index n')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(2* np.pi * c / w, n_FS, 'r-', label = 'Fused silica')
plt.plot(2* np.pi * c / w, n_SF10, 'g-', label = 'SF10')
plt.plot(2* np.pi * c / w, n_BK7, 'b-', label = 'BK7')
plt.ylabel('Refractive index n')
plt.xlabel('Waveleength (nm)')
plt.xlim(400, 1000)
plt.grid()
plt.tight_layout()

#%% defining some functions

def complex_spectrum(t, w, w_c): #t is the time FWHM, w_c is the carrier angular freq
    amplitude = np.sqrt(np.pi) * t / np.sqrt(8 * np.log(2))
    exp_term = np.exp(-(w - w_c)**2 * t**2 / (8 * np.log(2)))
    spectrum = amplitude * exp_term
    return spectrum

E1_w_in = complex_spectrum(tau_0, w, w_0)

#The spectral phase and transfer function
def transfer_function(n_w): #the refractive index to get the spectral phase
  A =  np.ones(N)
  L = 1e6
  spectral_phase = w/c * n_w * L
  H = A * np.exp(-1j * spectral_phase)
  return H

H1_w = transfer_function(n_FS)


i_min = np.argmin(np.abs(w - 1))
i_max = np.argmin(np.abs(w - 4))
i0 = np.argmin(np.abs(w - w_0))


#%%The output E field
#the output E field is a product of the transfer funnction and the input spectral field
#to get the output spectral field and the phase  
def output_spectral_field(H,E_in, i_0): #E_in is the input spectral domain here
    output_field = H * E_in
    output_phase = np.unwrap(-np.angle(output_field))
    n = round(output_phase[i_0] / (2 * np.pi))
    phase_out = output_phase - n * 2 * np.pi
    return output_field, phase_out


E1_w_out = output_spectral_field(H1_w, E1_w_in, i0)[0] #spectra domain
output_phase1 = output_spectral_field(H1_w, E1_w_in, i0)[1]
L = 1e6 #nm
spectral_phase = w/c * n_FS * L

#The  spectral field and phase
plt.figure(2)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E1_w_out[i_min:i_max]), 'r-', label = 'Output Electric spectrum')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], spectral_phase[i_min:i_max], 'b-', label = 'Output Phase') ######################
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral phase (rad)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%To get the relative GD (spectral domain)
def relative_GD_out(output_phase, i_0, w):
    dom = w - w_0
    GD_out = np.gradient(output_phase, dom)
    relGD_out = GD_out - GD_out[i_0]
    GDD = np.gradient(GD_out, dom) #Group Delay Dispersion
    return relGD_out, GDD, GD_out
relGD1_out = relative_GD_out(output_phase1, i0, w)[0] #The relative GD
GDD_out = relative_GD_out(output_phase1, i0, w)[1] #The  GDD
GD_out = relative_GD_out(output_phase1, i0, w)[2] #The  GD at i0

GD_0 = GD_out[i0]
print(f'The GD value closest to the carrier angular frequency = {GD_0:0.1f} fs')
# delta_GD = np.abs()
#The intensity and  GD (Group Delay)

plt.figure(3)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E1_w_out[i_min:i_max])**2, 'r-', label = 'Spectral Intensity')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral Intensity (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], GD_out[i_min:i_max], 'b-', label = 'GD')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Group Delay (fs)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(4)
plt.subplot(211)
plt.plot(w[i_min:i_max], np.abs(E1_w_out[i_min:i_max])**2, 'r-', label = 'Spectral Intensity')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Spectral Intensity (a.u)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(w[i_min:i_max], GDD_out[i_min:i_max], 'b-', label = 'GDD')
plt.xlabel('Angular frequency (PHz)')
plt.ylabel('Group Delay Dispersion (fs^2)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% Time Duration using GDD formular

def time_dur_GDD0 (tau_0, GDD0):
    tau = tau_0 * np.sqrt(1 + (4 * np.log(2) * GDD0/tau_0**2)**2)
    return tau
GDD_out_0 = GDD_out[i0]
print(f'The GDD at the angular frequency closest to the carrier frequency = {GDD_out_0 : 0.2f} fs^2 ')
time_dur = time_dur_GDD0(tau_0, GDD_out_0)
print(f'The time duration of the pulse using GDD value = {time_dur: 0.2f} fs')

#%% FINDING THE INDICES AND THE CLOSEST  ANGULAR FREQUENCY

def spec_FWHM(intensity, arg_FWHM):
    t1, t2=arg_FWHM[intensity>max(intensity)/2][0], arg_FWHM[intensity>max(intensity)/2][-1]
    i1, i2=np.argmin(arg_FWHM<t1), np.argmin(arg_FWHM<t2)
    return t1, t2, i1, i2

w1, w2, i0, i1=spec_FWHM(np.abs(E1_w_out)**2, w)

print(f"Indices for FWHM: i0={i0: 0.2f}, i1={i1:0.2f}")
print(f"w1 and w2 are : w1 = {w1: 0.2f}, w2 ={w2 :0.2f}")

delta_w = 4 * np.log(2) / tau_0

print(f'The analytical FWHM of the angular frequency = {delta_w: 0.2f} PHz')


delta_GD = np.abs(GD_out[i0] - GD_out[i1])
print(f'Delta GD = {delta_GD:0.2f}')


#%%Inverse transform 
def inverse_fourier(spectrum): #spectrum is the spectral field
    temporal_field = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(spectrum))) * omega / (2 * np.pi)
    return temporal_field

E1_t_out = inverse_fourier(E1_w_out) #temporal domain

#A  function to get the time index 

t_a = GD_0 - 3 * time_dur
t_b = GD_0 + 3 * time_dur
t_min = np.argmin(np.abs(t - t_a))
t_max = np.argmin(np.abs(t - t_b))
t0 = np.argmin(np.abs(t - 0)) #x#x########x#########




#In the temporal domain the phase and instantaneous frequency
def temporal_phase(t_field, t_0):
    phase_out_t = np.unwrap(np.angle(t_field))
    n = round(phase_out_t[t_0] / (2 * np.pi))
    phase_out_t = phase_out_t - n * 2 * np.pi
    w_inst = np.gradient(phase_out_t, dt) #instantaneous frequency
    return phase_out_t, w_inst

phase1_out_t = temporal_phase(E1_t_out, t0)[0]
w1_inst = temporal_phase(E1_t_out, t0)[1] #The instantaneous frequency

#%%
#Temporal intensity and phase

plt.figure(5)
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

plt.figure(6)
plt.subplot(211)
plt.plot(t[t_min:t_max], np.abs(E1_t_out[t_min:t_max])**2, 'r-', label = 'Output temporal intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Temporal intensity (a.u)')
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
    return FWHM, t_0, t_1,i_0, i_1


temporal_dur = FWHM(np.abs(E1_t_out[t_min:t_max])**2, t)[0]
t0 = FWHM(np.abs(E1_t_out[t_min:t_max])**2, t)[1]
t1 = FWHM(np.abs(E1_t_out[t_min:t_max])**2, t)[2]
print(f'The pulse  duration (FWHM) from the I(t) curve = {temporal_dur: 0.2f} fs ')
   
#%% ANALYTICAL DURATION FROM GD

#the analytical duration from GD
def analytical_dur_GDD(tau0, GDD0):
    delta_w = 4 * np.log(2) / tau0
    Time_duraion = delta_w * GDD0
    return Time_duraion #This is equivalent to the change in the GD
tg1_a = analytical_dur_GDD(6, GDD_out_0)


print(f'The pulse duration from GD of {GDD_out_0: 0.2f} fs is {tg1_a:0.2f} fs')

