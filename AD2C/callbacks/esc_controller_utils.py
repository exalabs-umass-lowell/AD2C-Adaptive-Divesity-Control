import numpy as np
import math
import matplotlib.pyplot as plt

def es_function(t, T, hpf, lpf, f, c, a_param, Jk, Jkm1, sigmakm1, psikm1, gammakm1, uhatkm1, akm1, lpf_a):
    
    #outputs:
    #uk: the control to be applied at the next timestep
    #uhatk: the output of the integrator
    #gammak: the output of the lowpass filter
    #psik: signal after demodulation (mutliplication by cosine)
    #sigmak: the output of the washout (highpass) filter
    
    #inputs:
    #Jk, Jkm1: values of the objective function (metric we're optimizing) at the present and last timestep
    #sigmakm1: output of highpass filter at last timestep
    #psikm1: signal after demodulation at last timestep
    #gammakm1: output of lowpass filter 
    #uhatkm1: output of integrator at last timestep
    #t: present time at this timestep
    #T: distance between timesteps, delta_T
    #hpf: high pass filter gain, usually make this an order of magnitude lower than w
    #lpf: low pass filter gain, usually make this equal to hpf, but I usually end up adjusting this quite a bit
    #c: gain on the integrator - how much to value new measurements compared to old measurements
    #ak: the amplitude of the probing signal

    #calculate angular frequency:
    w = 2*np.pi*f

    #extract the effect of the probing signal in the objective function
    #do this by passing the signal through a highpass filter
    sigmak = (Jk - Jkm1 - (hpf*T/2-1)*sigmakm1)/(1+hpf*T/2)

    #the resulting signal is a sinusoid, multiply by a sinusoid of the same frequency
    #this results in a cos**2 term, that has a DC component (we call this demodulation)
    psik = sigmak*np.cos(w*t)

    #pass the demodulated signal through a lowpass filter, to eliminate noise and "jumpiness"
    gammak = (T*lpf*(psik + psikm1) - (T*lpf - 2)*gammakm1)/(2 + T*lpf)
    
    #probe amplitude adaptation
    ak = a_param * (T*lpf_a*( (np.arctan(psik) / np.pi * 2 )**2 + (np.arctan(psikm1) / np.pi * 2)**2)) - (T*lpf_a - 2)*akm1/(2 + T*lpf_a)

    #pass the resulting signal through an integrator - this approximates a gradient descent
    uhatk = uhatkm1 + c*T/2*(gammak + gammakm1)

    #modulation - add the perturbation to the next control setpoint
    uk = (uhatk + ak*np.cos(w*t))

    return (uk, sigmak, psik, gammak, uhatk, ak)

