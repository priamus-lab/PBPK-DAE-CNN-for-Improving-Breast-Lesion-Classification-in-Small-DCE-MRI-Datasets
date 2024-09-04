import numpy as np
import math
import scipy.stats
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize
import json
import os
import argparse

#prende in ingresso il valore di un pixel e calcola la concentrazione di ct
#sui 10 tempi
def Contrast_Agent_Concentration_misured(si, si_t):
  T1 = 5.3 #0.820 T1 = 5.08 #0.820
  r1 = 2.6 #4.5 r1 = 2.39 #4.5
  #si è un vettore di 9 elementi -> escludiamo il primo
  RE = np.divide(np.subtract(si_t, si), si)
  Ct_misured = np.divide(RE, T1*r1)
  return Ct_misured
  
  
  
  
#------------------------------------------------------------------------- AIF
def Weinmann_AIF(time):
  Cp=np.zeros(5)
  D = 0.1
  a1 = 3.99
  a2 = 4.78
  m1 = 0.144
  m2 = 0.0111
  n_time = time.shape[0]
  for i in range(0,n_time):
    Cp[i] = D * ((a1 * math.exp(-m1*time[i])) + (a2 * math.exp(-m2*time[i])))

  return Cp

def Parker_AIF(time):
  A1 = 0.809
  A2 = 0.330
  T1 = 0.17046
  T2 = 0.365
  sigma1 = 0.0563
  sigma2 = 0.132
  alpha = 1.050
  beta = 0.1685
  s = 38.078
  tau = 0.483

  Cp=np.zeros(5)
  pdf_1 = scipy.stats.norm(loc = T1, scale = sigma1)
  pdf_2 = scipy.stats.norm(loc = T2, scale = sigma2)
  n_time = time.shape[0]

  for i in range(0,n_time):
    Cp[i] = A1*pdf_1.pdf(time[i]) + A2*pdf_2.pdf(time[i]) + alpha*((math.exp(-beta*time[i]))/(1 + math.exp(-s*(time[i] - tau))))
  return Cp

#------------------------------------------------------------------------- AIF SINGLE
def Weinmann_AIF_single(time):
  D = 0.1
  a1 = 3.99
  a2 = 4.78
  m1 = 0.144
  m2 = 0.0111
  Cp = D * ((a1 * math.exp(-m1*time)) + (a2 * math.exp(-m2*time)))
  return Cp

def Parker_AIF_single(time):
  A1 = 0.809
  A2 = 0.330
  T1 = 0.17046
  T2 = 0.365
  sigma1 = 0.0563
  sigma2 = 0.132
  alpha = 1.050
  beta = 0.1685
  s = 38.078
  tau = 0.483

  pdf_1 = scipy.stats.norm(loc = T1, scale = sigma1)
  pdf_2 = scipy.stats.norm(loc = T2, scale = sigma2)

  Cp = A1*pdf_1.pdf(time) + A2*pdf_2.pdf(time) + alpha*((math.exp(-beta*time))/(1 + math.exp(-s*(time - tau))))
  return Cp
  
  
  
def integral_trap(t,y):
  return np.sum(np.multiply( t[1:]-t[:-1], (y[:-1] + y[1:])/2. ) )

def TK_model_Weinmann_integral_trap(time,ktrans, ve):
  Ct = np.zeros(5)
  Cp = Weinmann_AIF(time)
  n_time = time.shape[0]
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y)
  return Ct

def ETK_model_Weinmann_integral_trap(time,ktrans, ve, vp):
  Ct = np.zeros(5)
  n_time = time.shape[0]
  Cp = Weinmann_AIF(time)
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y)  + vp*Cp[i]
  return Ct

def TK_model_Parker_integral_trap(time,ktrans, ve):
  Ct = np.zeros(5)
  Cp = Parker_AIF(time)
  n_time = time.shape[0]
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y)
  return Ct

def ETK_model_Parker_integral_trap(time,ktrans, ve, vp):
  Ct = np.zeros(5)
  Cp = Parker_AIF(time)
  n_time = time.shape[0]
  for i in range(0,n_time):
    y = np.multiply(np.exp(-(ktrans/ve)*(time[i] -time[:i+1])), Cp[:i+1])
    Ct[i] = ktrans*integral_trap(time[:i+1],y)  + vp*Cp[i]
  return Ct



def fitting_mediano(patient, patient_path, savePath):
  x = sio.loadmat(patient_path + patient)
  data = x['image']
  #print(data.shape)
  mroi = x['mask']
  #print(mroi.shape)

  #Matlab ha l'indice che inizia da 1
  cordz = x['cordZ'] -1

  timeline = x['timeInterval_seconds_cumulative'][0]/60.
  #print(timeline)

  #normalizziamo data
  single_slice = (data - np.min(data))/(np.max(data) - np.min(data))

  single_mroi = mroi
  selected_pixel = single_slice[single_mroi>0]
  median_pixel = np.median(selected_pixel, axis = 0)
  ct_misured = Contrast_Agent_Concentration_misured(median_pixel[0], median_pixel)

  param_tk_w, opt = scipy.optimize.curve_fit(TK_model_Weinmann_integral_trap, timeline,ct_misured)  
  tk_weiman = param_tk_w.tolist()
  
  c_fitted_tk_weinmann =  TK_model_Weinmann_integral_trap(timeline, tk_weiman[0], tk_weiman[1]).tolist()
  ct_misured_list = ct_misured.tolist()

  x['tk_weiman_param'] = tk_weiman
  x['tk_weiman'] = c_fitted_tk_weinmann
  x['ct_misured_list'] = ct_misured_list
  x['normalized_timeline'] = timeline

  sio.savemat(savePath + patient, x)
  
  return x
  
  
def plot_curve(infoPatient, savePath):
  timeline = infoPatient['normalized_timeline']
  c_fitted_tk_parker=  TK_model_Weinmann_integral_trap(timeline,infoPatient['tk_weiman_param'][0],infoPatient['tk_weiman_param'][1])
  
  plt.figure()
  plt.plot(timeline, infoPatient['ct_misured_list'], 'b', label='misured')
  plt.plot(timeline, c_fitted_tk_parker, 'r', label='fitted')
  plt.title('TK WEIMAN')
  plt.legend()
  plt.savefig(savePath)