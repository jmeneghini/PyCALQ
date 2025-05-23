import numpy as np
import math
import luescher.tools.kinematics as kin_tools

# single_channel parametrizations
def output(ecm,ma,mb,fit_param, fit_params):
    if fit_param == 'ERE':
        return ere(ecm,ma,mb,*fit_params)
    elif fit_param == 'ERE_delta':
        return ere_delta(ecm,ma,mb,*fit_params)

def delta_Sp(ecm,ma,mb):
    return (ecm**2 - (ma+mb)**2 )/ (ma+mb)**2

def ere_delta(ecm,ma,mb,a,b):
            return (ecm)*(a+b*delta_Sp(ecm,ma,mb))

# fit ere expansion
def ere(ecm,ma,mb,a,b):
    return ((-1/a)+0.5*b*kin_tools.q2(ecm,ma,mb) ) #in units of reference mass, usually mpi

