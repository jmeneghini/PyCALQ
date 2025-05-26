import os
from enum import Enum

class Task(Enum): #encode tasks into enum
    preview = 0          # Simplified from preview_corrs
    average = 1          # Simplified from average_corrs  
    rotate = 2           # Simplified from rotate_corrs
    fit = 3              # Simplified from fit_spectrum
    compare = 4          # Simplified from compare_spectrums

