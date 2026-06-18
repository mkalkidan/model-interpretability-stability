import scipy.io
import os

def load_data(filepath):
    # Load the MATLAB format file
    mat_data = scipy.io.loadmat(filepath)
    # Filter out MATLAB's internal metadata headers
    clean_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    # Assuming standard structure where keys are 'X' and 'y'
    X = clean_data['X']
    y = clean_data['y'].ravel()
    return X, y