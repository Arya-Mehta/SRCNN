import numpy as np

def calculate_psnr(output, target):
    output = np.asarray(output, dtype='float32')
    target = np.asarray(target, dtype='float32')
    
    mse = np.mean((target-output)**2)
    if mse == 0:
        return 100
    
    psnr = 10 * np.log10(1/mse)
    return psnr