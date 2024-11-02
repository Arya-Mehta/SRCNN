import numpy as np

def psnr(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    
    mse = np.mean((target-output)**2)
    if mse == 0:
        return 100
    
    psnr = 10 * np.log10(1/mse)
    return psnr