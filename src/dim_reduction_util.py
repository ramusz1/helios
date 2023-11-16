import numpy as np

def imadjust(x):
    # find 1st and 99st percentile
    h,w,c = x.shape
    low = np.percentile(x.reshape(-1,c), 1, axis=0)
    high = np.percentile(x.reshape(-1,c), 99, axis=0)
    x = (np.clip(x, low, high) - low) / (high - low)
    return x

def hsi_to_rgb(hsi, bands=None, in_max=255.0):
    if bands is None:
        bands = [14,2,8]
    rgb = hsi[:,:,bands].astype(np.float32).copy()
    gain = 1.0
    gamma = 0.4
#     rgb = np.power(gain * rgb / 255.0, gamma)
    rgb = np.power(gain * rgb / in_max, gamma)
    rgb = imadjust(rgb)
    return rgb

def interpolate_13th_channel(hsi):
    hsi = hsi.copy()
    hsi[...,12] = (hsi[...,11] + hsi[...,13]) / 2
    return hsi

def crop(img, bbox):
    x,y,w,h = bbox
    return img[y:y+h,x:x+w]

class CustomPreproc:
    
    def fit(self, X, y, **fit_params):
        return self
    
    def transform(self, X):
        return interpolate_13th_channel(X)    
        
class BandDenoiser:
    
    def fit(self, X, y, **fit_params):
        return self
    
    def transform(self, X):
#         kernel = np.array([1,2,1])
#         output = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), -1, output)
        c = X.shape[-1]
        output = 2 * X
        output[...,:-1] += X[...,1:]
        output[...,1:] += X[...,:-1]
        norm = np.full(c, 4)
        norm[0] = 3
        norm[-1] = 3
        output = output / norm
        output = output.astype(X.dtype)
        return output

