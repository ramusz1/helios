import math
import numpy as np
import cv2
import xgboost as xgb
from sklearn.pipeline import Pipeline
from .dim_reduction_util import CustomPreproc, BandDenoiser, crop
from .util import draw_rect


# modified from ostrack source
def get_ostrack_search_crop(im, target_bb, search_area_factor=5.0):

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
#     im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
#     return im_crop
    x1, y1, x2, y2 = x1 + x1_pad, y1 + y1_pad, x2 - x2_pad, y2 - y2_pad
    return x1, y1, x2 - x1, y2 - y1


def pick_best_spread(fi, alpha=0.005):
    i = np.argmax(fi)
    n = len(fi)
    best_score = -np.inf
    for j in range(n):
        for k in range(j+1, n):
            if j == i or k == i:
                continue
            bands = np.array(sorted([i,j,k], reverse=True))
            score = np.sum(fi[bands])
            bn = bands / (n-1)
            def spread_fn(x, y):
                return 1.0 / np.abs(x - y)
            spread = spread_fn(bn[0], bn[1]) + spread_fn(bn[1], bn[2]) 
            score -= spread * alpha
            if score > best_score:
                top_bands = bands
                best_score = score

    return top_bands, best_score


class ToUint8:
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        return (X * 255.0).astype(np.uint8)


class ImadjustV5:
    
    def __init__(self, alpha=0.5, transform_perc=(0.01,0.99), fit_perc=(0.0,1.0), hist_eq_strength=0.5, debug=False):
        self.alpha = alpha
        self.fit_perc = fit_perc
        self.transform_perc = transform_perc
        self.debug = debug
        self.hist_eq_strength = hist_eq_strength
    
#         thanks to
#         https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
    
    def _get_tonemapping(self, X, lowp, highp):
        number_bins = 1024
            
        X = np.clip(X, 0, number_bins-1)
        cdf = np.zeros((X.shape[-1], number_bins))
        for i in range(X.shape[-1]):
            channel = X[...,i].flatten()
            bins = np.arange(number_bins + 1)
            hist, bins = np.histogram(channel, bins=bins, density=True)
            cdf[i] = hist.cumsum()
            cdf[i] /= cdf[i][-1] # normalize
            # normalize cdf and clip value range
            low = np.min(np.argwhere(cdf[i] >= lowp))
            high = np.max(np.argwhere(cdf[i] <= max(highp,cdf[i][0])))
            span = cdf[i][high] - cdf[i][low]
            if span > 0:
                cdf[i] = np.clip((cdf[i] - cdf[i][low]) / span, 0, 1)
            else:
                cdf[i] = np.linspace(0, 1, number_bins)
                
            # also calculate a linear cdf and blend them
            ln_cdf = np.zeros(number_bins, dtype=np.float32)
#             idx = np.argwhere(hist > 0)
#             low = max(np.min(idx) - 1, 0)
#             high = np.max(idx)
#             print(low, high)
            ln_cdf[low:high+1] = np.linspace(0, 1, high + 1 - low)
            ln_cdf[high+1:] = 1.0
            
            cdf[i] = self.hist_eq_strength * cdf[i] + (1-self.hist_eq_strength) * ln_cdf
            cdf[i] /= cdf[i][-1] # renormalize to be sure
        return cdf
    
    def fit(self, X, y=None, **fit_params):
        self.cdf = self._get_tonemapping(X, *self.fit_perc)
        
    def transform(self, X):
        # blend current frame cdf and the target cdf
        frame_cdf = self._get_tonemapping(X, *self.transform_perc)
        
        if self.debug:
            for i in range(3):
                plt.plot(self.cdf[i], label="self_cdf")
                plt.plot(frame_cdf[i], label="frame_cdf")
                plt.legend()
                plt.show()
        
        frame_cdf = self.alpha * frame_cdf + (1 - self.alpha) * self.cdf
        
        # apply the tonemapping
        out = np.empty_like(X, dtype=np.float32)
        X = np.clip(X, 0, frame_cdf.shape[1] - 1)
        for i in range(X.shape[-1]):
            channel = X[...,i].flatten()
#             out[...,i] = frame_cdf[i][X[...,i].flatten()].reshape(*X.shape[:-1])
            out[...,i] = frame_cdf[i][channel].reshape(out.shape[:-1])
            
        return out


class XGBBandSelectorV6:
    
    def __init__(self, alpha=0.005, **params):
        self.classifier = xgb.XGBClassifier(**params)
        self.alpha = alpha
        
    def fit(self, X, y, **fit_params):
        c = X.shape[-1]
        self.classifier.fit(X.reshape(-1,c), y.flatten())
        fi = self.classifier.feature_importances_
        self.bands, pbs_score = pick_best_spread(fi, self.alpha)
        self.scores = fi[self.bands]
        return self
    
    def transform(self, X):
        return X[..., self.bands]
    
    def predict_proba(self, X):
        proba = self.classifier.predict_proba(X.reshape(-1, X.shape[-1]))
        return proba.reshape(*X.shape[:-1], proba.shape[-1])


class BilateralFilter:
    
    def __init__(self, d=5, sigma_color=10, sigma_space=10):
        self.d = d
        self.sigma_color = 10
        self.sigma_space = 10
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        out = X.astype(np.float32)
        c = X.shape[-1]
        if c <= 3:
            out = cv2.bilateralFilter(out, self.d, self.sigma_color, self.sigma_space)
        else:
            for i in range(X.shape[-1]):
                out[...,i] = cv2.bilateralFilter(out[...,i], self.d, self.sigma_color, self.sigma_space)
        return out.astype(X.dtype)
    
                
class DimReduceV4:
    
    def __init__(self, band_selector, imadjust, spatial_filtering):
        self.bs = band_selector
        self.imadjust = imadjust
        self.spatial_filtering = spatial_filtering
        
    def initialize(self, hsi, init_bbox, camera_type):
        self.camera_type = camera_type
        # initial preprocessing
        if self.camera_type == "HSI-VIS":
            hsi = CustomPreproc().transform(hsi)
        hsi = BandDenoiser().transform(hsi)
#         hsi = BilateralFilter().transform(hsi)
        hsi = self.spatial_filtering.transform(hsi)

        # preapare target mask
        h,w,c = hsi.shape
        mask = draw_rect(np.zeros((h,w),dtype=np.uint8), init_bbox, color=1, thickness=-1).astype(bool)
        roi = get_ostrack_search_crop(hsi, init_bbox)
        X = crop(hsi, roi).copy()
        y = crop(mask, roi)

        # imadjus
        self.imadjust.fit(crop(hsi, roi))
        hsi = self.imadjust.transform(hsi)
        hsi = (hsi * 255.0).astype(np.uint8)

        # feature selection
        X = crop(hsi, roi).copy()
        y = crop(mask, roi)
        self.bs.fit(X, y)
        rgb = self.bs.transform(hsi)
        
        # select bands in imadjust
        self.imadjust.cdf = self.imadjust.cdf[self.bs.bands]
        
        if self.camera_type == "HSI-VIS":
            self.band_selection_pipeline = Pipeline([
                ("custom_preproc", CustomPreproc()),
                ("denoising", BandDenoiser()),
                ("bs", self.bs),
                ("spatial_denoising", self.spatial_filtering),
            ])
        else:
            self.band_selection_pipeline = Pipeline([
                ("denoising", BandDenoiser()),
                ("bs", self.bs),
                ("spatial_denoising", self.spatial_filtering),
            ])
            
        return rgb

    def __call__(self, frame, bbox):
        frame = self.band_selection_pipeline.transform(frame)
        roi = get_ostrack_search_crop(frame, bbox)
        self.imadjust.fit(crop(frame, roi))
        frame = self.imadjust.transform(frame)
        return (frame * 255.0).astype(np.uint8)


