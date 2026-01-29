
import scipy
import numpy as np 
import skimage

from typing import Optional

def gauss_filter(
        std: float
        ):
    ''' Generate a 1D Gaussian filter and its derivatives

    Args: 
        std: standard deviation of the Gaussian filter

    Returns:
        g (np.ndarray): 1D Gaussian filter
        dg (np.ndarray): derivative of the Gaussian filter
        ddg (np.ndarray): second derivative of the Gaussian filter

    Example:
        ```python
        import fibretracker as ft

        vol = ft.detector.gauss_filter(std=2.5)
        ```
        
    '''
    x = np.arange(-np.ceil(5*std), np.ceil(5*std) + 1)[:,None]
    g = np.exp(-x**2/(2*std**2))
    g /= np.sum(g)
    dg = -x/std**2 * g
    ddg = -g/std**2 -x/std**2 * dg
    return g, dg, ddg

def blob_centre_detector(
        im: np.ndarray, 
        std: float=2.5, 
        min_distance: int=3, 
        threshold_abs: float=0.4
        ):
    ''' Predict coordinates of fibres centre in a volume slice using blob detector

    Args: 
        im: input image
        std: standard deviation of the Gaussian filter
        min_distance: minimum distance between peaks
        threshold_abs: threshold value for the peak from the background

    Returns: 
        pred_coords (np.ndarray): predicted coordinates of the fibre centre

    Example:
        ```python
        import fibretracker as ft

        vol = ft.detector.blob_centre_detector(im, std=2.5, min_distance=3, threshold_abs=0.4)
        ```

    '''
    g = gauss_filter(std)[0]
    im_g = scipy.ndimage.convolve(scipy.ndimage.convolve(im, g), g.T)
    pred_coords = skimage.feature.peak_local_max(im_g, min_distance=min_distance, threshold_abs=threshold_abs)
    return pred_coords
    
def avg_fibre_coord(
        pred_coord: np.ndarray, 
        im: np.ndarray, 
        window_size: int = 10,
        apply_filter: bool = False,
        std: Optional[float] = None
        ):
    ''' Recompute the fibre centre in a slice using weighted average of peak neighbourhood

    Args: 
        pred_coord: predicted coordinates of the peaks
        im: input image
        window_size: size of the neighbourhood window around the peak
        apply_filter: whether to apply Gaussian filter to the window
        std: standard deviation of the Gaussian filter

    Returns:
        coords (np.ndarray): recomputed fibre centre coordinates in the slice with weighted average
    
    Example:
        ```python
        import fibretracker as ft

        avg_coord = ft.detector.avg_fib_coord(pred_coord, im, window_size)
        ```

    '''
    coords = []
    for coord in pred_coord:
        x, y = coord
        window = im[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]
        # Apply Gaussian filter to the window
        if apply_filter:
            if std is not None:
                g = gauss_filter(std)[0]
            else:
                g = gauss_filter(std=2.5)[0]
            window = scipy.ndimage.convolve(scipy.ndimage.convolve(window, g), g.T)
        x_coords, y_coords = np.meshgrid(range(x-window_size//2, x+window_size//2+1), range(y-window_size//2, y+window_size//2+1))
        weighted_x = np.sum(window * x_coords) / np.sum(window)
        weighted_y = np.sum(window * y_coords) / np.sum(window)
        coords.append([weighted_x, weighted_y])
    return np.array(coords)

def get_fibre_coords(
        vol: np.ndarray, 
        std: float=2.5, 
        min_distance: int=3, 
        threshold_abs: float=0.4,
        weighted_avg: bool=False,
        window_size: int=10,
        apply_filter: bool=False,
        ):
    ''' Get list of fibres centre coordinates in a volume using blob detector

    Args:
        vol: input volume
        std: standard deviation of the Gaussian filter
        min_distance: minimum distance between fibres
        threshold_abs: threshold value for the peak from the background
        weighted_avg: whether to apply weighted average to the detected coordinates
        window_size: size of the neighbourhood window around the peak
        apply_filter: whether to apply Gaussian filter to the window

    Returns:
        coords (List(nd.array)): List of fibres centre coordinates in the volume
    
    Example:
        ```python
        import fibretracker as ft

        vol = ft.detector.get_fib_coords(vol, std=2.5, min_distance=3, threshold_abs=0.4)
        ```

    '''
    coords = []
    for i, im in enumerate(vol):
        coord = blob_centre_detector(im, std=std, min_distance=min_distance, threshold_abs=threshold_abs)
        if weighted_avg:
            coord = avg_fibre_coord(coord, im, window_size=window_size, apply_filter=apply_filter, std=std)
        coords.append(np.stack([coord[:,1], coord[:,0], np.ones(len(coord)) * i], axis=1))
        print(f'Detecting coordinates - slice: {i+1}/{len(vol)}', end='\r')
    print(' ' * len(f'Detecting coordinates - slice: {i+1}/{len(vol)}'), end='\r')
    return coords

from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log, blob_dog, blob_doh

# ---------------------------
# Helpers / subpixel refine
# ---------------------------

def _normalize01(im: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Percentile normalize a 2D image to [0,1] for stable thresholds."""
    im = im.astype(np.float32, copy=False)
    lo, hi = np.percentile(im, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    imn = (im - lo) / (hi - lo)
    return np.clip(imn, 0.0, 1.0)

def avg_fibre_coord2(
    pred_coord: np.ndarray,
    im: np.ndarray,
    window_size: int = 10,
    apply_filter: bool = False,
    std: Optional[float] = None,
) -> np.ndarray:
    """
    Subpixel refinement: local intensity-weighted centroid in a window around each peak.
    pred_coord: (N,2) in (y, x) pixel coords (ints or floats).
    im: 2D image.
    Returns: (N,2) refined (y, x) float32.
    """
    if pred_coord is None or len(pred_coord) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    H, W = im.shape
    r = int(max(1, window_size // 2))
    sigma = float(std) if std is not None else 2.5

    refined = []
    imf = im.astype(np.float32, copy=False)
    for (y0, x0) in pred_coord.astype(int):
        y1, y2 = max(0, y0 - r), min(H, y0 + r + 1)
        x1, x2 = max(0, x0 - r), min(W, x0 + r + 1)
        if y2 <= y1 or x2 <= x1:
            refined.append([float(y0), float(x0)])
            continue
        win = imf[y1:y2, x1:x2]
        if apply_filter:
            win = gaussian_filter(win, sigma=sigma, mode="nearest")

        yy, xx = np.mgrid[y1:y2, x1:x2]
        wsum = float(win.sum())
        if wsum <= 1e-6:
            refined.append([float(y0), float(x0)])
        else:
            wy = float((win * yy).sum()) / wsum
            wx = float((win * xx).sum()) / wsum
            refined.append([wy, wx])

    return np.asarray(refined, dtype=np.float32)

# ---------------------------
# Detector (single 2D slice)
# ---------------------------

def detect_centers(
    im2d: np.ndarray,
    method: str = "log",
    min_sigma: float = 3,
    max_sigma: float = 4,
    num_sigma: int = 10,
    threshold: float = 0.05,
    overlap: float = 0.5,
    return_radii: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Detect blob centers in one slice using LoG/DoG/DoH.
    Returns centers as (N,2) in (y, x). If return_radii=True, also returns (N,) radii.
    """
    if im2d.ndim != 2:
        raise ValueError("detect_centers expects a 2D image")

    imn = _normalize01(im2d)  # stabilize threshold across slices

    if method.lower() == "log":
        blobs = blob_log(imn, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold, overlap=overlap)
        # LoG: radius ≈ sqrt(2) * sigma
        centers_yx = blobs[:, :2] if blobs.size else np.zeros((0, 2))
        radii = (blobs[:, 2] * np.sqrt(2)) if blobs.size else np.zeros((0,))
    elif method.lower() == "dog":
        blobs = blob_dog(imn, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=threshold, overlap=overlap)
        centers_yx = blobs[:, :2] if blobs.size else np.zeros((0, 2))
        radii = (blobs[:, 2] * np.sqrt(2)) if blobs.size else np.zeros((0,))
    elif method.lower() == "doh":
        blobs = blob_doh(imn, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=threshold)
        centers_yx = blobs[:, :2] if blobs.size else np.zeros((0, 2))
        # DoH returns an approx radius already
        radii = (blobs[:, 2]) if blobs.size else np.zeros((0,))
    else:
        raise ValueError("method must be one of {'log','dog','doh'}")

    centers_yx = centers_yx.astype(np.float32, copy=False)
    radii = radii.astype(np.float32, copy=False)
    return (centers_yx, radii) if return_radii else centers_yx

# ---------------------------
# Main: volume wrapper
# ---------------------------

def get_fibre_coords2(
    vol: np.ndarray, 
    method: str = "log",
    min_sigma: float = 3,
    max_sigma: float = 4,
    num_sigma: int = 10,
    threshold: float = 0.05,
    overlap: float = 0.5,
    std: float = 2.5, 
    weighted_avg: bool = False,
    window_size: int = 10,
    apply_filter: bool = False,
) -> List[np.ndarray]:
    """
    Detect fibre cross-section centres per slice of a 3D volume using a multiscale blob detector.
    Returns a list (length Z) with arrays of shape (N_i, 3) holding (x, y, z).
    """
    if vol.ndim != 3:
        raise ValueError("vol must be (Z, Y, X)")

    Z, H, W = vol.shape
    out: List[np.ndarray] = []
    for z in range(Z):
        im = vol[z]
        coords_yx = detect_centers(
            im,
            method=method,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
            return_radii=False,
        )

        # Optional subpixel refinement
        if weighted_avg and coords_yx.size > 0:
            coords_yx = avg_fibre_coord2(
                coords_yx,
                im,
                window_size=window_size,
                apply_filter=apply_filter,
                std=std,
            )

        if coords_yx.size == 0:
            out.append(np.zeros((0, 3), dtype=np.float32))
        else:
            # Convert (y,x) -> (x,y) and append z
            x = coords_yx[:, 1].clip(0, W - 1)
            y = coords_yx[:, 0].clip(0, H - 1)
            zcol = np.full_like(x, fill_value=float(z), dtype=np.float32)
            out.append(np.stack([x.astype(np.float32), y.astype(np.float32), zcol], axis=1))

        print(f"Detecting coordinates - slice: {z+1}/{Z}", end="\r")
    print(" " * 50, end="\r")
    return out
