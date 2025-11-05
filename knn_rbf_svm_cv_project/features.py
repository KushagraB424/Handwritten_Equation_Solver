import numpy as np
import cv2


def resize_keep_aspect(img, target=32):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target, target), dtype=np.uint8)
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def hog_descriptor(img32: np.ndarray) -> np.ndarray:
    # Manual HOG (orientations=9, cell=8x8, block=2x2, L2-Hys)
    img = img32.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0  # [0,180)

    cell_size = 8
    n_cells_y = img.shape[0] // cell_size
    n_cells_x = img.shape[1] // cell_size
    bins = 9
    bin_width = 180.0 / bins

    # Compute cell histograms
    hist = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_mag = mag[y0:y1, x0:x1]
            cell_ang = ang[y0:y1, x0:x1]
            # Vote bilinearly into adjacent bins
            for yy in range(cell_size):
                for xx in range(cell_size):
                    m = cell_mag[yy, xx]
                    a = cell_ang[yy, xx]
                    b = int(a // bin_width) % bins
                    r = (a - b * bin_width) / bin_width
                    b_next = (b + 1) % bins
                    hist[cy, cx, b] += m * (1.0 - r)
                    hist[cy, cx, b_next] += m * r

    # Block normalization (2x2 cells per block)
    by = n_cells_y - 1
    bx = n_cells_x - 1
    blocks = []
    eps = 1e-6
    for y in range(by):
        for x in range(bx):
            block = hist[y:y+2, x:x+2, :].reshape(-1)
            norm = np.linalg.norm(block) + eps
            block = block / norm
            # Hys clipping (0.2) then renormalize
            block = np.minimum(block, 0.2)
            block = block / (np.linalg.norm(block) + eps)
            blocks.append(block)
    if blocks:
        return np.concatenate(blocks).astype(np.float32)
    else:
        return np.zeros((4 * 2 * 2 * bins,), dtype=np.float32)  # Fallback shape


def intensity_features(img32: np.ndarray) -> np.ndarray:
    flat = (img32.flatten() / 255.0).astype(np.float32)
    return flat


def extract_features(roi_gray: np.ndarray) -> np.ndarray:
    # ensure binary-like foreground for consistency
    _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi_bin = cv2.bitwise_not(roi_bin)  # foreground as white
    img32 = resize_keep_aspect(roi_bin, 32)
    f_hog = hog_descriptor(img32)
    f_pix = intensity_features(img32)
    return np.concatenate([f_hog, f_pix])