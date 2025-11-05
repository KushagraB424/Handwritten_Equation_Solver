import cv2
import numpy as np
from typing import List, Dict, Tuple


def connected_components(bin_img: np.ndarray):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    comps = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area < 50:
            continue
        comps.append({'bbox': (int(x), int(y), int(w), int(h)), 'centroid': (float(cx), float(cy)), 'area': int(area)})
    return comps, labels


def find_contours(bin_img: np.ndarray):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 50:
            continue
        boxes.append({'bbox': (x, y, w, h), 'area': area})
    # sort left-to-right, then top-to-bottom
    boxes.sort(key=lambda b: (b['bbox'][0], b['bbox'][1]))
    return boxes


def merge_equal_signs(comps: List[Dict]) -> List[Dict]:
    """
    Merge two horizontally-aligned components that are close together
    (likely the two bars of an equal sign) into a single component.
    """
    if len(comps) < 2:
        return comps
    
    merged = []
    skip = set()
    
    for i, comp1 in enumerate(comps):
        if i in skip:
            continue
            
        x1, y1, w1, h1 = comp1['bbox']
        cx1, cy1 = comp1['centroid']
        
        # Look for a nearby component that could be the other bar of '='
        merged_with = None
        for j in range(i + 1, len(comps)):
            if j in skip:
                continue
                
            comp2 = comps[j]
            x2, y2, w2, h2 = comp2['bbox']
            cx2, cy2 = comp2['centroid']
            
            # Check if components are horizontally aligned and close vertically
            horizontal_overlap = not (x1 + w1 < x2 or x2 + w2 < x1)
            x_distance = abs(cx1 - cx2)
            y_distance = abs(cy1 - cy2)
            
            # Heuristics for detecting equal sign bars:
            # 1. Vertically close (within 1.5x their average height)
            # 2. Similar widths (within 2x ratio)
            # 3. Both are relatively flat (width > 1.5 * height)
            avg_h = (h1 + h2) / 2
            width_ratio = max(w1, w2) / (min(w1, w2) + 1e-6)
            is_flat1 = w1 > 1.5 * h1
            is_flat2 = w2 > 1.5 * h2
            
            if (y_distance < 1.5 * avg_h and 
                width_ratio < 2.0 and 
                is_flat1 and is_flat2 and
                x_distance < max(w1, w2) * 1.5):  # horizontally aligned
                
                # Merge the two components
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_x2 = max(x1 + w1, x2 + w2)
                new_y2 = max(y1 + h1, y2 + h2)
                new_w = new_x2 - new_x
                new_h = new_y2 - new_y
                new_cx = (cx1 + cx2) / 2
                new_cy = (cy1 + cy2) / 2
                new_area = comp1['area'] + comp2['area']
                
                merged_with = {
                    'bbox': (new_x, new_y, new_w, new_h),
                    'centroid': (new_cx, new_cy),
                    'area': new_area,
                    'merged': True  # flag to indicate this was merged
                }
                skip.add(j)
                break
        
        if merged_with:
            merged.append(merged_with)
        else:
            merged.append(comp1)
    
    return merged


def extract_rois(gray_img: np.ndarray, bin_img: np.ndarray) -> List[Dict]:
    comps, _ = connected_components(bin_img)
    
    # use components as base; fallback to contours if empty
    if not comps:
        boxes = find_contours(bin_img)
        comps = [{'bbox': b['bbox'], 'centroid': (b['bbox'][0]+b['bbox'][2]/2, b['bbox'][1]+b['bbox'][3]/2), 'area': int(b['area'])} for b in boxes]
    
    # Merge components that likely form an equal sign
    comps = merge_equal_signs(comps)
    
    # compute baseline (median of centroid y)
    baseline_y = np.median([c['centroid'][1] for c in comps]) if comps else 0
    rois = []
    for c in comps:
        x, y, w, h = c['bbox']
        roi = gray_img[y:y+h, x:x+w]
        rois.append({
            'roi': roi,
            'bbox': (x, y, w, h),
            'centroid': c['centroid'],
            'area': c['area'],
            'baseline_y': baseline_y,
            'is_superscript': c['centroid'][1] < baseline_y - (0.25 * h)
        })
    # order by x (reading order)
    rois.sort(key=lambda r: (r['bbox'][1]//10, r['bbox'][0]))
    return rois