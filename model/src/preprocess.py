"""Image preprocessing utilities (example)"""
import os, cv2

def resize_image(in_path, out_path, size=(640,480)):
    img = cv2.imread(in_path)
    if img is None:
        return False
    h,w = size
    img2 = cv2.resize(img, (w,h))
    cv2.imwrite(out_path, img2)
    return True
