"""
The project is originally created on 01-Sep-2020 by Eric.

@author: Junxiang(Eric) JIA

***** Version 2.0 *****
The version 2.0 is updated on 21-Aug-2021. Compare to version 1.0, the update version 2.0 uses a different
approach (OpenCV rather than customized pattern recognation algorithm).
"""

import numpy as np
import cv2
import sxmReader


class NanonisSXM:

    def __init__(self, fname):
        self.fname = fname
        self.load = sxmReader.NanonisSXM(fname)
        self.header = self.load.header
        self.pixel_x = self.load.size['pixels']['x']; self.pixel_y = self.load.size['pixels']['y'];
        self.len_x = self.load.size['real']['x'] * 1e9; self.len_y = self.load.size['real']['y'] * 1e9;
        self.scan_dir = self.load.header['SCAN_DIR'][0][0]

    def extract_image(self, channel='Z'):
        '''Extract topography image of the Z channel w/o any rotation. This method takes 'Analysis.py'
        as the reference.'''
        xx = self.load.get_channel(channel)
        if self.scan_dir == 'up':
            Map = xx
        else:
            Map = np.flip(xx, axis=0)

        return Map


def return_img(fname):
    aa = NanonisSXM(fname)
    img = aa.extract_image()
    return img, aa.len_x, aa.len_y


def matching_cal(fname1, fname2, display, threshold, Matching_algor='TM_CCOEFF_NORMED'):
    a1 = NanonisSXM(fname1)
    a2 = NanonisSXM(fname2)
    img1_len_x = a1.len_x
    img1_len_y = a1.len_y
    img2_len_x = a2.len_x
    img2_len_y = a2.len_y
    img1 = a1.extract_image()
    img2 = a2.extract_image()
    img1 = np.flip(img1, axis=0)
    img2 = np.flip(img2, axis=0)
    img1_norm = np.zeros_like(img1)
    img1_norm = cv2.normalize(img1,  img1_norm, 0, 1, cv2.NORM_MINMAX)
    img2_norm = np.zeros_like(img2)
    img2_norm = cv2.normalize(img2,  img2_norm, 0, 1, cv2.NORM_MINMAX)

    img1_norm = cv2.resize(img1_norm, (int(img1_len_x*800/img1_len_y), 800))  # normalize the height of the large image to 800px.
    img2_norm = cv2.resize(img2_norm, (int(img2_len_x*800/img1_len_y), int(img2_len_y*800/img1_len_y)))
    img1_norm = img1_norm.astype(np.float32)
    img2_norm = img2_norm.astype(np.float32)

    if Matching_algor == 'TM_CCOEFF_NORMED (default)':
        result = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCOEFF_NORMED)
    elif Matching_algor == 'TM_CCOEFF':
        result = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCOEFF)
        result /= np.max(result)
    elif Matching_algor == 'TM_CCORR_NORMED':
        result = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCORR_NORMED)
        result /= np.max(result)
    elif Matching_algor == 'TM_CCORR':
        result = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCORR)
        result /= np.max(result)
    else:
        print('No available matching algorithm!')
        return 0

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    h, w = img2_norm.shape

    if display == 'All':
        img1_norm = cv2.cvtColor(img1_norm, cv2.COLOR_GRAY2RGB)

        yloc, xloc = np.where(result >= threshold)

        if len(xloc) >= 10000:
            return 1

        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w), int(h)])
            rectangles.append([int(x), int(y), int(w), int(h)])

        rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

        for (x, y, w, h) in rectangles:
            cv2.rectangle(img1_norm, (x, y), (x+w, y+h), (0, 255, 255), 1)

    cv2.rectangle(img1_norm, max_loc, (max_loc[0]+w, max_loc[1]+h), (0, 0, 255), 2)
    if Matching_algor == 'TM_CCOEFF_NORMED (default)':
        cv2.putText(img1_norm, '{:.0f}%'.format(max_val*100), (max_loc[0], max_loc[1]-3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

    cv2.imshow('Matching Results', img1_norm)
    cv2.waitKey()
    return 0
