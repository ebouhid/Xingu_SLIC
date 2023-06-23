'''
    create_cs_tasks.py 
    
    This code generates the png images of the Citizen Science tasks. Given an input image, subimages are created zooming on the segment of interest, that is highlighted by a yellow border.

    Input parameters:

        - seg_file:   File (extension .dat for example) that contains the segments created by a previous segmentation algorithm (as SLIC, IFT-SLIC, or maskSLIC)
        - input_img:  Image that will be used to create the Citizen Science tasks
        - output_dir: directory to save the new image (if it doesn't exist, 
                      it will be created) 

    Output:

        - png images for each segment in seg_file (Citizen Science task) 
 
    Tested in Anaconda environment with Python 3.7.16, OpenCV 4.6.0, and SciPy 1.7.3.

    Execution in command line:
    $ python create_cs_tasks.py --seg_file path/to/seg_file.dat --input_img path/to/input_img.tif --output_dir path/to/output_dir

    Example:
    $ python create_cs_tasks.py --seg_file example_tasks\iftslic.dat --input_img example_tasks\set8_753.tif --output_dir example_tasks\TASKS_false_color
'''

import os
import sys
import cv2
import argparse
import numpy as np
from scipy.ndimage import label, zoom
from tqdm import tqdm

sys.setrecursionlimit(15500)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seg_file",
                        type=str,
                        required=True,
                        help="File with image's segmentation")

    parser.add_argument("--input_img",
                        type=str,
                        required=True,
                        help="Input image to be segmented")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output directory")

    args = parser.parse_args()

    return args


def check_args(args):

    # -------------------------------------------------------------------------
    # arg = seg_file
    arg = args.seg_file

    if not os.path.isfile(arg):
        print("ERROR: File {} not found.".format(arg))
        return False

    # -------------------------------------------------------------------------
    # arg = input_img
    arg = args.input_img

    if not os.path.isfile(arg):
        print("ERROR: Input image {} not found.".format(arg))
        return False

    # -------------------------------------------------------------------------
    # arg = output_dir
    arg = args.output_dir

    if not os.path.isdir(arg):
        print("Output directory not found. Creating...")
        os.makedirs(arg)

    return True


def neighborDetection(labels, row, col, nrows, ncols, label):
    finalIndex = ncols * (nrows - 1) + (ncols - 1)
    #print row,col,label
    if row - 1 >= 0:
        index1 = ncols * (row - 1) + col
        if index1 >= 0 and index1 <= finalIndex and labels[index1] != 250:
            if labels[index1] == label:
                labels[index1] = 250
                labels = neighborDetection(labels, row - 1, col, nrows, ncols,
                                           label)
            else:
                labels[index1] = 75
    if row + 1 < nrows:
        index2 = ncols * (row + 1) + col
        if index2 >= 0 and index2 <= finalIndex and labels[index2] != 250:
            if labels[index2] == label:
                labels[index2] = 250
                labels = neighborDetection(labels, row + 1, col, nrows, ncols,
                                           label)
            else:
                labels[index2] = 75
    if col - 1 >= 0:
        index3 = ncols * row + col - 1
        if index3 >= 0 and index3 <= finalIndex and labels[index3] != 250:
            if labels[index3] == label:
                labels[index3] = 250
                labels = neighborDetection(labels, row, col - 1, nrows, ncols,
                                           label)
            else:
                labels[index3] = 75
    if col + 1 < ncols:
        index4 = ncols * row + col + 1
        if index4 >= 0 and index4 <= finalIndex and labels[index4] != 250:
            if labels[index4] == label:
                labels[index4] = 250
                labels = neighborDetection(labels, row, col + 1, nrows, ncols,
                                           label)
            else:
                labels[index4] = 75
    return labels


def getNeighbor(labels):
    nrows, ncols = labels.shape
    row, col = np.where(labels == 255)
    auxLabels = labels.copy()
    auxLabels[row[0]][col[0]] = 250
    auxLabels = auxLabels.flatten()
    auxMask = neighborDetection(auxLabels, row[0], col[0], nrows, ncols, 255)
    return auxMask.reshape(nrows, ncols)


def buildImg(image, irow, icol, frow, fcol, nbands):
    nrows = frow - irow + 1
    ncols = fcol - icol + 1
    if nrows <= 0 or ncols <= 0:
        return np.array([])
    subimg = np.zeros((nrows, ncols, nbands), dtype=int)
    countrow = 0
    for i in range(irow, frow + 1):
        countcol = 0
        for j in range(icol, fcol + 1):
            for k in range(nbands):
                subimg[countrow][countcol][k] = image[i][j][k]
            countcol += 1
        countrow += 1
    return subimg


def saveSubImage(output_dir, segVal, newimg, irows, icols, frows, fcols):
    nrows, ncols, nbands = newimg.shape
    if irows - 30 >= 0:
        firstrow = irows - 30
    else:
        firstrow = 0
    if icols - 30 >= 0:
        firstcol = icols - 30
    else:
        firstcol = 0
    if frows + 30 >= nrows:
        lastrow = nrows - 1
    else:
        lastrow = frows + 30
    if fcols + 30 >= ncols:
        lastcol = ncols - 1
    else:
        lastcol = fcols + 30

    subimg = buildImg(newimg, firstrow, firstcol, lastrow, lastcol, nbands)
    if subimg.shape[0] != 0:
        zoomimg = zoom(subimg, [5.0, 5.0, 1.0])
        cv2.imwrite(
            os.path.join(output_dir, 'task_' + str(int(segVal)) + '.png'),
            zoomimg)


def main(args):

    img = cv2.imread(args.input_img)
    segments = np.load(args.seg_file)
    unique_segments = np.unique(segments)
    unique_segments = unique_segments[unique_segments >= 0]
    nunique = unique_segments.shape[0]

    loop = tqdm(enumerate(unique_segments))

    for i, segVal in loop:
        mask = np.zeros(img.shape[:2], dtype='uint8')
        mask[segments == segVal] = 255
        newmask = getNeighbor(mask)
        newimg = img.copy()
        rows, cols = np.where(newmask == 75)
        newmask = []
        for j, k in zip(rows, cols):
            newimg[j][k] = [0, 255, 255]
        saveSubImage(args.output_dir, segVal, newimg, rows[0], cols[0],
                     rows[-1], cols[-1])


if __name__ == "__main__":

    args = parse_args()
    if check_args(args):
        ok = main(args)
