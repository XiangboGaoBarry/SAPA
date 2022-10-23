import argparse
import os
import sys
import pdb

import cv2
from joblib import load

from utils.quality_assessment.brisque import brisque
from utils.quality_assessment.niqe import niqe
from utils.quality_assessment.piqe import piqe

parser = argparse.ArgumentParser('Test an image')
parser.add_argument(
    '--mode', choices=['brisque', 'niqe', 'piqe'], help='iqa algorithoms,brisque or niqe or piqe')
parser.add_argument('--path', required=True, help='image path')
args = parser.parse_args()

# if __name__ == "__main__":
#     '''
#     test conventional blindly image quality assessment methods(brisque/niqe/piqe)
#     '''
#     mode = args.mode
#     path = args.path
#     im = cv2.imread(path)
#     if im is None:
#         print("please input correct image path!")
#         sys.exit(0)
#     if mode == "piqe":
#         score, _, _, _ = piqe(im)
#     elif mode == "niqe":
#         score = niqe(im)
#     elif mode == "brisque":
#         feature = brisque(im)
#         feature = feature.reshape(1, -1)
#         clf = load('utils/quality_assessment/svr_brisque.joblib')
#         score = clf.predict(feature)[0]
#     print("{}-----{} score:{}".format(path, mode, score))

if __name__ == "__main__":
    '''
    test conventional blindly image quality assessment methods(brisque/niqe/piqe)
    '''
    mode = args.mode
    path = args.path
    p_score_sum = n_score_sum = b_score_sum = 0
    
    for idx, filename in enumerate(os.listdir(path)):
        im = cv2.imread(f"{filename}/{path}")
        p_score, _, _, _ = piqe(im)
        p_score_sum += p_score
        n_score = niqe(im)
        n_score_sum += n_score
        feature = brisque(im)
        feature = feature.reshape(1, -1)
        clf = load('utils/quality_assessment/svr_brisque.joblib')
        b_score = clf.predict(feature)[0]
        b_score_sum += b_score
        print("{}----- piqe:{}, niqe:{}, brisque:{}".format(path, p_score, n_score, b_score))
    print("{} avg:  piqe:{}, niqe:{}, brisque:{}".format(path, p_score/(idx+1), n_score/(idx+1), b_score/(idx+1)))

