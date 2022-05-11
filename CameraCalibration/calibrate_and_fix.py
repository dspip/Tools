import numpy as np
import cv2
import os
import yaml
import argparse
import pandas as pd
import pickle as pkl
from PIL import Image, ImageEnhance
from glob import glob

def calibrate(dir_path, find_corners_shape, base_corners_shape, calib_files_dir="./calibration_stats", remove_failure=True):
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((find_corners_shape[0]*find_corners_shape[1], 3), np.float32)
    objp[:,:2] = np.mgrid[:find_corners_shape[0], :find_corners_shape[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob(os.path.join(dir_path, "*.jpg"))
    h, w = None, None

    for fname in images:
        # Read Checkboard image
        img = Image.open(fname)

        # Optional fixing augmentations
        enhance_contrast = ImageEnhance.Contrast(img)
        enhance_sharpen = ImageEnhance.Sharpness(img)
        img = enhance_contrast.enhance(2)
        img = enhance_sharpen.enhance(2)
        img = np.array(img)
        #img = cv2.resize(img,(2160,1632))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        #Image.fromarray(gray).show()

        ret, corners =  cv2.findChessboardCorners(gray, find_corners_shape, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), subpix_criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            cv2.imwrite('res.jpg',cv2.drawChessboardCorners(img, (7,9), corners2, ret))
        else:
            if remove_failure:
                os.system(f"rm {fname}")
                print(f"Failed to find corners for {fname}, file removed")
            else:
                print(f"Failed to find corners for {fname}")

    #cv2.destroyAllWindows()

    # calculate K & D
    N_imm = len(images) # number of calibration images
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # retval, K, D, rvecs, tvecs
    #objpoints = np.expand_dims(np.asarray(objpoints), -2)
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if calib_files_dir is not None:
        if not os.path.exists(calib_files_dir):
            os.makedirs(calib_files_dir)
        np.save(os.path.join(calib_files_dir, f"K_{w}X{h}"), mtx)
        np.save(os.path.join(calib_files_dir, f"D_{w}X{h}"), dist)

    #return retval, K, D, rvecs, tvecs
    return ret, mtx, dist, rvecs, tvecs


def test_image(in_path, out_path, mtx, dist):
    img = cv2.imread(in_path)
    h, w = img.shape[:-1]

    # Undistort the test image
    print("Undistorting the test image...")
    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, [w,h], cv2.CV_16SC2)
    #st = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dst = cv2.undistort(img,mtx,dist,None,None)

    cv2.imwrite(out_path, dst)
    print(f"Result image saved to {out_path}")


def test_video(source, out_path, mtx, dist, width, height, fps):
    cap = cv2.VideoCapture(source)
    # cap = cv2.VideoCapture(source, apiPreference=cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Cannot open video source {source}")
        exit()

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width, height))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    elif isinstance(source, str):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m','p','4','v'))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        print(frame.shape[:-1])
        if frame.shape[:-1] != (height, width):
            print("Invalid arguments width/height")
            break

        # Undistort the test frame
        print("Undistorting the test frame...")
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, frame.shape[:2][::-1], cv2.CV_16SC2)
        dst = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Display the resulting frame
        cv2.imshow('frame', dst)

        # write the flipped frame
        writer.write(dst)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(calibration_images_dir, test_path, find_corners_shape, base_corners_sape,
         out_path, test_source, width, height, fps, calib_files_dir=None):

    mtx, dist = None, None
    print("Getting calibration parameters...")
    if calib_files_dir is None:
        base_corners_sape = tuple(base_corners_sape)
        find_corners_shape = tuple(find_corners_shape)
        ret, mtx, dist, rvecs, tvecs = calibrate(calibration_images_dir, find_corners_shape, base_corners_sape)
    else:
        K_name = os.path.join(calib_files_dir, f"K_{width}X{height}.npy")
        D_name = os.path.join(calib_files_dir, f"D_{width}X{height}.npy")
        assert os.path.exists(K_name) and os.path.exists(D_name), "Calibration files do not exist"
        mtx = np.load(K_name)
        dist = np.load(D_name)

    if test_source == "image":
        test_image(test_path, out_path, mtx, dist)
    elif test_source == "video":
        test_video(test_path, out_path, mtx, dist, width, height, fps)
    elif isinstance(test_source, int):
        test_video(test_source, out_path, mtx, dist, width, height, fps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Calibration Module')
    parser.add_argument('-c', '--config_path', default=None, type=str,
        help="Path to a configuration .yaml file")

    args = parser.parse_args()
    assert args.config_path is not None, "Invalid config_path"

    with open(args.config_path, 'r') as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)
        main(cfg["calibration_images_dir"], cfg["test_path"], cfg["find_corners_shape"],
            cfg["base_corners_shape"], cfg["out_path"], cfg["test_source"], cfg["width"],
            cfg["height"], cfg["fps"], cfg["calibration_files_dir"])
