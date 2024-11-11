from typing import Tuple

import numpy as np
import cv2

import rospy


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    try:
        max_rate = rospy.get_param('/steering_max/left', -0.3)
    except:
        max_rate = -0.3
    steer_matrix_left = np.zeros(shape)
    steer_matrix_left[0:round(shape[0]/2), 0:round(shape[1]/2)] = np.tile(np.arange(max_rate, 0, (-1*max_rate)/round(shape[1]/2)), (round(shape[0]/2),1))
    steer_matrix_left[round(shape[0]/2):, 0:round(shape[1]/2)] = np.tile(np.arange(0, max_rate, (max_rate)/round(shape[1]/2)), (shape[0]-round(shape[0]/2),1))
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    try:
        max_rate = rospy.get_param('/steering_max/right', 0.1)
    except:
        max_rate = 0.1
    steer_matrix_right = np.zeros(shape)
    steer_matrix_right[round(shape[0]/4)*2:round(shape[0]/4)*3, round(shape[1]/2):] = np.tile(np.arange(0, max_rate, (max_rate)/round(shape[1]/2)), (round(shape[0]/4),1))
    steer_matrix_right[round(shape[0]/4)*3:, round(shape[1]/2):] = np.tile(np.arange(max_rate, 0, (-1*max_rate)/round(shape[1]/2)), (round(shape[0]/4),1))
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sigma = 3 # CHANGE ME

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # TODO: implement your own solution here
    H = np.array([-4.137917960301845e-05, -0.00011445854191468058, -0.1595567007347241, 
                    0.0008382870319844166, -4.141689222457687e-05, -0.2518201638170328, 
                    -0.00023561657746150284, -0.005370140574116084, 0.9999999999999999])

    H = np.reshape(H,(3, 3))
    Hinv = np.linalg.inv(H)
    
    world_coordinate_homogeneous = np.array([[2,0,1]]).T
    image_coordinate_homogeneous = Hinv.dot(world_coordinate_homogeneous)
    image_coordinate_cartesian = image_coordinate_homogeneous[:2] / image_coordinate_homogeneous[2]

    mask_ground = np.ones_like(img).astype(np.uint8)
    mask_ground[0:round(image_coordinate_cartesian[1][0])] = 0

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    threshold = 30
    mask_mag = (Gmag > threshold)

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    white_lower_hsv = np.array([100, 20, 230])         
    white_upper_hsv = np.array([130, 60, 255])    

    yellow_lower_hsv = np.array([20, 90, 165])
    yellow_upper_hsv = np.array([38, 140, 215])

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
