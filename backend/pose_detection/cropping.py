import numpy as np
import tensorflow as tf
from .keypoints import KEYPOINT_DICT
# 程序功能： 1. 减少输入矩阵的大小。2.输入模型识别的的结果，输出所需要的每个关键点之间的信息
# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.55#设置最小置信度


# 判断驱赶是否可见，即模型输出置信度是否足够
def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.
    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.

    Args:
        keypoints: a [17,3] keypoint numpy array
    Returns:
        True if there are enough keypoints to accurately predict the position of the torso
        False if not
    """
    return (
        keypoints[KEYPOINT_DICT["left_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[KEYPOINT_DICT["right_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
    ) and (
        keypoints[KEYPOINT_DICT["left_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[KEYPOINT_DICT["right_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
    )

#找到驱赶和身体的最大距离
def determine_torso_and_body_range(keypoints, position_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    Args:
        keypoints: a [17,3] keypoint numpy array
        position_keypoints: dictionary of keypointdata for keypoints representing the torso
        center_y: y-position of the center of the torso
        center_x: x-position of the center of the torso
    Returns:
        The maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:#遍历驱赶关节位置，找到脚距离肩膀的最大距离
        dist_y = abs(center_y - position_keypoints[joint][0])
        dist_x = abs(center_x - position_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():#key是字典中的索引。该循环遍历所有关节，找到每个关键点距离中心的最远距离
        if keypoints[KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:#如果置信度不够将跳过该帧
            continue
        dist_y = abs(center_y - position_keypoints[joint][0])
        dist_x = abs(center_x - position_keypoints[joint][1])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

#确定图像中人体的裁剪区域，以便模型可以在此区域内进行推理
#该方法主要是为了增加模型的运算效率，减少计算资源的浪费
def determine_crop_region(keypoints, image_height, image_width):

    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.

    Args:
        keypoints: a [17,3] keypoint numpy array
        image_height: the height of the image in pixels
        image_width: the width of the image in pixels
    Returns:
        Dictionary {y_min, x_min, y_max, x_max, height, width} representing the bounding box around the person.
    """
    position_keypoints = {}
    for joint in KEYPOINT_DICT.keys():#遍历17个关节，将关节位置存储在target_keypoints中
        position_keypoints[joint] = [
            keypoints[KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[KEYPOINT_DICT[joint], 1] * image_width,
        ]


    if torso_visible(keypoints):#判断驱赶是否可见，如果可见返回计算的裁切区域
        center_y = (
            position_keypoints["left_hip"][0] + position_keypoints["right_hip"][0]
        ) / 2
        center_x = (
            position_keypoints["left_hip"][1] + position_keypoints["right_hip"][1]
        ) / 2

        (
            max_torso_yrange,
            max_torso_xrange,
            max_body_yrange,
            max_body_xrange,
        ) = determine_torso_and_body_range(
            keypoints, position_keypoints, center_y, center_x
        )

        crop_length_half = np.amax(
            [
                max_torso_xrange * 1.9,
                max_torso_yrange * 1.9,
                max_body_yrange * 1.2,
                max_body_xrange * 1.2,
            ]
        )

        tmp = np.array(
            [center_x, image_width - center_x, center_y, image_height - center_y]
        )
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                "y_min": crop_corner[0] / image_height,
                "x_min": crop_corner[1] / image_width,
                "y_max": (crop_corner[0] + crop_length) / image_height,
                "x_max": (crop_corner[1] + crop_length) / image_width,
                "height": (crop_corner[0] + crop_length) / image_height
                - crop_corner[0] / image_height,
                "width": (crop_corner[1] + crop_length) / image_width
                - crop_corner[1] / image_width,
            }
    else:
        return init_crop_region(image_height, image_width)
#默认裁切区域
def init_crop_region(image_height, image_width):#默认裁切区域
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.

    Args:
        image_height: the height of the image in pixels
        image_width: the width of the image in pixels
    Returns:
        Dictionary {y_min, x_min, y_max, x_max, height, width} representing the bounding box around the person.
    """
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        "y_min": y_min,
        "x_min": x_min,
        "y_max": y_min + box_height,
        "x_max": x_min + box_width,
        "height": box_height,
        "width": box_width,
    }
#调整图像的大小，使其完全适配模型输入的需要
def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input.

    Args:
        image: the image as a [H,W,3] tensor
        crop_region: the dictionary representing the bounding box used to crop the image around the cyclist
        crop_size: the size of the bounding box
    Returns:
        an image as a [256, 256, 3] tensor, cropped around the cyclist and resized to the correct input size"""
    boxes = [
        [
            crop_region["y_min"],
            crop_region["x_min"],
            crop_region["y_max"],
            crop_region["x_max"],
        ]
    ]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size
    )
    return output_image
