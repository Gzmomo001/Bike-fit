import tensorflow as tf
import tensorflow_hub as hub
import kagglehub
from cropping import init_crop_region, determine_crop_region, crop_and_resize

#加载模型
def load_model_from_tfhub():
    input_size = 256

    path = kagglehub.model_download("google/movenet/tensorFlow2/singlepose-thunder")
    module = hub.load(path)

    model = module.signatures["serving_default"]
    # this prevents the python 3.8.x garbage collector from
    # deleting variable references if the model is stored in memory for a long time
    model._backref_to_saved_model = module
    return model, input_size
#模型推理函数
def _movenet(model, input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.
    Returns:
      A [17, 3] float numpy array representing the predicted keypoint
      coordinates and scores. The keypoint-order is shown in KEYPOINT_DICT.
      The 3 results are {y, x, confidence}.
    """
    # SavedModel format expects tensor type of int32.
    outputs = model(input=tf.cast(input_image, dtype=tf.int32))
    # output_0 is a [1,1,17, 3] array
    keypoint_with_scores = outputs["output_0"].numpy()
    return keypoint_with_scores.squeeze()

#通过模型计算找到每个关键点的位置和置信度。格式：x,y,置信度
def _run_inference(model, image, crop_region, crop_size):
    """Runs model inferece on the cropped region. The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.

    Args:
      model: the model to use
      image: the image to run inference on
      crop_region: the region of the image to crop the image to
      crop_size: the size
    Returns:
      (17,3) array of the keypoints
    """
    #
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(
        tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size
    )#裁切原始图像
    # Run model inference.
    keypoints_with_scores = _movenet(model, input_image)
    # Update the coordinates.
    for idx in range(17):#遍历每个关键点，根据裁剪区域的参数将其坐标转换回原始图像的坐标系
        keypoints_with_scores[idx, 0] = (
            crop_region["y_min"] * image_height
            + crop_region["height"] * image_height * keypoints_with_scores[idx, 0]
        ) / image_height
        keypoints_with_scores[idx, 1] = (
            crop_region["x_min"] * image_width
            + crop_region["width"] * image_width * keypoints_with_scores[idx, 1]
        ) / image_width
    return keypoints_with_scores# 返回包含17个关键点坐标的数组，每个关键点包含y坐标、x坐标和置信度
#找到每一帧的关键点数据
def get_keypoints_from_video(video_tensor, model, input_size):
    """
    从视频中提取关键点。

    参数：
    - video_tensor：一个包含视频数据的张量，其shape为(num_frames, height, width, channels)。
    - model：用于关键点检测的模型。
    - input_size：模型输入的尺寸。

    返回值：
    - all_keypoints_with_scores：一个列表，包含每一帧的关键点及其分数。
    """

    # 获取视频张量的形状信息
    num_frames, video_height, video_width, _ = video_tensor.shape

    # 初始化一个列表来存储所有帧的关键点及其分数
    all_keypoints_with_scores = []

    # 初始化裁剪区域，覆盖整个视频帧
    crop_region = init_crop_region(video_height, video_width)

    # 遍历每一帧
    for frame_idx in range(num_frames):
        # 对当前帧运行推理，获取关键点及其分数，并添加到列表中
        all_keypoints_with_scores.append(
            _run_inference(
                model,
                video_tensor[frame_idx, :, :, :],
                crop_region,
                crop_size=[input_size, input_size],
            )
        )

        # 根据当前帧的关键点，确定下一帧的裁剪区域。这个操作可以使模型的越来越聚焦，提高运算效率
        crop_region = determine_crop_region(
            all_keypoints_with_scores[frame_idx], video_height, video_width
        )

    # 返回所有帧的关键点及其分数
    return all_keypoints_with_scores

