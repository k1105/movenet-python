import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def movenet(input_image, interpreter=None, module=None):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.
      interpreter: TFLite interpreter object.
      module: TF Hub module.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    if interpreter is not None:
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    elif module is not None:
        model = module.signatures['serving_default']
        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
    else:
        raise ValueError("Either interpreter or module must be provided.")
    
    return keypoints_with_scores

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.30):
    """Returns high confidence keypoints for visualization.

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      keypoints_xy: A numpy array containing the coordinates of all keypoints.
    """
    keypoints_all = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    return keypoints_xy

def visualize_keypoints(image, keypoints_with_scores):
    """Visualizes the keypoints on the image.

    Args:
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
    """
    height, width, _ = image.shape
    keypoints_xy = _keypoints_and_edges_for_display(keypoints_with_scores, height, width)

    for keypoint in keypoints_xy:
        cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 5, (0, 0, 255), -1)  # Red dots for keypoints
    return image

def save_keypoints_to_json(keypoints_list, filename='keypoints.json'):
    with open(filename, 'w') as f:
        json.dump(keypoints_list, f)

model_name = "movenet_lightning"
interpreter = None
module = None

if "tflite" in model_name:
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
else:
    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Initialize a list to store all keypoints
all_keypoints = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and pad the frame to keep the aspect ratio and fit the expected size.
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    
    # Run model inference.
    keypoints_with_scores = movenet(input_image, interpreter, module)

    # Visualize the keypoints.
    output_frame = visualize_keypoints(frame, keypoints_with_scores)

    # Add the keypoints to the list
    keypoints_list = keypoints_with_scores[0, 0].tolist()
    frame_keypoints_data = {name: keypoints_list[idx] for name, idx in KEYPOINT_DICT.items()}
    all_keypoints.append(frame_keypoints_data)

    # Display the frame.
    cv2.imshow('MoveNet Lightning', output_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Save all keypoints to a JSON file when the loop is exited
save_keypoints_to_json(all_keypoints)

cap.release()
cv2.destroyAllWindows()
