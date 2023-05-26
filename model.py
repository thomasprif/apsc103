import os
import socket
import json
# device's IP address
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

s = socket.socket()
s.bind((SERVER_HOST, SERVER_PORT))
s.listen(10)
print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")

# accept connection if there is any
client_socket, address = s.accept() 
# if below code is executed, that means the sender is connected
print(f"[+] {address} is connected.")




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Starting tensorflow", end="  ", flush=True)
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
print("Done")


tf.get_logger().setLevel('ERROR') 

#Checkes for GPUS on system should the client ever use one
print("Looking for GPUs", flush=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True);
if len(gpus) == 0:
    print("No GPUS found")

#configures the pipleline for the model ex: training, test, production
print("Configuring model pipeline", end="  ", flush=True)
PATH_TO_CFG = os.path.join("model", "pipeline.config")
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
print("Done")


#Loads the labels for the model
print("Configuring model labels", end="  ", flush=True)
PATH_TO_LABELS = os.path.join("model", "mscoco_label_map.pbtxt")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)
print("Done")

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
PATH_TO_CKPT = os.path.join("model", "checkpoint")
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()



#Loads video stream
print("Starting video stream", end="  ", flush=True)
import cv2
camera = cv2.VideoCapture(0)
print("Done")

@tf.function
def detect_object(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def calculate_box_coordinates(classes, boxes, scores, category_index):

    classes = np.array(classes)
    boxes = np.array(boxes)
    scores = np.array(scores)


    # Filter scores above 0.3
    above_threshold_indices = np.where(scores > 0.3)
    filtered_classes = classes[above_threshold_indices]
    filtered_boxes = boxes[above_threshold_indices]

    class_names = [category_index[c]['name'] for c in filtered_classes]
    
    # Calculate box coordinates
    box_coordinates = []
    for box in filtered_boxes:
        ymin, xmin, ymax, xmax = box
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax
        box_coordinates.append((x1, y1, x2, y2))
    
    return class_names, box_coordinates



def find_overlap(obj_names, obj_boxes, box_coords):
    overlap = []
    for i in range(len(obj_names)):
        if obj_boxes[i][0] < float(box_coords["tl"]["y"]) and obj_boxes[i][2] > float(box_coords["br"]["y"]) \
        and obj_boxes[i][1] < float(box_coords["tl"]["x"]) and obj_boxes[i][3] > float(box_coords["br"]["x"]):
            overlap.append(obj_names[i])
    return overlap



#Model itself
client_socket.send(bytes(f"Nothing{SEPARATOR}green{SEPARATOR}", encoding="utf-8"))

with open('default.jpg', 'rb') as f:
    image_data = f.read()
client_socket.recv(BUFFER_SIZE)
client_socket.sendall(image_data)
client_socket.recv(BUFFER_SIZE)
print("done", flush=True)

import numpy as np
while True:
    ret, frame = camera.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("test1", flush=True)

    #expand dimension of image
    frame_expanded = np.expand_dims(frame, axis=0)

    tf_input = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_object(tf_input)

    label_id_offset = 1
    frame_with_boxes = frame.copy()

    print("test2", flush=True)

    viz_utils.visualize_boxes_and_labels_on_image_array(
          frame_with_boxes,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    classes = tf.keras.backend.get_value(detections["detection_classes"][0])
    classes = [x+1 for x in classes]
    boxes = tf.keras.backend.get_value(detections["detection_boxes"][0])
    scores = tf.keras.backend.get_value(detections["detection_scores"][0])
    objects, coords = calculate_box_coordinates(classes, boxes, scores, category_index)
    #print(objects, coords)

    print("test", flush=True)
    received = client_socket.recv(BUFFER_SIZE).decode()[0:133]
    print(received)
    boxCoords= json.loads(received)
    #print(boxCoords)

    objects_in_frame = find_overlap(objects, coords, boxCoords)
    if len(objects_in_frame) == 0 and len(objects) == 0:
        client_socket.send(bytes(f"Nothing{SEPARATOR}green{SEPARATOR}", encoding="utf-8"))
    elif len(objects_in_frame) == 0:
        client_socket.send(bytes(f"{objects[0]}{SEPARATOR}yellow{SEPARATOR}", encoding="utf-8"))
    else:
        client_socket.send(bytes(f"{objects_in_frame[0]}{SEPARATOR}red{SEPARATOR}", encoding="utf-8"))

    print("data sent", flush=True)

    client_socket.recv(BUFFER_SIZE)

    cv2.imwrite("test.jpg", frame_with_boxes)

    with open('test.jpg', 'rb') as f:
        image_data = f.read()

    client_socket.sendall(image_data)

    print("image sent", flush=True)

    client_socket.recv(BUFFER_SIZE)


    cv2.imshow('object detection', cv2.resize(frame_with_boxes, (800, 600)))
  




# After the loop release the cap object
camera.release()
# Destroy all the windows
cv2.destroyAllWindows()
