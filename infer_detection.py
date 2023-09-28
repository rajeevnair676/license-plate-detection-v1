import onnxruntime
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img
import time

opt_session = onnxruntime.SessionOptions()
opt_session.enable_mem_pattern = False
opt_session.enable_cpu_mem_arena = False
opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

model_path = 'Model\\best2.onnx'
EP_list = ['CPUExecutionProvider']

def preprocess(image_path):

    ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]

    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    input_height, input_width = input_shape[2:]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2,0,1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
    input_tensor.shape

    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
    return outputs,input_height, input_width,image_height, image_width

def predict(outputs,conf_threshold=0.6):
    predictions = np.squeeze(outputs).T
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]
    return predictions,scores

def bboxes(predictions,input_height, input_width,image_height, image_width):
    boxes = predictions[:, :4]

    #rescale box
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)
    return boxes
    
def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def visualize(image_draw,score,boxes,indices):
    cv2.rectangle(image_draw,tuple(xywh2xyxy(boxes[indices])[0][:2]),tuple(xywh2xyxy(boxes[indices])[0][2:]),(0,255,0),2)
    # cv2.putText(image_draw, "class",
    #             (xywh2xyxy(boxes[indices])[0][0], xywh2xyxy(boxes[indices])[0][1] - 2),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.60, [225, 255, 255],
    #             thickness=3)
    cv2.imshow('License plate',image_draw)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

def extract_image(img,ymin,xmin,ymax,xmax):
    # img = np.array(load_img(img_path))
    # xmin ,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax]
    cv2.imwrite("ocr_input\\image1.jpeg",roi)
    # cv2.imshow('Extracted image',roi)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image


def main():
    PATH = 'TEST\Images\TEST.jpeg'
    start = time.time()
    print("Start: ",start)
    
    outputs,input_height, input_width,image_height, image_width=preprocess(PATH)
    pre = time.time()
    print("Preprocess time: ",pre-start)
    predictions,scores = predict(outputs,conf_threshold=0.1)
    pred = time.time()
    print("Predict time: ",pred-pre)
    boxes = bboxes(predictions,input_height, input_width,image_height, image_width)
    bx = time.time()
    print("Bbox time: ",bx-pred)
    indices = nms(boxes, scores, 0.3)
    nmst = time.time()
    print("NMS time: ",nmst-bx)
    # print("Prediction ",predictions)
    # print("Scores :",scores)
    img = cv2.imread(PATH)
    
    if predictions.size>0:
        visualize(img,scores,boxes,indices)
        xmin = tuple(xywh2xyxy(boxes[indices])[0][:2])[0]
        ymin = tuple(xywh2xyxy(boxes[indices])[0][:2])[1]
        xmax = tuple(xywh2xyxy(boxes[indices])[0][2:])[0]
        ymax = tuple(xywh2xyxy(boxes[indices])[0][2:])[1]
        print("Cordinate time",time.time()-nmst)
        print("Xmin",xmin,
          "Ymin",ymin,
          "Xmax",xmax,
          "Ymax",ymax)
        print("Total time: ",time.time()-start)
        extract_image(img,ymin,xmin,ymax,xmax)
    else:
        print("No license plate found. Try lowering the confidence threshold")


if __name__=="__main__":
    main()