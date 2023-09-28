import cv2
import numpy as np
from ultralytics import YOLO
import os
from paddleocr import PaddleOCR

output_path = 'C:\\Users\\rajee\\Python_projects\\License_plate_detection_with_OCR\\Outputs\\Video_detection'

model = YOLO('C:\\Users\\rajee\\Python_projects\\License_plate_detection_with_OCR\\Models\\License plate detection\\best_licensedetect.pt')
ocr = PaddleOCR(use_angle_cls=True,lang='en')
cap = cv2.VideoCapture('C:\\Users\\rajee\\Python_projects\\License_plate_detection_with_OCR\\TEST\\Videos\\TEST.mp4')

frame_no = 0
while cap.isOpened():
    success,frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    seconds = current_frame_number/fps
    videotime = str(round(cap.get(cv2.CAP_PROP_POS_MSEC)))

    print("Frame no: ",frame_no, " Video time is",videotime)
    
    if success:
        results = model(frame)
        classes = results[0].boxes.cls
        annotated_frame = results[0].plot()
        cv2.namedWindow('Inference',cv2.WINDOW_NORMAL)
        cv2.imshow("Inference",annotated_frame)
        if classes.nelement()!=0:
            bbox=[]
            for index,clas in enumerate(classes):
                if clas == 50.0:
                    bbox.append(results[0].boxes.xyxy[index].cpu().numpy())
            for m in range(len(bbox)):
                bbox_arr = bbox[m]
                roi = frame[int(bbox_arr[1]):int(bbox_arr[3]), int(bbox_arr[0]):int(bbox_arr[2])]
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                # magic_color = apply_brightness_contrast(gray, brightness=70, contrast=90)
                ocr_img_name = f'detect_{m}_{videotime}.jpg'
                full_img_name = os.path.join(output_path,ocr_img_name)
                cv2.imwrite(full_img_name,gray)
                ocr_results = ocr.ocr(full_img_name)
                pred_list=[]
                if ocr_results[0]:
                    if len(ocr_results[0]) > 1:
                        preds=""
                        for k in range(len(ocr_results[0])):
                            preds+= " " +str(ocr_results[0][k][1][0])
                        pred_list.append(preds)
                    else:
                        preds = str(ocr_results[0][0][1][0])
                        pred_list.append(preds)
                    
                    print(f"Prediction for {frame_no}-{videotime}-{m} : {preds}")
                else:
                    print(f"No detections found in {frame_no}-{videotime}-{m}")
        # else:
        #     print(f"No detections found for image {videotime} - {frame_no}")
        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

    frame_no +=1

cap.release()
cv2.destroyAllWindows()