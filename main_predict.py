import cv2
import os
from ultralytics import YOLO
from PIL import Image
import time
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

ocr = PaddleOCR(use_angle_cls=True,lang='en',verbose=False)

CAR_MODEL_PATH = 'Model\\Car detection\\best.pt'
PLATE_DETECT_PATH = 'Model\\License plate detection\\best_torch.pt'
INPUT_CAR_PATH = 'TEST\\Images\\'
OUTPUT_CAR_PATH = 'ocr_input\\Car_detect_input'
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_DETECT_PATH)
OUT_PATH = 'ocr_input\\License_plates\\'


def process_input(input_car_path):
    image_path = []
    image_batch = []
    test_images = os.listdir(input_car_path)
    for image in test_images:
        im_path = os.path.join(input_car_path,image)
        im = Image.open(im_path)
        image_batch.append(im)
        image_path.append(im_path)
    return image_batch,image_path


def detect_cars(input_car_path,model,output_path):
    image_batch,image_path = process_input(input_car_path)
    s_time = time.time()
    result = model(image_batch)
    res_time = time.time()
    print(f"Model Inference time for {len(image_batch)} images:",round(res_time-s_time,2),"seconds")
    for i in range(len(image_batch)):
        bbox = result[i].boxes.xyxy
        res_plot = result[i].plot()
        # cv2.imshow(f'Detect-{image_path[i]}',res_plot)
        # cv2.waitKey(0)
        if bbox.nelement() != 0:
            imdis = cv2.imread(image_path[i])
            file_name = os.path.split(image_path[i])[1]
            for m in range(len(bbox)):
                bbox_arr = bbox[m].cpu().numpy()
                roi = imdis[int(bbox_arr[1]):int(bbox_arr[3]), int(bbox_arr[0]):int(bbox_arr[2])]
                # roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                # gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                # magic_color = apply_brightness_contrast(gray, brightness=40, contrast=40)
                img_name = f'car-{m}-{file_name}'
                main_path = os.path.join(output_path,img_name)
                cv2.imwrite(main_path,roi)

    return len(image_batch)
                

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    return buf 


def license_detect(input_plate_path,plate_model,output_path):
    image_batch2,image_path2 = process_input(input_plate_path)
    result = plate_model(image_batch2)
    for i in range(len(image_batch2)):
        bbox = result[i].boxes.xyxy
        if bbox.nelement()!=0:
            imdis = cv2.imread(image_path2[i])
            file_name = os.path.split(image_path2[i])[1]
            for m in range(len(bbox)):
                bbox_arr = bbox[m].cpu().numpy()
                roi = imdis[int(bbox_arr[1]):int(bbox_arr[3]), int(bbox_arr[0]):int(bbox_arr[2])]
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                magic_color = apply_brightness_contrast(gray, brightness=20, contrast=70)
                ocr_img_name = f'detect_{i}_{file_name}'
                full_img_name = os.path.join(OUT_PATH,ocr_img_name)
                cv2.imwrite(full_img_name,magic_color)
                results = ocr.ocr(full_img_name,verbose=False)
                pred_list=[]
                if len(results[0])!=0:
                    if len(results[0]) > 1:
                        preds=""
                        for k in range(len(results[0])):
                            preds+= " " +str(results[0][k][1][0])
                        pred_list.append(preds)
                    else:
                        preds = str(results[0][0][1][0])
                        pred_list.append(preds)
                    
                    print(preds)
                    if 7>=len(preds)>=5 :
                        serial_code = preds[:-5]
                        main_number = preds[-5:]
                        print(f"Serial code for {file_name}-{m} is",serial_code)
                        print(f"Main number for {file_name}-{m} is",main_number)
                    elif len(preds)>7:
                        print(f"Number captured for {file_name}-{m} is too long")
                    else:
                        print(f"Number for {file_name}-{m} is not complete")
                else:
                    print(f"Cannot read the license plate {i}-{m}")


def predict(input_car_path,car_model,plate_model,output_path,OUT_PATH):
    image_batch,image_path = process_input(input_car_path)
    s_time = time.time()
    result = car_model(image_batch)
    res_time = time.time()
    print(f"Model Inference time for {len(image_batch)} images:",round(res_time-s_time,2),"seconds")
    for i in range(len(image_batch)):
        bbox = result[i].boxes.xyxy
        res_plot = result[i].plot()
        # cv2.imshow(f'Detect-{image_path[i]}',res_plot)
        # cv2.waitKey(0)
        if bbox.nelement() != 0:
            imdis = cv2.imread(image_path[i])
            file_name = os.path.split(image_path[i])[1]
            for m in range(len(bbox)):
                bbox_arr = bbox[m].cpu().numpy()
                roi = imdis[int(bbox_arr[1]):int(bbox_arr[3]), int(bbox_arr[0]):int(bbox_arr[2])]
                # roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                # gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                # magic_color = apply_brightness_contrast(gray, brightness=40, contrast=40)
                img_name = f'car-{m}-{file_name}'
                main_path = os.path.join(output_path,img_name)
                cv2.imwrite(main_path,roi)
                detect_plate_res = plate_model(roi)
                box_lic = detect_plate_res[0].boxes.xyxy
                if box_lic.nelement()!=0:
                    box_lic_arr = box_lic[0].cpu().numpy()
                    lic_roi = roi[int(box_lic_arr[1]):int(box_lic_arr[3]), int(box_lic_arr[0]):int(box_lic_arr[2])]
                    lic_roi_bgr = cv2.cvtColor(lic_roi, cv2.COLOR_RGB2BGR)
                    lic_gray = cv2.cvtColor(lic_roi_bgr, cv2.COLOR_BGR2GRAY)
                    magic_color = apply_brightness_contrast(lic_gray, brightness=40, contrast=70)
                    ocr_img_name = f'detect_{i}_{m}_{file_name}'
                    full_img_name = os.path.join(OUT_PATH,ocr_img_name)
                    cv2.imwrite(full_img_name,magic_color)
                    results = ocr.ocr(full_img_name)
                    pred_list=[]
                    if len(results[0])!=0:
                        if len(results[0]) > 1:
                            preds=""
                            for k in range(len(results[0])):
                                preds+= " " +str(results[0][k][1][0])
                            pred_list.append(preds)
                        else:
                            preds = str(results[0][0][1][0])
                            pred_list.append(preds)
                    
                        print(preds)
                        # if 7>=len(preds)>=5 :
                        #     serial_code = preds[:-5]
                        #     main_number = preds[-5:]
                        #     print(f"Serial code for {file_name}-{m} is",serial_code)
                        #     print(f"Main number for {file_name}-{m} is",main_number)
                        # elif len(preds)>7:
                        #     print(f"Number captured for {file_name}-{m} is {preds[-5:]}")
                        # else:
                        #     print(f"Number for {file_name}-{m} is not complete")
                    else:
                        print(f"Cannot read the license plate {i}-{m}")
                else:
                    print(f"No region detected for {img_name}")

    return len(image_batch)


def main():
    pass
    start_time = time.time()
    # total_images = detect_cars(INPUT_CAR_PATH,car_model,OUTPUT_CAR_PATH)
    # license_detect(OUTPUT_CAR_PATH,plate_model,OUT_PATH)
    total_images = predict(INPUT_CAR_PATH,car_model,plate_model,OUTPUT_CAR_PATH,OUT_PATH)
    end_time = time.time()
    print(f"Total time taken to process {total_images} is {end_time-start_time} seconds")


if __name__ == '__main__':
    main()