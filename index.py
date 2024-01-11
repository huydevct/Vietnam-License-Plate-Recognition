from flask import Flask, request, jsonify, make_response, send_file
# from PIL import Image
# import json
import cv2
import torch
# import math
import function.utils_rotate as utils_rotate
# from IPython.display import display
import os
import time
# import argparse
import function.helper as helper

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to input image')
# args = ap.parse_args()

def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj

app = Flask(__name__)

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')

@app.route("/detect-lp", methods=["POST"])
def detectLp():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}),422
    
    print("Recieved files...")

    image = request.files["image"]
    file_path = os.path.join('temp', str(int(time.time() * 1_000_000)) + "." + "jpg")
    print("file_path: ",file_path)
    out_path = str(int(time.time() * 1_000_000)) + "-out." + "jpg"
    file_path_ouput = "/var/www/html/graduation/temp/detect-lp/" + out_path
    crop_path = os.path.join("temp", "crop.jpg")

    image.save(file_path)

    yolo_license_plate.conf = 0.60

    print("Start processing...")

    try:
        print("file_path: ","/media/huy/ubuntu_2/coding/License-Plate-Recognition/"+file_path)
        img = cv2.imread(file_path)
        plates = yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        if len(list_plates) == 0:
            lp = helper.read_plate(yolo_license_plate,img)
            if lp != "unknown":
                cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                list_read_plates.add(lp)
        else:
            print("Processing...")

            for plate in list_plates:
                flag = 0
                x = int(plate[0]) # xmin
                y = int(plate[1]) # ymin
                w = int(plate[2] - plate[0]) # xmax - xmin
                h = int(plate[3] - plate[1]) # ymax - ymin
                crop_img = img[y:y+h, x:x+w]
                cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
                cv2.imwrite(crop_path, crop_img)
                rc_image = cv2.imread(crop_path)
                lp = ""
                for cc in range(0,2):
                    for ct in range(0,2):
                        lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (23,23,226), 2)
                            flag = 1
                            break
                    if flag == 1:
                        break

        cv2.imwrite(file_path_ouput, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # return send_file(file_path_ouput, download_name="output.jpg")
        lps = serialize_sets(list_read_plates)
        response = {
            "liscense_plates": lps,
            "file_path_out": out_path
        }
        print("Done...")

        return jsonify(response)
    except():
        print("Occur a Error")
        torch.cuda.empty_cache()
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("The file does not exist")

        if os.path.exists(file_path_ouput):
            os.remove(file_path_ouput)
        else:
            print("The file does not exist")

        if os.path.exists(crop_path):
            os.remove(crop_path)
        else:
            print("The file does not exist")
        return {"Server error": "An exception occurred"}, 400
    finally:
        print("remove disk")
        torch.cuda.empty_cache()

        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("The file does not exist")

        if os.path.exists(crop_path):
            os.remove(crop_path)
        else:
            print("The file does not exist")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
