# from PIL import Image
import cv2
import torch
# import math 
import function.utils_rotate as utils_rotate
# from IPython.display import display
import os
import datetime
import numpy as np
from flask import Flask, request, jsonify, make_response, send_file, Response
import time
# import argparse
import function.helper as helper

app = Flask(__name__)

# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

@app.route("/detect-lp-video", methods=["POST"])
def detectLpVideo():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 422
    
    video = request.files["video"]
    # nowTime = datetime.datetime.now()
    # hour = str(nowTime.hour)
    # folder_path = "temp/" + hour

    # folder_path = "C:\\Users\\Administrator\\Documents" + "\\" + hour
    # isExist = os.path.exists(folder_path)
    # if not isExist:
    #     # Create a new directory because it does not exist
    #     os.makedirs(folder_path)
    #     print("The new pdf directory is created!")

    file_path = os.path.join(
        "temp",
        str(int(time.time() * 1_000_000)) + "." + "mp4",
    )
    dest = os.path.join(
        "temp",
        str(int(time.time() * 1_000_000)) + "-out." + "mp4",
    )

    video.save(file_path)
    print('running...')

    # vid = cv2.VideoCapture(1)
    vid = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    video_out = cv2.VideoWriter(dest, fourcc, 20.0, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    i = 0
    try:
        while(True):
            i += 1
            path = "out_" + str(i) + ".png"
            ret, frame = vid.read()

            if ret:
                plates = yolo_LP_detect(frame, size=640)
                list_plates = plates.pandas().xyxy[0].values.tolist()
                list_read_plates = set()
                for plate in list_plates:
                    flag = 0
                    x = int(plate[0]) # xmin
                    y = int(plate[1]) # ymin
                    w = int(plate[2] - plate[0]) # xmax - xmin
                    h = int(plate[3] - plate[1]) # ymax - ymin  
                    crop_img = frame[y:y+h, x:x+w]
                    cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
                    cv2.imwrite("crop.jpg", crop_img)
                    # rc_image = cv2.imread("crop.jpg")
                    lp = ""
                    for cc in range(0,2):
                        for ct in range(0,2):
                            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                            if lp != "unknown":
                                list_read_plates.add(lp)
                                cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                video_out.write(frame)
                                flag = 1
                                break
                        if flag == 1:
                            break
            # new_frame_time = time.time()
            # fps = 1/(new_frame_time-prev_frame_time)
            # prev_frame_time = new_frame_time
            # fps = int(fps)
            # cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            # cv2.imshow('frame', frame)
            # video_out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        print('running 2 ...')

        vid.release()
        video_out.release()
        cv2.destroyAllWindows()

        print('running 3 ...')

        # Đọc nội dung video từ tệp (ví dụ: video.mp4)
        with open(dest, 'rb') as video_file:
            video_data = video_file.read()
        
        # Tạo một đối tượng Response với MIME type là video/mp4
        response = Response(video_data, content_type='video/mp4')
        
        return response
    except Exception as e:
        print("Occur a Error" + str(e))

        vid.release()
        video_out.release()
        cv2.destroyAllWindows()
        
        # Đọc nội dung video từ tệp (ví dụ: video.mp4)
        with open(dest, 'rb') as video_file:
            video_data = video_file.read()
        
        # Tạo một đối tượng Response với MIME type là video/mp4
        response = Response(video_data, content_type='video/mp4')
        
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #     else:
    #         print("The file does not exist")

        # if os.path.exists(dest):
        #     os.remove(dest)
        # else:
        #     print("The file does not exist")

        # return {"Server error": "An exception occurred"}, 400
    # finally:
    #     print("remove files")
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #     else:
    #         print("The file does not exist")

        # if os.path.exists(dest):
        #     os.remove(dest)
        # else:
        #     print("The file does not exist")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)