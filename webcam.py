# from PIL import Image
import cv2
import torch
import imageio.v2 as imageio
from moviepy.editor import VideoFileClip
import base64
# import math 
import function.utils_rotate as utils_rotate
# from IPython.display import display
import os
import datetime
import numpy as np
from flask import Flask, request, jsonify, make_response, send_file, Response
from flask_cors import CORS, cross_origin
import time
# import argparse
import function.helper as helper

app = Flask(__name__)
CORS(app, origins=["https://datn-huy.vercel.app", "http://localhost:3000"])

# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

@app.route("/", methods=["GET"])
def getApp():
    return jsonify("License Plate Detect on Video Server")

@app.route("/detect-lp-video", methods=["POST"])
# @cross_origin(origin='*')
def detectLpVideo():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 422
    
    video = request.files["video"]

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

    vid = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    video_out = cv2.VideoWriter(dest, fourcc, 20.0, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if not vid.isOpened():
        print("Không thể mở video.")
        exit()

    frame_rate = 30  # Tốc độ 3 frame/giây
    frame_count = 0
    list_read_plates = set()

    i = 0
    try:
        while(True):
            i += 1
            path = "out_" + str(i) + ".png"
            ret, frame = vid.read()

            if ret:
                plates = yolo_LP_detect(frame, size=640)
                list_plates = plates.pandas().xyxy[0].values.tolist()
                for plate in list_plates:
                    flag = 0
                    x = int(plate[0]) # xmin
                    y = int(plate[1]) # ymin
                    w = int(plate[2] - plate[0]) # xmax - xmin
                    h = int(plate[3] - plate[1]) # ymax - ymin  
                    crop_img = frame[y:y+h, x:x+w]
                    cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
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
            else:
                print("Đã đọc hết video.")
                break
            
            frame_count += 1
            # Tính thời gian cần chờ giữa các frame
            wait_time = int(1000 / frame_rate)

            # new_frame_time = time.time()
            # fps = 1/(new_frame_time-prev_frame_time)
            # prev_frame_time = new_frame_time
            # fps = int(fps)
            # cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            # cv2.imshow('frame', frame)
            # video_out.write(frame)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        print('running 2 ...')

        if(video_out.isOpened()):
            video_out.release()
        

        vid.release()
        video_out.release()
        cv2.destroyAllWindows()

        print('running 3 ...')

        # Load the MP4 video
        video = VideoFileClip(dest)

        # # Set the output GIF file name and duration (in seconds)
        output_file = os.path.join(
        "temp",
        str(int(time.time() * 1_000_000)) + "-out." + "gif",
        )
        # duration = video.duration

        # Set the start and end times for the GIF (in seconds)
        start_time = 0
        end_time = video.duration  # Set this to the desired duration

        # Convert a subclip of the video to GIF
        video.subclip(start_time, end_time).to_gif(output_file)

        # # Convert the video to GIF
        # video.to_gif(output_file, duration=duration)
        # print(f"Video converted to {output_file}")


        # frames = imageio.imread(dest) 
        # gif_image = pygifsicle.optimize(frames)
        # with open('temp/out.gif', 'wb') as f: 
        #     f.write(gif_image) 

        # cap = cv2.VideoCapture(dest)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # start_time = 20*fps
        # end_time = 25*fps
        # image_lst = []
        # i = 0
        
        # while True:
        #     ret, frame = cap.read()
        #     if ret == False:
        #         break
        #     if (i>=start_time and i<=end_time):
        #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         image_lst.append(frame_rgb)
        
        #         cv2.imshow('a', frame)
        #         key = cv2.waitKey(1)
        #         if key == ord('q'):
        #             break
        #     i +=1
        
        # cap.release()
        # cv2.destroyAllWindows()
        
        # dest_out = 'temp/output.gif'
        # imageio.mimsave(dest_out, image_lst, fps=60)

        # time.sleep(5)

        # Đọc nội dung video từ tệp (ví dụ: video.mp4)
        # with open(dest, 'rb') as video_file:
        #     video_data = video_file.read()
        
        # # Tạo một đối tượng Response với MIME type là video/mp4
        # response = Response(video_data, content_type='video/mp4')

        # Open the video file
        # cap = cv2.VideoCapture(dest)
        
        # # Check if the video file opened successfully
        # if not cap.isOpened():
        #     return "Error: Unable to open video file."

        # def generate_frames():
        #     while True:
        #         success, frame = cap.read()
        #         if not success:
        #             break
        #         ret, buffer = cv2.imencode('.jpg', frame)
        #         if not ret:
        #             break
        #         frame = buffer.tobytes()
        #         yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        # time.sleep(5)

        # Read the GIF file as binary data
        with open(output_file, "rb") as gif_file:
            gif_binary_data = gif_file.read()

        # Encode the binary data as base64
        base64_string = base64.b64encode(gif_binary_data).decode()
        
        # return send_file(output_file, mimetype='image/gif')
        return jsonify({
            'lps': list(list_read_plates),
            'data': base64_string
        })
    except Exception as e:
        print("Occur a Error" + str(e))

        vid.release()
        video_out.release()
        cv2.destroyAllWindows()
        
        time.sleep(5)
        
        # Đọc nội dung video từ tệp (ví dụ: video.mp4)
        with open(dest, 'rb') as video_file:
            video_data = video_file.read()
        
        # Tạo một đối tượng Response với MIME type là video/mp4
        response = Response(video_data, content_type='video/mp4')

        return response

    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #     else:
    #         print("The file does not exist")

        # if os.path.exists(dest):
        #     os.remove(dest)
        # else:
        #     print("The file does not exist")

        # return {"Server error": "An exception occurred"}, 400
    finally:
        print("remove files")
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("The file does not exist")

        if os.path.exists(output_file):
            os.remove(output_file)
        else:
            print("The file does not exist")

        if os.path.exists(dest):
            os.remove(dest)
        else:
            print("The file does not exist")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)