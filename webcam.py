# from PIL import Image
import MySQLdb
import cv2
import torch
import subprocess
import imageio.v2 as imageio
from moviepy.editor import VideoFileClip
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64
from flask_sqlalchemy import SQLAlchemy
# import math 
import function.utils_rotate as utils_rotate
# from IPython.display import display
import os
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify, make_response, send_file, Response
from flask_cors import CORS, cross_origin
import time
# import argparse
import function.helper as helper
datetime.utcnow()

app = Flask(__name__)
CORS(app, origins=["https://datn-huy.vercel.app", "http://localhost:3000"])
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Test@123456@localhost/graduation_db'
# db = SQLAlchemy(app)
db = MySQLdb.connect(host="localhost", user="root", passwd="Test@123456", db="graduation_db")
cur = db.cursor()

# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0
# class LicensePlate(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     queue_id = db.Column(db.Integer)
#     lps = db.Column(db.String(255), unique=False, nullable=False)
#     title = db.Column(db.String(255))
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)
#     updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@app.route("/", methods=["GET"])
def getApp():
    return jsonify("License Plate Detect on Video Server")

@app.route("/detect-lp-video", methods=["POST"])
# @cross_origin(origin='*')
def detectLpVideo():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 422
    
    video = request.files["video"]
    queue_id = request.form.get('queue_id')
    if not queue_id:
        return jsonify({'message': 'Thiếu thông tin để thêm bản ghi mới'}), 400


    file_path = os.path.join(
        "temp",
        str(int(time.time() * 1_000_000)) + "." + "mp4",
    )
    dest = os.path.join(
        "temp",
        str(int(time.time() * 1_000_000)) + "-out." + "mp4",
    )
    dest_out = os.path.join(
        "temp",
        str(int(time.time() * 1_000_000)) + "-output." + "mp4",
    )
    video.save(file_path)
    print('running...')

    vid = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(dest, fourcc, 20.0, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if not vid.isOpened():
        print("Không thể mở video.")
        exit()

    frame_rate = 1  # Tốc độ 3 frame/giây
    frame_count = 0
    list_read_plates = set()

    i = 0
    try:
        while(True):
            i += 1
            path = "out_" + str(i) + ".png"

            # Thêm dòng này để bỏ qua các frame không cần thiết
            if i % frame_rate != 0:
                ret, frame = vid.read()
                continue
            
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

            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        print('running 2 ...')

        if(video_out.isOpened()):
            video_out.release()
        

        vid.release()
        video_out.release()
        cv2.destroyAllWindows()

        print('running 3 ...')
        
        # return send_file(output_file, mimetype='video/mp4')
        # return jsonify({
        #     'lps': list(list_read_plates),
        #     'data': base64_string
        # })

        # Xây dựng lệnh FFmpeg
        ffmpeg_command = [
            "ffmpeg",
            "-i", dest,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            "-b:a", "192k",
            "-movflags", "faststart",
            dest_out
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
            print("Chuyển đổi thành công!")
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi chuyển đổi: {e}")

        if os.path.exists(dest_out):
            # data = { 'lps':  list(list_read_plates)}
            new_lps = ', '.join(str(item) for item in list_read_plates)
            # new_license_plate = LicensePlate(
            #     lps=new_lps,
            #     title='video',
            #     queue_id=queue_id
            # )
            print('lps', new_lps)

            try:
                # db.session.add(new_license_plate)
                cur.execute("INSERT INTO license_plates (title, lps, queue_id) VALUES (%s,%s,%s)",('video',new_lps,queue_id))    

                db.commit()
                return send_file(dest_out, mimetype='video/mp4')
            except Exception as e:
                print(f'Lỗi khi thêm bản ghi mới: {str(e)}')
                db.rollback()
                return jsonify({'message': 'Lỗi khi thêm bản ghi mới'}), 500
            finally:
                db.close()
        else:
            return jsonify({"error": "Video file not found"}), 404
        
        
    except Exception as e:
        print("Occur a Error" + str(e))

        vid.release()
        video_out.release()
        cv2.destroyAllWindows()
        
        time.sleep(5)
        
        # Đọc nội dung video từ tệp (ví dụ: video.mp4)
        with open(dest_out, 'rb') as video_file:
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

        if os.path.exists(dest_out):
            os.remove(dest_out)
        else:
            print("The file does not exist")

        if os.path.exists(dest):
            os.remove(dest)
        else:
            print("The file does not exist")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)