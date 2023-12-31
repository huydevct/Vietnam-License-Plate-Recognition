document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('outputCanvas');
  const ctx = canvas.getContext('2d');

  async function loadModels() {
      // Load YOLO models using TensorFlow.js
      // You may need to adjust the paths to your specific model files
      const yoloLPDetect = await tf.loadGraphModel('/media/huy/ubuntu_2/coding/License-Plate-Recognition/model/LP_detector_web_model/model.json');
      const yoloLicensePlate = await tf.loadGraphModel('/media/huy/ubuntu_2/coding/License-Plate-Recognition/model/LP_ocr_web_model/model.json');

      // Set confidence threshold
      const confidenceThreshold = 0.60;

      // Get the webcam feed
      navigator.mediaDevices.getUserMedia({ video: true })
          .then((stream) => {
              video.srcObject = stream;
          })
          .catch((error) => {
              console.error('Error accessing webcam:', error);
          });

      // Perform detection and recognition
      async function detectAndRecognize() {
          const img = tf.browser.fromPixels(video);

          // Detect license plates
          const plates = await yoloLPDetect.predict(img);
          const listPlates = await plates.array();

          // Process detected plates
          const listReadPlates = new Set();
          if (listPlates.length === 0) {
              const lp = await readPlate(yoloLicensePlate, img);
              if (lp !== 'unknown') {
                  // Draw the license plate on the canvas
                  ctx.font = '20px Arial';
                  ctx.fillStyle = 'rgb(36, 255, 12)';
                  ctx.fillText(lp, 7, 70);

                  listReadPlates.add(lp);
              }
          } else {
              for (const plate of listPlates) {
                  const [x, y, w, h] = plate;
                  const cropImg = img.slice([y, x, 0], [h, w, 3]);

                  // Draw the bounding box on the canvas
                  ctx.strokeStyle = 'rgb(0, 0, 225)';
                  ctx.lineWidth = 2;
                  ctx.strokeRect(x, y, w, h);

                  // Perform OCR on the cropped image
                  const lp = await readPlate(yoloLicensePlate, deskew(cropImg, 1, 1));
                  if (lp !== 'unknown') {
                      listReadPlates.add(lp);

                      // Draw the recognized license plate on the canvas
                      ctx.font = '20px Arial';
                      ctx.fillStyle = 'rgb(23, 23, 226)';
                      ctx.fillText(lp, x, y - 10);
                  }
              }
          }

          // Release the webcam frame
          img.dispose();

          // Render the canvas
          requestAnimationFrame(detectAndRecognize);
      }

      // Start the detection and recognition loop
      detectAndRecognize();
  }

  // Helper function: license plate type classification
  function linearEquation(x1, y1, x2, y2) {
      const b = y1 - ((y2 - y1) * x1) / (x2 - x1);
      const a = (y1 - b) / x1;
      return [a, b];
  }

  function checkPointLinear(x, y, x1, y1, x2, y2) {
      const [a, b] = linearEquation(x1, y1, x2, y2);
      const yPred = a * x + b;
      return Math.abs(yPred - y) <= 3;
  }

  async function readPlate(yoloLicensePlate, im) {
      const LPType = '1';
      const results = yoloLicensePlate.predict(im);
      const bbList = results.pandas().xyxy[0].values.tolist();

      if (bbList.length === 0 || bbList.length < 7 || bbList.length > 10) {
          return 'unknown';
      }

      const centerList = [];
      let ySum = 0;
      for (const bb of bbList) {
          const xC = (bb[0] + bb[2]) / 2;
          const yC = (bb[1] + bb[3]) / 2;
          ySum += yC;
          centerList.push([xC, yC, bb[bb.length - 1]]);
      }

      // Find 2 points to draw a line
      let lPoint = centerList[0];
      let rPoint = centerList[0];
      for (const cp of centerList) {
          if (cp[0] < lPoint[0]) {
              lPoint = cp;
          }
          if (cp[0] > rPoint[0]) {
              rPoint = cp;
          }
      }

      for (const ct of centerList) {
          if (lPoint[0] !== rPoint[0]) {
              if (!checkPointLinear(ct[0], ct[1], lPoint[0], lPoint[1], rPoint[0], rPoint[1])) {
                  return '2';
              }
          }
      }

      const yMean = parseInt(ySum / bbList.length);

      // 1 line plates and 2 line plates
      const line1 = [];
      const line2 = [];
      let licensePlate = '';

      if (LPType === '2') {
          for (const c of centerList) {
              if (parseInt(c[1]) > yMean) {
                  line2.push(c);
              } else {
                  line1.push(c);
              }
          }

          for (const l1 of line1.sort((a, b) => a[0] - b[0])) {
              licensePlate += l1[2];
          }

          licensePlate += '-';

          for (const l2 of line2.sort((a, b) => a[0] - b[0])) {
              licensePlate += l2[2];
          }
      } else {
          for (const l of centerList.sort((a, b) => a[0] - b[0])) {
              licensePlate += l[2];
          }
      }

      return licensePlate;
  }

  function changeContrast(img) {
      const lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
      const [lChannel, a, b] = cv2.split(lab);
      const clahe = cv2.createCLAHE(3.0, [8, 8]);
      const cl = clahe.apply(lChannel);
      const limg = cv2.merge([cl, a, b]);
      return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR);
  }

  function rotateImage(image, angle) {
      const imageCenter = [image.shape[1] / 2, image.shape[0] / 2];
      const rotMat = cv2.getRotationMatrix2D(imageCenter, angle, 1.0);
      return cv2.warpAffine(image, rotMat, [image.shape[1], image.shape[0]], cv2.INTER_LINEAR);
  }

  function computeSkew(srcImg, centerThres) {
      const [h, w] = srcImg.shape;
      const img = cv2.medianBlur(srcImg, 3);
      const edges = cv2.Canny(img, 30, 100, 3, true);
      const lines = cv2.HoughLinesP(edges, 1, Math.PI / 180, 30, w / 1.5, h / 3.0);

      if (lines === null) {
          return 1;
      }

      let minLine = 100;
      let minLinePos = 0;

      for (let i = 0; i < lines.size; i++) {
          for (const [x1, y1, x2, y2] of lines[i]) {
              const centerPoint = [(x1 + x2) / 2, (y1 + y2) / 2];

              if (centerThres === 1 && centerPoint[1] < 7) {
                  continue;
              }

              if (centerPoint[1] < minLine) {
                  minLine = centerPoint[1];
                  minLinePos = i;
              }
          }
      }

      let angle = 0.0;
      let cnt = 0;

      for (const [x1, y1, x2, y2] of lines[minLinePos]) {
          const ang = Math.atan2(y2 - y1, x2 - x1);

          if (Math.abs(ang) <= 30) {
              angle += ang;
              cnt++;
          }
      }

      if (cnt === 0) {
          return 0.0;
      }

      return (angle / cnt) * 180 / Math.PI;
  }

  function deskew(srcImg, changeCons, centerThres) {
      if (changeCons === 1) {
          return rotateImage(srcImg, computeSkew(changeContrast(srcImg), centerThres));
      } else {
          return rotateImage(srcImg, computeSkew(srcImg, centerThres));
      }
  }

  // Start loading models when the DOM is ready
  loadModels();
});
