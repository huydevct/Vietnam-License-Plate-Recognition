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
