// license plate type classification helper function
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

// detect character and number in license plate
function readPlate(yoloLicensePlate, im) {
    let LPType = "1";
    const results = yoloLicensePlate.predict(im);
    const bbList = results.pandas().xyxy[0].values.tolist();
    
    if (bbList.length === 0 || bbList.length < 7 || bbList.length > 10) {
        return "unknown";
    }

    const centerList = [];
    let yMean = 0;
    let ySum = 0;

    for (const bb of bbList) {
        const xC = (bb[0] + bb[2]) / 2;
        const yC = (bb[1] + bb[3]) / 2;
        ySum += yC;
        centerList.push([xC, yC, bb[bb.length - 1]]);
    }

    // find 2 points to draw line
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
                LPType = "2";
            }
        }
    }

    yMean = parseInt(ySum / bbList.length);
    const size = results.pandas().s;

    // 1 line plates and 2 line plates
    const line1 = [];
    const line2 = [];
    let licensePlate = "";

    if (LPType === "2") {
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

        licensePlate += "-";

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
