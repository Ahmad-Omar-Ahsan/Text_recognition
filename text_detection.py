from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import pytesseract
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, help='path to input image')
ap.add_argument('-e', '--east', type=str, help='path to input EAST text detector')
ap.add_argument('-c', '--min-confidence', type=float, default=0.5,
                help='minimum probability required to inspect a region')
ap.add_argument('-w', '--width', type=int, default=320, help='resized image width (should be a multiple of 32)')
ap.add_argument('-t', '--height', type=int, default=320, help='resized image height (should be a multiple of 32)')
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


def decode_predictions(scores, geometry):
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data_0 = geometry[0, 0, y]
        x_data_1 = geometry[0, 1, y]
        x_data_2 = geometry[0, 2, y]
        x_data_3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < args['min_confidence']:
                continue

            (offset_x, offset_y) = (x*4.0, y*4.0)
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data_0[x] + x_data_2[x]
            w = x_data_1[x] + x_data_3[x]

            end_x = int(offset_x + (cos * x_data_1[x]) + (sin * x_data_2[x]))
            end_y = int(offset_y - (sin * x_data_1[x]) + (cos * x_data_2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rects, confidences


image = cv2.imread(args['image'])
orig = image.copy()
(orig_h, orig_w) = image.shape[:2]


(new_w, new_h) = (args['width'], args['height'])
r_w = orig_w / float(new_w)
r_h = orig_h / float(new_h)

image = cv2.resize(image, (new_w, new_h))
(H, W) = image.shape[:2]

layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    'feature_fusion/concat_3']

print("[INFO] loading EAST text detector")
net = cv2.dnn.readNet(args['east'])

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layer_names)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)


results = []

for (x1, y1, x2, y2) in boxes:
    x1 = int(x1 * r_w)
    y1 = int(y1 * r_h)
    x2 = int(x2 * r_w)
    y2 = int(y2 * r_h)

    dx = int((x2 - x1) * args['padding'])
    dy = int((y2 - y1) * args['padding'])

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(orig_w, x2 + (dx * 2))
    y2 = min(orig_h, y2 + (dy * 2))

    roi = orig[y1:y2, x1:x2]

    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)

    results.append(((x1, y1, x2, y2), text))

results = sorted(results, key=lambda r: (r[0][0], r[0][1]))

for ((x1, y1, x2, y2), text) in results:
    print("OCR text")
    print("========")
    print("{}\n".format(text))
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255, 2))
    cv2.putText(output, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2
                  , (0, 0, 255), 3)
    cv2.imshow("text detection", output)
    cv2.waitKey(0)