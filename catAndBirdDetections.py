import cv2
import numpy as np
import urllib.request
import traceback

url = 'http://192.168.2.156/cam-mid.jpg'

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
classNames = []

with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_cat = False
    found_bird = False

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        print("No objects detected")
        return

    for i in indices:
        box = bbox[i]
        x, y, w, h = box
        label = classNames[classIds[i]]

        if label == 'bird':
            found_bird = True
        elif label == 'cat':
            found_cat = True

        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(im, f'{label.upper()} {int(confs[i]*100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(f'Found {label} with confidence {confs[i]*100:.2f}%')

    if found_cat and found_bird:
        print('Alert: Both cat and bird detected!')

while True:
    try:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
        if im is None:
            print("Failed to load image")
            continue

        blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layernames = net.getLayerNames()
        unconnected_out_layers = net.getUnconnectedOutLayers()

        outputNames = [layernames[i - 1] for i in unconnected_out_layers]
        outputs = net.forward(outputNames)
        findObject(outputs, im)
        cv2.imshow('Image', im)
        if cv2.waitKey(1) == ord('q'):
            break
    except Exception as e:
        print('Error:', e)
        traceback.print_exc()
        break

cv2.destroyAllWindows()
