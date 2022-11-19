import cv2
import numpy as np

class YOLOdetector():
    def __init__(self):
        self.config='yoloV3/yolov3.cfg'
        self.weights='yoloV3/yolov3.weights'
        self.classes = [['person'], ['bicycle'], ['car'], ['motorcycle'], ['airplane'], ['bus'], ['train'], ['truck'], ['boat'], ['traffic light'], ['fire hydrant'], ['stop sign'], [
            'parking meter'], ['bench'], ['bird'], ['cat'], ['dog'], ['horse'], ['sheep'], ['cow'], ['elephant'], ['bear'], ['zebra'], ['giraffe'], ['backpack'], ['umbrella'], [
            'handbag'], ['tie'], ['suitcase'], ['frisbee'], ['skis'], ['snowboard'], ['sports ball'], ['kite'], ['baseball bat'], ['baseball glove'], ['skateboard'], [
            'surfboard'], ['tennis racket'], ['bottle'], ['wine glass'], ['cup'], ['fork'], ['knife'], ['spoon'], ['bowl'], ['banana'], ['apple'], ['sandwich'], [
            'orange'], ['broccoli'], ['carrot'], ['hot dog'], ['pizza'], ['donut'], ['cake'], ['chair'], ['couch'], ['potted plant'], ['bed'], ['dining table'], [
            'toilet'], ['tv'], ['laptop'], ['mouse'], ['remote'], ['keyboard'], ['cell phone'], ['microwave'], ['oven'], ['toaster'], ['sink'], ['refrigerator'], [
            'book'], ['clock'], ['vase'], ['scissors'], ['teddy bear'], ['hair drier'], ['toothbrush']]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet(self.weights, self.config)


    def get_output_layers(self,net):

        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(self.classes[class_id])

        color = self.COLORS[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def predict(self,image=cv2.imread('yoloV3/dog.jpg')):
        self.image=image
        self.Width = self.image.shape[1]
        self.Height = self.image.shape[0]
        self.scale = 0.00392
        self.blob = cv2.dnn.blobFromImage(self.image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(self.blob)
        self.outs = self.net.forward(self.get_output_layers(self.net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.Width)
                    center_y = int(detection[1] * self.Height)
                    w = int(detection[2] * self.Width)
                    h = int(detection[3] * self.Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    if (class_id==0 or class_id==16):
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])


        self.indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in self.indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(self.image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        return self.image

if __name__ == '__main__':
    yolo= YOLOdetector()
    img=yolo.predict()
    cv2.imshow("object detection", img)
    cv2.waitKey()
