import onnxruntime
import cv2
import numpy as np
 
 
# 转换后的onnx权重
onnx_weights_path = 'yolov3-tiny.onnx'
# 指定模型的网络大小
network_size = (416, 416)
 
# 声明onnxruntime会话
session = onnxruntime.InferenceSession(onnx_weights_path)
session.get_modelmeta()
input_name = session.get_inputs()[0].name
output_name_1 = session.get_outputs()[0].name
output_name_2 = session.get_outputs()[1].name
 
# 阅读图片
#img_path = 'test_pic/dog.jpg'
img_path = 'test_pic/giraffe.jpg'
img = cv2.imread(img_path)
image_blob = cv2.dnn.blobFromImage(img, 1 / 255.0, network_size, swapRB=True, crop=False)
 
# 运行推理
layers_result = session.run([output_name_1, output_name_2],
                                         {input_name: image_blob})
layers_result = np.concatenate([layers_result[1], layers_result[0]], axis=1)
 
# 将layers_result转换为bbox，conf和类
def get_final_predictions(outputs, img, threshold, nms_threshold):
    height, width = img.shape[0], img.shape[1]
    boxes, confs, class_ids = [], [], []
    matches = outputs[np.where(np.max(outputs[:, 4:], axis=1) > threshold)]
    for detect in matches:
        scores = detect[4:]
        class_id = np.argmax(scores)
        conf = scores[class_id]
        center_x = int(detect[0] * width)
        center_y = int(detect[1] * height)
        w = int(detect[2] * width)
        h = int(detect[3] * height)
        x = int(center_x - w/2)
        y = int(center_y - h / 2)
        boxes.append([x, y, w, h])
        confs.append(float(conf))
        class_ids.append(class_id)
    
    merge_boxes_ids = cv2.dnn.NMSBoxes(boxes, confs, threshold, nms_threshold)
    
    #将layers_result转换为bbox，conf和类
    boxes = [boxes[int(i)] for i in merge_boxes_ids]
    confs = [confs[int(i)] for i in merge_boxes_ids]
    class_ids = [class_ids[int(i)] for i in merge_boxes_ids]
    return boxes, confs, class_ids
 
boxes, confs, class_ids = get_final_predictions(layers_result, img, 0.3, 0.3)

print(img_path)
print(boxes)
print(confs)
print(class_ids)

label = ["person",
	"bicycle", "car", "motorbike", "aeroplane",
	"bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
	"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
	"pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
	"toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"]

index = 0
for box in boxes:
   #x1 = int((box[0] - box[2] / 2) * img_shape[1])
   #y1 = int((box[1] - box[3] / 2) * img_shape[0])
   #x2 = int((box[0] + box[2] / 2) * img_shape[1])
   #y2 = int((box[1] + box[3] / 2) * img_shape[0])
   x1 = int(box[0])
   y1 = int(box[1])
   x2 = int(box[0] + box[2])
   y2 = int(box[1] + box[3])
   cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
   cv2.putText(img, label[int(class_ids[index])] + ":" + str(round(confs[index], 3)), (x1 + 5, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
   index += 1

cv2.imshow("detect",img)
cv2.waitKey(0)
 
