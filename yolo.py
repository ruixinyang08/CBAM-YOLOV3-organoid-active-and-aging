import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from nets.yolo3 import yolo_body,yolo_eval
from utils.utils import letterbox_image

class YOLO(object):
    _defaults = {
        "model_path"        : 'logs/ep247-loss20.146-val_loss22.507.h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/new_classes.txt',
        "score"             : 0.5,
        "iou"               : 0.3,
        "max_boxes"         : 100,
        "model_image_size"  : (416, 416)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2, ))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape, max_boxes = self.max_boxes,
                score_threshold = self.score, iou_threshold = self.iou)
        return boxes, scores, classes


    def detect_image(self, image):
        start = timer()

        new_image_size = (self.model_image_size[1],self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.


        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        sumall = sum(out_classes)

        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic=[]
        number = len(out_boxes)
        number = str(number)
        print(number)


        height = []
        width = []
        diameter = []
        volume = []
        confidence = []

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            heigh = (bottom - top)*0.883
            height.append(heigh)
            height
            # print('height',height)


            wid = (right - left)*0.883
            width.append(wid)
            width
            # print('width',width)

            Diamet = (heigh + wid)/2
            diameter.append(Diamet)
            # print('diameter',diameter)

            type(Diamet)
            volum = (4/3)*3.14*((heigh + wid)/4)*((heigh + wid)/4)
            volume.append(volum)


            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            #print(label)


            height_mean = np.mean(height)
            height_mean = str(height_mean)

            height_median = np.median(height)
            height_median = str(height_median)


            width_mean = np.mean(width)
            width_mean = str(width_mean)

            width_median = np.median(width)
            width_median = str(width_median)


            diameter_mean = np.mean(diameter)
            diameter_mean = str(diameter_mean)

            diameter_median = np.median(diameter)
            diameter_median = str(diameter_mean)


            volume_mean = np.mean(volume)
            volume_mean = str(volume_mean)


            confidence_mean = np.mean(confidence)
            confidence_mean = str(confidence_mean)

            # print('height_mean', height_mean)
            # print('height_median', height_median)
            # print('width_mean',width_mean)
            # print('width_median', width_median)

            # print('box_number',number)

            # print('diameter_mean',diameter_mean)
            # print('diameter_median',diameter_median)
            # print('volume_mean',volume_mean)
            # print('confidence_mean',confidence_mean)

            #print(number)


            #print(number)




            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        #print(end - start)
        #print(number)
        #print(diameter_mean)
        return image

    def close_session(self):
        self.sess.close()
