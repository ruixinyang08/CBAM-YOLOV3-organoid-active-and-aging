from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image
from PIL import Image
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
yolo = YOLO()

from keras.layers import Input




fi = open('img/test-cut/test.txt')
txt = fi.readlines()
im_names = []
for line in txt:
    line = line.strip('\n')
    line = ('img/test-cut/predict/' + line + '.tif')
    im_names.append(line)
print(im_names)
fi.close()
a = 0
for im_name in im_names:
    a = a + 1
    image = Image.open(im_name)
    image = image.convert('RGB')
    r_image = yolo.detect_image(image)
    r_image.save("img/test-cut/img_out/"+im_name[67:78]+'_'+str(a)+'.jpg')
yolo.close_session()
