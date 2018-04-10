from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os

model = VGG16(weights='imagenet', include_top=False)
model.summary()

for filename in os.listdir("./media"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        pic_id = filename[6:24]
        print(pic_id)
        img_path = './media/photo_'+pic_id+'.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        # P = imagenet_utils.decode_predictions(vgg16_feature)
        print(vgg16_feature)