import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from optparse import OptionParser
from keras import applications
from keras.engine import Input, Model
from keras.layers import Flatten
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from scipy import spatial


def analyze_duplicate_images(imgs_path, weights_path):
    # use ResNet50 model extract feature from fc1 layer
    base_model = applications.ResNet50(weights='imagenet', pooling=max, include_top=False)
    input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(input)
    x = Flatten()(x)
    model = Model(inputs=input, outputs=x)
    # model.load_weights(weights_path)

    imgs_features = []
    dir_files = [os.path.join(imgs_path, p) for p in os.listdir(imgs_path)]
    for img_path in dir_files:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_reduce = features.squeeze()
        imgs_features.append(features_reduce)

    tree = spatial.KDTree(imgs_features)

    for inx, feature in enumerate(imgs_features):
        nearest = tree.query(feature, k=4)
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.imread(dir_files[inx]))
        plt.title(os.path.basename(dir_files[inx]))
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.imread(dir_files[nearest[1][1]]))
        plt.title(os.path.basename(dir_files[nearest[1][1]]))
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.imread(dir_files[nearest[1][2]]))
        plt.title(os.path.basename(dir_files[nearest[1][2]]))
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.imread(dir_files[nearest[1][3]]))
        plt.title(os.path.basename(dir_files[nearest[1][3]]))
        plt.show()


def rename_img(path):
    dir_files = os.listdir(path)
    for i, f in enumerate(dir_files):
        new_file = "coke-bottle-%s.%s" % (i, f.split(".")[1])
        os.rename(os.path.join(path ,f), os.path.join(path, new_file))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", help="Please specify action to do: rename, analyze")
    parser.add_option("-p", "--dir-path", dest="dir_path", help="Please specify dir images path.")
    parser.add_option("-w", "--weights-path", dest="weights_path", help="Please specify weights path.")

    (options, args) = parser.parse_args()
    if options.action is None or options.action not in ['rename', 'analyze']:
        raise Exception('Correct argument should be specify.')

    if options.dir_path is None:
        raise Exception('dir-path arg should be specify.')

    if options.action in 'rename':
        rename_img(options.dir_path)

    if options.action in 'analyze':
        analyze_duplicate_images(options.dir_path, options.weights_path)