from tensorflow.keras import layers
import tensorflow as tf
from segmentation.mrcnn import resize_image
from segmentation import segmentation as seg

seg_model = seg.SegmentationModel("../segmentation/model")


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        for i in range(3):
            self.encoder.add(layers.Conv2D(32, (3, 3), padding="same"))
            self.encoder.add(layers.LeakyReLU(0.1))
            self.encoder.add(layers.MaxPooling2D((2, 2), padding="same"))
            self.decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same"))
            self.decoder.add(layers.LeakyReLU(0.1))
        self.decoder.add(layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same"))

    def call(self, _input):
        encoded = self.encoder(_input)
        decoded = self.decoder(encoded)
        return decoded


def resize_images(images, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    for i in range(len(images)):
        images[i] = resize_image(images[i], min_dim, max_dim, min_scale, mode)[0]
    return images


def strip_text(images):
    results = seg_model.detect(images)
    for i in range(len(results)):
        r = results[i]
        masks = r["masks"][r["class_ids"] == 2]
        for mask in masks:
            images[i][mask] = 255
    return images


class AutoencoderModel(object):
    def __init__(self, model_dir):
        self.model = tf.keras.models.load_model("model/autoencoder", custom_objects={"Autoencoder": Autoencoder})

    def encode(self, images):
        images = strip_text(images)
        images = resize_images(images, max_dim=512)
        images = tf.convert_to_tensor(images)
        return self.model.encoder.predict(images)

    def decode(self, encoded):
        images = tf.convert_to_tensor(encoded)
        return self.model.decoder.predict(images)
