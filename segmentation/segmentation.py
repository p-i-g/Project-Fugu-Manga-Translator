from skimage import io

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config


class InferenceConfig(Config):
    """
    Configuration for segmentation of manga pages
    """
    # Give the configuration a recognizable name
    NAME = "manga"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 classes (frame + text)

    STEPS_PER_EPOCH = 1000

    TRAIN_ROIS_PER_IMAGE = 128

    IMAGES_PER_GPU = 1

    GPU_COUNT = 1

    DETECTION_MIN_CONFIDENCE = 0.7


class SegmentationModel:
    def __init__(self, model_dir, weights_path=None):
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=self.config)

        if weights_path is not None:
            model_path = weights_path
        else:
            model_path = self.model.find_last()

        print("Loading weights from:" + model_path)
        self.model.load_weights(model_path, by_name=True)

    def detect(self, images, verbose=0):
        return self.model.detect(images, verbose=verbose)

    def showDetections(self, images, verbose=0):
        results = self.detect(images, verbose=verbose)
        for i, image in enumerate(images):
            r = results[i]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        ["BG", "frame", "text"], r['scores'])


def main():
    model = SegmentationModel(model_dir="model")
    original_image = io.imread(
        "../datasets/Manga109/Manga109_released_2021_12_30/images/ARMS/005.jpg")
    model.showDetections([original_image], verbose=1)


if __name__ == "__main__":
    main()

# TODO optimise model, evaluate model
