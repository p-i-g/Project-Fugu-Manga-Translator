from mrcnn.config import Config
import mrcnn.model as modellib
import skimage
from mrcnn.model import log
from mrcnn import visualize


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


if __name__ == "__main__":
    model = SegmentationModel(model_dir="model")
    original_image = skimage.io.imread("../datasets/Manga109/Manga109_released_2021_12_30/images/AisazuNihaIrarenai/004.jpg")
    results = model.detect([original_image], verbose=1)
    r = results[0]

    log("original_image", original_image)
    log("class_id", r["class_ids"])
    log("bbox", r["rois"])
    log("masks", r["masks"])
    log("scores", r["scores"])

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                ["BG", "frame", "text"], r['scores'])