import keras.layers as KL
import keras
import keras.models as KM
from prediction import Attention
from sequence_modeling import BidirectionalLSTM
from feature_extractor import ResNet
from transformation import TPS_SpatialTransFormerNetwork
import tensorflow as tf
from config import Config


class Model(KM.Model):
    def __init__(self, opt):
        super(Model, self).__init__()

        self.Transformation = TPS_SpatialTransFormerNetwork(
            F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
            I_channel_num=opt.input_channel)
        self.FeatureExtraction = ResNet(Config()).model
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = KL.GlobalAveragePooling2D(data_format='channels_last',
                                                         keepdims=True)  # Transform final (imgH/16-1) -> 1
        self.SequenceModeling = keras.Sequential([
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size)])
        self.SequenceModeling_output = opt.hidden_size

        self.Prediction = Attention(None, self.SequenceModeling_output, opt.hidden_size, opt.num_class,
                                    opt.batch_max_length)
        self.opt = opt

    def call(self, input, is_train=True, **kwargs):
        """ Transformation stage """
        input, text = input[0], input[1]
        input = self.Transformation(input)
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature)  # .permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = tf.squeeze(visual_feature, axis=[1])
        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction((contextual_feature, text), is_train)

        return prediction

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self([x, y], training=True)  # Forward pass
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
