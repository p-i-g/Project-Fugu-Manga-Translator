import keras.layers as KL
import keras
import keras.models as KM
from prediction import Attention
from sequence_modeling import BidirectionalLSTM
from feature_extractor import ResNet
from transformation import TPS_SpatialTransFormerNetwork
import tensorflow as tf

class Model(KM.Model):
    def __init__(self, opt):
        super(Model, self).__init__()

        self.Transformation = TPS_SpatialTransFormerNetwork(
            F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
            I_channel_num=opt.input_channel)
        self.FeatureExtraction = ResNet((opt.input_channel, opt.output_channel)).build()
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = KL.GlobalAvgPool2D(data_format='channels_first')  # Transform final (imgH/16-1) -> 1

        self.SequenceModeling = keras.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class, opt.num_class, opt.batch_max_length)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature)#.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = tf.squeeze(visual_feature)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature, text, is_train,
                                     batch_max_length=self.opt.batch_max_length)

        return prediction