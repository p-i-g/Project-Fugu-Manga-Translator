import keras.layers as KL
import keras.models as KM
from keras import activations
import tensorflow as tf
import numpy as np


class BidirectionalLSTM(KM.Model):

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            "hidden_size": self.hidden_size,
            "output_size": self.output_size
        })

    def __init__(self, hidden_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = KL.Bidirectional(KL.LSTM(hidden_size, return_sequences=True))
        self.linear = KL.Dense(output_size)

    def call(self, inputs, training=None, mask=None):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent = self.rnn(inputs)
        output = self.linear(recurrent)
        return output


if __name__ == "__main__":
    model = BidirectionalLSTM(16, 16)
    input = KL.Input(shape=[1, 16])
    model.build(input_shape=[1, 16])
    print(model.summary())
