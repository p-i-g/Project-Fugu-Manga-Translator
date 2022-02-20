import keras.layers as KL
import keras.models as KM
from keras import activations
import tensorflow as tf


class AttentionCell(KM.Model):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = KL.Dense(hidden_size, use_bias=False)
        self.h2h = KL.Dense(hidden_size)  # either i2i or h2h should have bias
        self.score = KL.Dense(1, use_bias=False)
        self.rnn = KL.LSTMCell(hidden_size)
        self.hidden_size = hidden_size

    def call(self, inputs, training=False):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        prev_hidden, batch_H, char_onehots = inputs
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), 1)
        e = self.score(activations.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = KL.Softmax(1)(e)
        context = tf.linalg.matmul(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha