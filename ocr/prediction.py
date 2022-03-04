import keras.layers as KL
import keras.models as KM
from keras import activations
import tensorflow as tf


class Attention(KM.Model):
    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'batch_max_length': self.batch_max_length
        })
        return cfg

    def __init__(self, config, input_size, hidden_size, num_classes, batch_max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.batch_max_length = batch_max_length
        self.num_classes = num_classes
        self.attention_cell = AttentionCell(config, input_size, hidden_size, num_classes, name="attention_cell")
        self.generator = KL.Dense(self.num_classes)
        # self.model = self.build()

    # def build(self):
    #     batch_H_input = KL.Input(shape=[self.input_size], batch_size=1)
    #     text_input = KL.Input(shape=[100], batch_size=1, dtype=tf.dtypes.int32)
    #     cur_hidden, alpha = self.graph(batch_H_input, text_input)
    #     return KM.Model(inputs=[batch_H_input, text_input], output=[cur_hidden, alpha], name="attention")

    def _char_to_onehot(self, input_char, onehot_dim=38):
        # input_char = tf.expand_dims(input_char)
        one_hot = tf.one_hot(indices=input_char, depth=onehot_dim)
        # print(one_hot.shape)
        return one_hot

    def call(self, inputs, is_train=True):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_H = inputs[0]
        text = inputs[1]
        batch_size = tf.shape(batch_H)[0]
        num_steps = self.batch_max_length + 1  # +1 for [s] at end of sentence.
        output_hiddens_list = []

        hidden = [tf.zeros([batch_size, self.hidden_size]), tf.zeros([batch_size, self.hidden_size])]

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                # print(hidden.shape, batch_H.shape, char_onehots.shape)
                hidden, alpha = self.attention_cell([hidden, batch_H, char_onehots])
                output_hiddens_list.append(hidden[0])  # LSTM hidden index (0: hidden, 1: Cell)
            output_hiddens = tf.stack(output_hiddens_list)
            probs = self.generator(output_hiddens)

        else:
            targets = tf.zeros(batch_size, dtype=tf.dtypes.int64)  # [GO] token
            probs = tf.zeros([batch_size, num_steps, self.num_classes])

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell([hidden, batch_H, char_onehots])
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(KM.Model):
    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'input_size': self.input_size
        })
        return cfg

    def __init__(self, config, input_size, hidden_size, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.i2h = KL.Dense(self.hidden_size, use_bias=False, name="i2h")
        self.h2h = KL.Dense(self.hidden_size, name="h2h")
        self.score = KL.Dense(1, use_bias=False, name="score")
        self.lstm = KL.LSTMCell(self.hidden_size, name="lstm")

    # def build(self):
    #     prev_hidden_input = KL.Input(shape=[2, self.hidden_size])
    #     batch_H_input = KL.Input(shape=[self.input_size])
    #     char_onehots_input = KL.Input(shape=[self.num_classes])
    #
    #     output = self.graph(prev_hidden_input, batch_H_input, char_onehots_input)
    #
    #     return KM.Model(inputs=[prev_hidden_input, batch_H_input, char_onehots_input], outputs=output,
    #                     name='attention_cell')

    # def call(self, inputs, training=None, mask=None):
    #     # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
    #     print(inputs.shape)
    #     prev_hidden = inputs[0]
    #     batch_H = inputs[1]
    #     char_onehots = inputs[2]
    #     batch_H_proj = self.i2h(batch_H)
    #     prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), 1)
    #     e = self.score(activations.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1
    #
    #     alpha = KL.Softmax(1)(e)
    #     alpha = KL.Permute((2, 1))(alpha)
    #     context = tf.squeeze(tf.linalg.matmul(alpha, batch_H), 1) # batch_size x num_channel
    #     print(context.shape, char_onehots.shape)
    #     concat_context = tf.concat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
    #     cur_hidden = self.rnn(concat_context, prev_hidden)
    #     return cur_hidden, alpha

    def call(self, inputs):  # prev_hidden, batch_H, char_onehots
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        prev_hidden = inputs[0]
        batch_H = inputs[1]
        char_onehots = inputs[2]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0])
        prev_hidden_proj = tf.expand_dims(prev_hidden_proj, 1)
        e = self.score(
            activations.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = KL.Softmax(1)(e)
        # print(prev_hidden_proj.shape, alpha.shape)
        alpha = KL.Permute((2, 1))(alpha)
        context = tf.squeeze(tf.linalg.matmul(alpha, batch_H), 1)  # batch_size x num_channel
        concat_context = tf.concat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        # print(concat_context.shape, prev_hidden.shape)
        cur_hidden = self.lstm(concat_context, prev_hidden)[1]
        return cur_hidden, alpha


if __name__ == "__main__":
    batch_H_input = KL.Input(shape=[16], batch_size=1)
    text_input = KL.Input(shape=[100], batch_size=1, dtype=tf.dtypes.int32)
    model = Attention(None, 16, 16, 3, 25)
    model([batch_H_input, text_input])
    print(model.summary())
