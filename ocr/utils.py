import tensorflow as tf


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image.
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
        """
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        text_ = tf.zeros(batch_max_length + 1, dtype=tf.dtypes.int64)
        text = list(text)
        text.append('[s]')
        mask = [0] * (batch_max_length + 1)
        mask[1:1 + len(text)] = [1] * len(text)
        text = [self.dict[char] for char in text]
        text = [0] + text + [0] * (batch_max_length - len(text))
        mask = tf.convert_to_tensor(mask, dtype=tf.dtypes.int64)
        text = tf.convert_to_tensor(text, dtype=tf.dtypes.int64)
        tmp1 = tf.math.multiply(mask, text)
        tmp2 = tf.math.multiply(1 - mask, text_)
        text_ = tmp1 + tmp2  # batch_text[:, 0] = [GO] token
        return text_

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
