import keras.layers as KL
import keras.models as KM
from keras import activations
import tensorflow as tf
import numpy as np


class TPS_SpatialTransFormerNetwork(KM.Model):
    """ Rectification Network of RARE, namely TPS based STN """

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            "F": self.F,
            "I_size": self.I_size,
            "I_channel_num": self.I_channel_num
        })
        return cfg

    def __init__(self, F, I_size, I_r_size, I_channel_num=1, *args, **kwargs):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super().__init__(*args, **kwargs)
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def call(self, inputs, training=None, mask=None):
        batch_C_prime = self.LocalizationNetwork(inputs)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = tf.reshape(build_P_prime, [-1, self.I_r_size[0], self.I_r_size[1], 2])
        return bilinear_sampler(inputs, build_P_prime_reshape)


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, coords):
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')

    # ----------------- Changes below -------------------------
    # -> padding_mode = 'border'
    # "#o" means original,  "#t" means modified
    # zero = tf.zeros([], dtype='int32')     #o
    zero = tf.zeros([1], dtype=tf.int32)  # t
    eps = tf.constant([0.5], 'float32')  # t

    # rescale x and y to [0, W-1/H-1]
    x, y = coords[:, ..., 0], coords[:, ..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))
    x = tf.clip_by_value(x, eps, tf.cast(max_x, tf.float32) - eps)  # t
    y = tf.clip_by_value(y, eps, tf.cast(max_y, tf.float32) - eps)  # t
    # -------------- Changes above --------------------

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


class LocalizationNetwork(KM.Model):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            "F": self.F,
            "I_channel_num": self.I_channel_num
        })
        return cfg

    def __init__(self, F, I_channel_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = KM.Sequential([KL.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", use_bias=False),
                                   KL.BatchNormalization(),
                                   KL.ReLU(),
                                   KL.MaxPooling2D(pool_size=(2, 2)),
                                   KL.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                                             use_bias=False),
                                   KL.BatchNormalization(),
                                   KL.ReLU(),
                                   KL.MaxPooling2D(pool_size=(2, 2)),
                                   KL.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                                             use_bias=False),
                                   KL.BatchNormalization(),
                                   KL.ReLU(),
                                   KL.MaxPooling2D(pool_size=(2, 2)),
                                   KL.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same",
                                             use_bias=False),
                                   KL.BatchNormalization(),
                                   KL.ReLU(),
                                   KL.GlobalAvgPool2D()
                                   ], name="conv")  # todo might have to change to channels last
        self.localization_fc1 = KM.Sequential([KL.Dense(256), KL.ReLU()], name="localization_fc1")
        self.localization_fc2 = KL.Dense(self.F * 2, name="localization_fc2")

        self.localization_fc2.build([None, 256])

        weights = self.localization_fc2.get_weights()
        weights[0].fill(0)
        self.localization_fc2.set_weights(weights)

        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.reshape(np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0), -1)
        self.localization_fc2.set_weights([weights[0], initial_bias])

    def call(self, inputs, training=None, mask=None):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = tf.shape(inputs)[0]
        features = tf.reshape(self.conv(inputs), [-1, 512])
        batch_C_prime = self.localization_fc1(features)
        batch_C_prime = self.localization_fc2(batch_C_prime)

        batch_C_prime = tf.reshape(batch_C_prime, (batch_size, self.F, 2))
        return batch_C_prime


class GridGenerator:
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        self.inv_delta_C = tf.expand_dims(tf.convert_to_tensor(self._build_inv_delta_C(self.F, self.C), dtype=tf.dtypes.float32), 0)
        self.P_hat = tf.expand_dims(tf.convert_to_tensor(self._build_P_hat(self.F, self.C, self.P), dtype=tf.dtypes.float32), 0)

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.shape[0]
        print(self.inv_delta_C.shape)
        batch_inv_delta_C = tf.tile(self.inv_delta_C, tf.constant([batch_size, 1, 1]))
        batch_P_hat = tf.tile(self.P_hat, tf.constant([batch_size, 1, 1]))
        print(batch_inv_delta_C.shape)
        batch_C_prime_with_zeros = tf.concat([batch_C_prime, tf.zeros(
            [batch_size, 3, 2])], axis=1)  # batch_size x F+3 x 2
        batch_T = tf.linalg.matmul(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = tf.linalg.matmul(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


if __name__ == "__main__":
    model = TPS_SpatialTransFormerNetwork(16, (128, 128), (128, 128), 3)
    input = KL.Input(shape=[128, 128, 3], batch_size=1)
    model(input)
    print(model.summary())
