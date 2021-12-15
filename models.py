import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, GRU, Bidirectional


class MultiTimeAttention(Layer):
    def __init__(self, dim_time, d_qk, d_v, num_heads, dim_hidden, name="multi_time_attention", **kwargs):
        super(MultiTimeAttention, self).__init__(name=name, **kwargs)

        self.dim_time = dim_time
        self.d_qk = d_qk
        self.d_v = d_v
        self.h = num_heads
        self.dim_hidden = dim_hidden

        self.w_q = Dense(units=self.d_qk * self.h, use_bias=False)
        self.w_k = Dense(units=self.d_qk * self.h, use_bias=False)

        self.w_o = Dense(units=self.dim_hidden, use_bias=False)
        self.w_o.build([None, None, self.h * self.d_v])

    def split_heads(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.h, self.d_qk))

        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Arguments:
            q -- shape: (batch_size, h, t_q, d_qk)
            k -- shape: (batch_size, h, t, d_qk)
            v -- shape: (batch_size, 1, t, d)
            mask -- shape: (batch_size, 1, t, d)
        Returns:
            outputs -- shape: (batch_size, h, t_q, d)
            attention_weight -- shape: (batch_size, h, t_q, t, d)
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scale = matmul_qk / tf.math.sqrt(tf.cast(self.d_qk, tf.float32))
        scale = tf.expand_dims(scale, axis=-1)

        if mask is not None:
            mask = tf.expand_dims(mask, axis=2)
            scale += ((1 - mask) * -1e9)

        attention_weight = tf.nn.softmax(scale, axis=-2)
        v = tf.expand_dims(v, axis=2)
        outputs = attention_weight * v
        outputs = tf.reduce_sum(outputs, axis=-2)

        return outputs, attention_weight

    @tf.function
    def call(self, q, k, v, mask=None):
        """
        Arguments:
            q -- shape: (batch_size, t_q, d_t)
            k -- shape: (batch_size, t, d_t)
            v -- shape: (batch_size, t, d)
            mask -- shape: (batch_size, t, d)
        Returns:
            outputs -- shape: (batch_size, t_q, d_h)
            attention_weight -- shape: (batch_size, h, t_q, t, d)
        """

        batch_size = tf.shape(v)[0]

        q = self.w_q(q)
        k = self.w_k(k)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = tf.expand_dims(v, axis=1)

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)

        attention, attention_weight = self.scaled_dot_product_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, shape=(batch_size, -1, self.h * self.d_v))

        outputs = self.w_o(attention)

        return outputs, attention_weight


class mTAND_enc(Layer):
    def __init__(self, dim_inputs, num_ref, dim_time, dim_attn, num_heads, dim_hidden, dim_ffn, dim_latent,
                 name="mTAND_enc", **kwargs):
        super(mTAND_enc, self).__init__(name=name, **kwargs)

        self.dim_inputs = dim_inputs
        self.num_ref = num_ref
        self.dim_time = dim_time
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.dim_ffn = dim_ffn
        self.dim_latent = dim_latent

        self.time_linear = Dense(units=1)
        self.time_periodic = Dense(units=self.dim_time - 1)

        self.attn = MultiTimeAttention(dim_time=self.dim_time, d_qk=self.dim_attn, d_v=self.dim_inputs,
                                       num_heads=self.num_heads, dim_hidden=self.dim_hidden)
        self.rnn = Bidirectional(GRU(units=self.dim_hidden, recurrent_initializer='zeros', return_sequences=True))
        self.ffn = Sequential([Dense(units=self.dim_ffn, activation='relu'),
                               Dense(units=self.dim_latent * 2)])

    def time_embedding(self, time):
        time = tf.expand_dims(time, axis=-1)
        te_linear = self.time_linear(time)
        te_periodic = tf.math.sin(self.time_periodic(time))
        te = tf.concat([te_linear, te_periodic], axis=-1)

        return te

    @tf.function
    def call(self, inputs):
        """
        Arguments:
            inputs -- {inputs_t, inputs_time}
                inputs_t -- General Descriptors + Time Series
                    shape: (N, (x, m), t, d)
                inputs_time -- Time Stamp
                    shape: (N, t)
        Returns:
            outputs -- ref_time-series latent vectors
                shape: (N, t_ref, d_l * 2)
        """

        inputs_t = inputs["inputs_t"]
        inputs_time = inputs["inputs_time"]

        query = self.time_embedding(tf.expand_dims(tf.linspace(0.0, 1.0, self.num_ref), axis=0))
        key = self.time_embedding(inputs_time)
        value = inputs_t[:, 0, :, :]
        mask = inputs_t[:, 1, :, :]

        outputs_attn, weight_attn = self.attn(query, key, value, mask)
        self.weight_attn = weight_attn

        outputs_rnn = self.rnn(outputs_attn)
        outputs = self.ffn(outputs_rnn)

        return outputs


class mTAND_dec(Layer):
    def __init__(self, dim_hidden, num_ref, dim_time, dim_attn, num_heads, dim_ffn, dim_inputs,
                 name="mTAND_dec", **kwargs):
        super(mTAND_dec, self).__init__(name=name, **kwargs)

        self.dim_hidden = dim_hidden
        self.num_ref = num_ref
        self.dim_time = dim_time
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.dim_inputs = dim_inputs

        self.time_linear = Dense(units=1)
        self.time_periodic = Dense(units=self.dim_time - 1)

        self.rnn = Bidirectional(GRU(units=self.dim_hidden, recurrent_initializer='zeros', return_sequences=True))
        self.attn = MultiTimeAttention(dim_time=self.dim_time, d_qk=self.dim_attn, d_v=self.dim_hidden * 2,
                                       num_heads=self.num_heads, dim_hidden=self.dim_hidden * 2)
        self.ffn = Sequential([Dense(units=self.dim_ffn, activation='relu'),
                               Dense(units=self.dim_inputs)])

    def time_embedding(self, time):
        time = tf.expand_dims(time, axis=-1)
        te_linear = self.time_linear(time)
        te_periodic = tf.math.sin(self.time_periodic(time))
        te = tf.concat([te_linear, te_periodic], axis=-1)

        return te

    @tf.function
    def call(self, inputs):
        """
        Arguments:
            inputs -- {latent, outputs_time}
                latent -- ref_time-series latent vectors
                    shape: (N, t_ref, d_l)
                outputs_time -- Time Stamp for reconstruction
                    shape: (N, t)
        Returns:
            outputs -- reconstruction of inputs
                shape: (N, t, d)
        """

        latent = inputs["latent"]
        outputs_time = inputs["outputs_time"]

        outputs_rnn = self.rnn(latent)

        query = self.time_embedding(outputs_time)
        key = self.time_embedding(tf.expand_dims(tf.linspace(0.0, 1.0, self.num_ref), axis=0))

        outputs_attn, weight_attn = self.attn(query, key, outputs_rnn)
        self.weight_attn = weight_attn

        outputs = self.ffn(outputs_attn)

        return outputs


class mTAND_clf(Model):
    def __init__(self, num_ref, dim_time, dim_attn, num_heads, dim_hidden_enc, dim_ffn, dim_latent,
                 dim_clf, dim_hidden_dec=None, type_clf='enc', name="mTAND_clf", **kwargs):
        super(mTAND_clf, self).__init__(name=name, **kwargs)

        self.num_ref = num_ref
        self.dim_time = dim_time
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.dim_hidden_enc = dim_hidden_enc
        self.dim_hidden_dec = dim_hidden_dec
        self.dim_ffn = dim_ffn
        self.dim_latent = dim_latent
        self.dim_clf = dim_clf
        self.type_clf = type_clf

        self.clf = Sequential([GRU(units=self.dim_hidden_enc, recurrent_initializer='zeros'),
                               Dense(units=self.dim_clf, activation='relu'),
                               Dense(units=self.dim_clf, activation='relu'),
                               Dense(units=1, activation='sigmoid')])

    def build(self, input_shape):
        self.dim_inputs = input_shape["inputs_t"][-1]

        self.enc = mTAND_enc(self.dim_inputs, self.num_ref, self.dim_time, self.dim_attn, self.num_heads,
                             self.dim_hidden_enc, self.dim_ffn, self.dim_latent)

        if not self.type_clf == 'enc':
            self.dec = mTAND_dec(self.dim_hidden_dec, self.num_ref, self.dim_time, self.dim_attn, self.num_heads,
                                 self.dim_ffn, self.dim_inputs)

    @tf.function
    def call(self, inputs):
        """
        Arguments:
            inputs -- {inputs_t, inputs_time}
                inputs_t -- General Descriptors + Time Series
                    shape: (N, (x, m), t, d)
                inputs_time -- Time Stamp
                    shape: (N, t)
        Outputs:
            if mTAND-enc:
                outputs_pred_y
                    shape: (N, 1)
            if mTAND-full:
                pred -- shape: (N, 1)
                recon
                    outputs_pred_x -- shape: (N, t, d)
                    q_z_mean -- shape: (N, t_ref, d_l)
                    q_z_logvar -- shape: (N, t_ref, d_l)
        """

        latent = self.enc(inputs)

        q_z_mean, q_z_logvar = latent[:, :, :self.dim_latent], latent[:, :, self.dim_latent:]
        epsilon = tf.random.normal(tf.shape(q_z_mean), dtype=tf.float32)
        z = q_z_mean + tf.math.exp(0.5 * q_z_logvar) * epsilon

        outputs_pred_y = self.clf(z)

        if self.type_clf == 'enc':
            return outputs_pred_y

        else:
            inputs_dec = {"latent": z, "outputs_time": inputs["inputs_time"]}
            outputs_pred_x = self.dec(inputs_dec)

            outputs_recon = {"outputs_recon": outputs_pred_x, "q_z_mean": q_z_mean, "q_z_logvar": q_z_logvar}
            outputs = {"pred": outputs_pred_y, "recon": outputs_recon}

            return outputs
