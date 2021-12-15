"""
It is just Pseudocode
"""


import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam


# Hyperparameters
NUM_GPU = None

tensor_train = None
tensor_valid = None
tensor_test = None

epoch = None
batch_size = None
learning_rate = None

type_clf = 'full'

lw_pred = None
lw_recon = None

num_ref = None
dim_time = None
num_heads = None
dim_attn = dim_time // num_heads
dim_hidden_enc = None
dim_hidden_dec = None
dim_ffn = None
dim_latent = None
dim_clf = None


def recon_loss(y_true, y_pred, epoch=None, wait_kl=None):
    """
    Arguements:
        y_true -- shape: (N, (x, m), t, d)
        y_pred -- {outputs_recon, q_z_mean, q_z_logvar}
            outputs_recon -- shape: (N, t, d)
            q_z_mean -- shape: (N, t_ref, d_l)
            q_z_logvar -- shape: (N, t_ref, d_l)
    Returns:
        loss -- shape: (N, )
    """

    def log_normal_pdf(x, mean, logvar, mask):
        const = tf.convert_to_tensor(np.array([2 * np.pi]), dtype=tf.float32)
        const = tf.math.log(const)
        pdf = -0.5 * (const + logvar + (x - mean) ** 2 / tf.math.exp(logvar)) * mask

        return pdf

    def kl_divergence_normal(mu1, logvar1, mu2, logvar2):
        var1 = tf.math.exp(logvar1)
        var2 = tf.math.exp(logvar2)

        kl = 0.5 * (tf.math.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1)

        return kl

    X_true = y_true[:, 0, :, :]
    m = y_true[:, 1, :, :]

    X_pred = y_pred["outputs_recon"]
    q_z_mean = y_pred["q_z_mean"]
    q_z_logvar = y_pred["q_z_logvar"]

    X_pred_std = 0.01 * tf.ones_like(X_pred, dtype=tf.float32)
    X_pred_logvar = 2 * tf.math.log(X_pred_std)

    p_z_mean = tf.zeros_like(q_z_mean, dtype=tf.float32)
    p_z_logvar = tf.zeros_like(q_z_logvar, dtype=tf.float32)

    logpx = tf.reduce_sum(log_normal_pdf(X_true, X_pred, X_pred_logvar, m), axis=(1, 2))
    kl = tf.reduce_sum(kl_divergence_normal(q_z_mean, q_z_logvar, p_z_mean, p_z_logvar), axis=(1, 2))

    logpx = logpx / tf.reduce_sum(m, axis=(1, 2))
    kl = kl / tf.reduce_sum(m, axis=(1, 2))

    if wait_kl is not None:
        if epoch < wait_kl:
            kl_coef = 0
        else:
            kl_coef = (1 - 0.99 ** (epoch - wait_kl))
    else:
        kl_coef = 1

    loss = - (logpx - kl_coef * kl)

    return loss


def train():
    with tf.device('/device:GPU:' + str(NUM_GPU)):
        model = models.mTAND_clf(num_ref, dim_time, dim_attn, num_heads, dim_hidden_enc, dim_ffn, dim_latent, dim_clf,
                                 dim_hidden_dec=dim_hidden_dec, type_clf=type_clf)

        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


    for ep in range(1, epoch + 1):

        # Train
        for step, (X, Y) in enumerate(tensor_train):
            Y_pred = Y["pred"]
            Y_recon = Y["recon"]

            with tf.GradientTape() as tape:
                Y_hat = model(X)
                Y_pred_hat = Y_hat["pred"]
                Y_recon_hat = Y_hat["recon"]

                loss_pred = binary_crossentropy(Y_pred, Y_pred_hat, from_logits=False)
                loss_recon = recon_loss(Y_recon, Y_recon_hat, ep, wait_kl=10)

                loss = lw_pred * loss_pred + lw_recon * loss_recon
                loss_mean = tf.reduce_mean(loss)

            gradients = tape.gradient(loss_mean, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return None
