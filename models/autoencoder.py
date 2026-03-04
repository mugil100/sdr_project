"""
Base Autoencoder (Original working version).
Encoder: k bits -> n complex symbols (I/Q)
Decoder: n complex symbols -> k bits
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AWGNChannel(layers.Layer):
    """
    AWGN channel used ONLY during training.
    """
    def __init__(self, snr_db=10.0, **kwargs):
        super().__init__(**kwargs)
        self.snr_db = snr_db

    def call(self, inputs, training=None):
        if training:
            snr_linear = 10.0 ** (self.snr_db / 10.0)
            noise_var = 1.0 / snr_linear
            std = tf.cast(tf.sqrt(noise_var / 2.0), tf.float32)
            noise = tf.complex(
                tf.random.normal(tf.shape(inputs), stddev=std),
                tf.random.normal(tf.shape(inputs), stddev=std)
            )
            return inputs + noise
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"snr_db": self.snr_db})
        return cfg

class PhaseOffsetChannel(layers.Layer):
    """
    Applies a random phase rotation [0, 2π) to the complex symbols.
    Used ONLY during training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        if training:
            # batch size shape
            batch_size = tf.shape(inputs)[0]
            
            # Generate one random phase per batch element
            phase_rad = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=2.0 * np.pi)
            
            # Complex exponential: e^(j * phase)
            rotation = tf.complex(tf.cos(phase_rad), tf.sin(phase_rad))
            
            return inputs * rotation
        return inputs
        
    def get_config(self):
        return super().get_config()

class FrequencyOffsetChannel(layers.Layer):
    """
    Applies a time-varying phase rotation (CFO) to complex symbols.
    y[t] = x[t] * exp(j * 2π * f * t)
    Used ONLY during training.
    """
    def __init__(self, max_freq_offset=0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_freq_offset = max_freq_offset

    def call(self, inputs, training=None):
        if training:
            batch_size = tf.shape(inputs)[0]
            num_symbols = tf.shape(inputs)[1]
            
            # Generate random normalized frequency offset per batch element
            # Range: [-max_freq_offset, max_freq_offset]
            f = tf.random.uniform(
                shape=[batch_size, 1], 
                minval=-self.max_freq_offset, 
                maxval=self.max_freq_offset
            )
            
            # Time indices: [0, 1, 2, ..., n-1] -> shape [1, n]
            t = tf.cast(tf.range(num_symbols), tf.float32)
            t = tf.reshape(t, [1, num_symbols])
            
            # 2 * pi * f * t
            phase_rad = 2.0 * np.pi * f * t
            
            # Complex exponential rotation
            rotation = tf.complex(tf.cos(phase_rad), tf.sin(phase_rad))
            
            return inputs * rotation
        return inputs
        
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_freq_offset": self.max_freq_offset})
        return cfg

@tf.keras.utils.register_keras_serializable(name='PowerNormalization')
class PowerNormalization(layers.Layer):
    """Normalizes power of symbols to 1.0"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        avg_power = tf.reduce_mean(tf.square(inputs))
        return inputs / tf.sqrt(avg_power + 1e-8)
        
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(name='ToComplex')
class ToComplex(layers.Layer):
    """Converts 2D real features into 1D complex symbols"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.complex(inputs[:, :, 0], inputs[:, :, 1])
        
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(name='FromComplex')
class FromComplex(layers.Layer):
    """Converts 1D complex symbols back to 2D real features"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.stack([tf.math.real(inputs), tf.math.imag(inputs)], axis=-1)
        
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(name='RayleighFadingChannel')
class RayleighFadingChannel(layers.Layer):
    """
    Rayleigh Flat Fading channel.
    Generates a single complex fading coefficient h ~ CN(0,1) per batch element
    and multiplies all symbols in that element by h.
    Only applied during training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        if training:
            # batch size shape
            batch_size = tf.shape(inputs)[0]
            
            # Generate single scalar h per batch element: (batch_size, 1)
            # h = (randn() + j*randn()) / sqrt(2)
            h_real = tf.random.normal([batch_size, 1]) / tf.sqrt(2.0)
            h_imag = tf.random.normal([batch_size, 1]) / tf.sqrt(2.0)
            h = tf.complex(h_real, h_imag)
            
            # Broadcast scalar h over all symbols
            return inputs * h
        return inputs
        
    def get_config(self):
        return super().get_config()

CUSTOM_OBJECTS = {
    'AWGNChannel': AWGNChannel,
    'PhaseOffsetChannel': PhaseOffsetChannel,
    'FrequencyOffsetChannel': FrequencyOffsetChannel,
    'RayleighFadingChannel': RayleighFadingChannel,
    'PowerNormalization': PowerNormalization,
    'ToComplex': ToComplex,
    'FromComplex': FromComplex
}

def create_autoencoder(k=4, n=4, snr_db=10.0, use_phase_offset=False, use_cfo=False, max_cfo=0.05, use_fading=False):
    width = 128
    
    # ── ENCODER ──
    enc_inp = layers.Input(shape=(k,), name='encoder_input')
    x = layers.Dense(width, activation='relu')(enc_inp)
    x = layers.Dense(width // 2, activation='relu')(x)
    x = layers.Dense(2 * n, activation='linear')(x)
    x = layers.Reshape((n, 2))(x)
    
    # Power Normalization inside graph (no custom layers)
    x = PowerNormalization(name='power_norm')(x)
    
    # Complex Conversion
    enc_out = ToComplex(name='to_complex')(x)
    
    encoder = keras.Model(enc_inp, enc_out, name='encoder')

    # ── DECODER ──
    dec_inp = layers.Input(shape=(n,), dtype=tf.complex64, name='decoder_input')
    
    # From Complex Conversion
    y = FromComplex(name='from_complex')(dec_inp)
    y = layers.Reshape((2 * n,))(y)
    
    y = layers.Dense(width // 2, activation='relu')(y)
    y = layers.Dense(width, activation='relu')(y)
    dec_out = layers.Dense(k, activation='sigmoid', name='dec_out')(y)
    
    decoder = keras.Model(dec_inp, dec_out, name='decoder')

    # ── AUTOENCODER ──
    ch_out = encoder.output
    
    if use_fading:
        ch_out = RayleighFadingChannel()(ch_out)
        
    if use_cfo:
        ch_out = FrequencyOffsetChannel(max_freq_offset=max_cfo)(ch_out)
    if use_phase_offset:
        ch_out = PhaseOffsetChannel()(ch_out)
        
    ch_out = AWGNChannel(snr_db=snr_db)(ch_out)
    ae_out = decoder(ch_out)
    autoencoder = keras.Model(encoder.input, ae_out, name='autoencoder')
    
    return autoencoder, encoder, decoder

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
