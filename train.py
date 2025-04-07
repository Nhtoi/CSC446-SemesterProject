import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import sentencepiece as spm
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# --- Preprocessing & Dataset Preparation ---

PROCESSED_DATA_PATH = "Processed_Data/processed_comments.csv"
TOKENIZER_MODEL_PATH = "Processed_Data/tokenizer.model"
BATCH_SIZE = 32
BUFFER_SIZE = 20000
MAX_LENGTH = 100

def load_tokenizer(model_path=TOKENIZER_MODEL_PATH):
    return spm.SentencePieceProcessor(model_file=model_path)

def pad_sequences(sequences, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=maxlen, padding='post', truncating='post'
    )

def prepare_dataset(file_path, val_split=0.1):
    df = pd.read_csv(file_path)
    input_ids = df['input_ids'].apply(eval).tolist()
    label_ids = df['label_ids'].apply(eval).tolist()

    input_ids = pad_sequences(input_ids, MAX_LENGTH)
    label_ids = pad_sequences(label_ids, MAX_LENGTH)

    X_train, X_val, y_train, y_val = train_test_split(
        input_ids, label_ids, test_size=val_split, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

@tf.function
def val_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(
        inp, tar_inp,
        training=False,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask
    )

    v_loss = loss_function(tar_real, predictions)
    v_acc = accuracy_function(tar_real, predictions)
    return v_loss, v_acc

# --- Transformer Components ---

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32") if mask is not None else None
        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32") if mask is not None else None
        if padding_mask is not None:
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        return tf.tile(mask, [batch_size, 1, 1])

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.token_emb_input = PositionalEmbedding(pe_input, input_vocab_size, d_model)
        self.token_emb_target = PositionalEmbedding(pe_target, target_vocab_size, d_model)

        self.encoder = [TransformerEncoder(d_model, dff, num_heads) for _ in range(num_layers)]
        self.decoder = [TransformerDecoder(d_model, dff, num_heads) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.token_emb_input(inp)
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output)

        dec_output = self.token_emb_target(tar)
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output)

        dec_output = self.dropout(dec_output, training=training)
        final_output = self.final_layer(dec_output)

        return final_output, None

# --- Load Tokenizer and Prepare Data ---
sp = load_tokenizer()
train_dataset, val_dataset = prepare_dataset(PROCESSED_DATA_PATH)

ENCODER_VOCAB = sp.get_piece_size()
DECODER_VOCAB = sp.get_piece_size()

# --- Instantiate Model ---
num_layers = 6
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.3
EPOCHS = 60

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=ENCODER_VOCAB,
    target_vocab_size=DECODER_VOCAB,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    pred_ids = tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype)
    accuracies = tf.equal(real, pred_ids)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# --- Checkpoints ---
checkpoint_path = "checkpoints"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")

# --- Training Step ---
@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask)

        loss = loss_function(tar_real, predictions)

        max_gradient_norm = 5.0
        gradients = tape.gradient(loss, transformer.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -max_gradient_norm, max_gradient_norm) for grad in gradients]
        optimizer.apply_gradients(zip(clipped_gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

# --- Train Model ---
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_state()
    train_accuracy.reset_state()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 100 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    val_losses = []
    val_accuracies = []

    for val_batch, (inp, tar) in enumerate(val_dataset):
        v_loss, v_acc = val_step(inp, tar)
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)

    val_loss = np.mean(val_losses)
    val_accuracy = np.mean(val_accuracies)

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print(f'Validation Loss {val_loss:.4f} Accuracy {val_accuracy:.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
