import tensorflow as tf
import sentencepiece as spm
from train import Transformer, create_masks

# Fallback MAX_LENGTH if not imported
try:
    MAX_LENGTH
except NameError:
    MAX_LENGTH = 100

# --- Load tokenizer ---
sp = spm.SentencePieceProcessor(model_file='Processed_Data/tokenizer.model')

# --- Load model ---
transformer = Transformer(
    num_layers=6,
    d_model=128,
    num_heads=8,
    dff=512,
    input_vocab_size=sp.get_piece_size(),
    target_vocab_size=sp.get_piece_size(),
    pe_input=1000,
    pe_target=1000,
    rate=0.3
)

ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, 'checkpoints', max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print("Model loaded from checkpoint.")
else:
    print("No checkpoint found.")

# --- Encode sentence ---
def encode_sentence(sentence, sp, max_length=MAX_LENGTH):
    tokens = sp.encode(sentence, out_type=int)
    tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=max_length, padding='post')
    return tf.convert_to_tensor(tokens, dtype=tf.int64)

# --- Evaluate ---
def evaluate(sentence, transformer, sp):
    input_tensor = encode_sentence(sentence, sp)
    output = tf.convert_to_tensor([[sp.bos_id()]], dtype=tf.int64)

    for _ in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_tensor, output)

        predictions, _ = transformer(
            input_tensor,
            output,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int64)  # (batch_size, 1)

        if tf.equal(predicted_id[0, 0], sp.eos_id()):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    predicted_tokens = tf.squeeze(output, axis=0).numpy().tolist()

    # Remove BOS, EOS, and padding
    cleaned = [token for token in predicted_tokens if token != sp.bos_id() and token != 0 and token != sp.eos_id()]
    
    # Decode cleaned tokens
    return sp.decode_ids(cleaned)

# --- Test ---
test_sentence = "The internet should be free and open to all users."
output = evaluate(test_sentence, transformer, sp)
print("Input:", test_sentence)
print("Output:", output)