import pandas as pd
import re
import os
import csv
import sentencepiece as spm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove non-alphabetic characters (preserve spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    return text.strip()

def train_sentencepiece_tokenizer(input_file, model_prefix='tokenizer', vocab_size=1146.):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

def load_tokenizer(model_path='tokenizer.model'):
    return spm.SentencePieceProcessor(model_file=model_path)

def tokenize_with_special_tokens(text, sp):
    return [sp.bos_id()] + sp.encode(text, out_type=int) + [sp.eos_id()]

def main():
    # Paths
    data_folder = 'Raw_Data'
    file_name = 'comments_detailed_with_topic_summary.csv'
    file_path = os.path.join(data_folder, file_name)

    processed_data_folder = 'Processed_Data'
    processed_file_path = os.path.join(processed_data_folder, 'processed_comments.csv')
    os.makedirs(processed_data_folder, exist_ok=True)

    # Load and clean data
    data = pd.read_csv(file_path)

    # Drop rows with missing or empty summaries
    data = data[data['summary'].notnull() & (data['summary'].str.strip() != '')]

    data['cleaned_text'] = data['comment_body'].apply(clean_text)
    data['cleaned_summary'] = data['summary'].apply(clean_text)

    # Save cleaned text to train tokenizer
    corpus_file = os.path.join(processed_data_folder, 'corpus.txt')
    all_text = pd.concat([data['cleaned_text'], data['cleaned_summary']])
    all_text.to_csv(corpus_file, index=False, header=False)

    # Train tokenizer if not already trained
    model_prefix = os.path.join(processed_data_folder, 'tokenizer')
    if not os.path.exists(f'{model_prefix}.model'):
        print("Training tokenizer...")
        train_sentencepiece_tokenizer(corpus_file, model_prefix=model_prefix)

    # Load tokenizer
    sp = load_tokenizer(model_path=f'{model_prefix}.model')

    # Tokenize input and label text
    data['input_ids'] = data['cleaned_text'].apply(lambda x: tokenize_with_special_tokens(x, sp))
    data['label_ids'] = data['cleaned_summary'].apply(lambda x: tokenize_with_special_tokens(x, sp))

    # Save the tokenized input and target sequences
    data_to_save = data[['comment_score', 'input_ids', 'label_ids']]
    data_to_save.to_csv(processed_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Processed data saved to: {processed_file_path}")

if __name__ == "__main__":
    main()
