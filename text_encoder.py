import tensorflow as tf
import keras
from keras.layers import Embedding, LSTM, Dropout
# from transformers import TFBertModel, BertTokenizer

def create_text_encoder(embedding_dim=32):
        # Load pre-trained BERT model
    input_text = keras.layers.Input(shape=(30,768), dtype=tf.float32)
    # bert_layer = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)(input_text)
    print('bert_layer')
    # Add LSTM layers
    lstm_layer1 = tf.keras.layers.LSTM(256, return_sequences=True)(input_text)
    dropout_layer1 = tf.keras.layers.Dropout(0.5)(lstm_layer1)
    lstm_layer2 = tf.keras.layers.LSTM(64)(dropout_layer1)

    # Add attention mechanism
    attention_probs = tf.keras.layers.Dense(1, activation='tanh')(lstm_layer2)
    attention_probs = tf.keras.layers.Flatten()(attention_probs)
    attention_probs = tf.keras.layers.Activation('softmax')(attention_probs)
    attention_probs = tf.keras.layers.RepeatVector(lstm_layer2.shape[-1])(attention_probs)
    attention_probs = tf.keras.layers.Permute([2, 1])(attention_probs)
    attention_probs = tf.keras.layers.Multiply()([lstm_layer2, attention_probs])
    text_embedding = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_probs)

    # Add dense layers
    text_embedding = tf.keras.layers.Dense(512, activation='relu')(text_embedding)
    text_embedding = tf.keras.layers.Dropout(0.5)(text_embedding)
    text_embedding = tf.keras.layers.Dense(256, activation='relu')(text_embedding)
    text_embedding = tf.keras.layers.Dropout(0.5)(text_embedding)
    text_embedding = tf.keras.layers.Dense(embedding_dim, name='text_embedding')(text_embedding)

    # Define text encoder model
    text_encoder = tf.keras.models.Model(inputs=input_text, outputs=text_embedding, name='text_encoder')

    return text_encoder
# embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), 

# def encode_text(text):
#     tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
#     tokenizer.fit_on_texts([text])
#     sequences = tokenizer.texts_to_sequences([text])
#     padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post')
#     encoder = create_text_encoder()
#     encoder.load_weights('text_encoder_weights.h5')
#     text_embedding = encoder.predict(padded_sequences)
#     return text_embedding
