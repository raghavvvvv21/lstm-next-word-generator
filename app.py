import os
import pickle
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_LEN_PATH = "max_len.pickle"


st.set_page_config(
    page_title="LSTM Next Word Generator",
    layout="centered",
)


# Some older tokenizer pickle files reference the standalone `keras` package.
# This lets them unpickle in TensorFlow-only environments.
sys.modules.setdefault("keras", tf.keras)
sys.modules.setdefault("keras.preprocessing", tf.keras.preprocessing)
sys.modules.setdefault("keras.preprocessing.text", tf.keras.preprocessing.text)
sys.modules.setdefault("keras.preprocessing.sequence", tf.keras.preprocessing.sequence)


def import_ml_dependencies():
    return load_model, pad_sequences


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    require_file(MODEL_PATH)
    require_file(TOKENIZER_PATH)
    require_file(MAX_LEN_PATH)

    load_model, pad_sequences = import_ml_dependencies()

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    with open(MAX_LEN_PATH, "rb") as file:
        max_len = pickle.load(file)

    if not isinstance(max_len, int):
        raise ValueError("max_len.pickle should contain an integer sequence length.")

    index_word = {index: word for word, index in tokenizer.word_index.items()}
    return model, tokenizer, index_word, max_len, pad_sequences


def resolve_input_length(model, max_len):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) > 1 and isinstance(input_shape[1], int):
        return input_shape[1]

    return max(1, max_len - 1)


def sample_prediction(probabilities, temperature=1.0, top_k=10):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.squeeze(probabilities)

    if probabilities.ndim != 1:
        probabilities = probabilities.reshape(-1)

    top_k = max(1, min(top_k, probabilities.shape[0]))
    top_indices = np.argpartition(probabilities, -top_k)[-top_k:]
    top_probs = probabilities[top_indices]

    if temperature <= 0:
        return int(top_indices[np.argmax(top_probs)])

    top_probs = np.log(top_probs + 1e-12) / temperature
    top_probs = np.exp(top_probs - np.max(top_probs))
    top_probs = top_probs / np.sum(top_probs)
    return int(np.random.choice(top_indices, p=top_probs))


def generate_text(
    seed_text,
    words_to_generate,
    model,
    tokenizer,
    index_word,
    max_len,
    pad_sequences,
    temperature,
    top_k,
):
    generated_words = []
    input_length = resolve_input_length(model, max_len)
    current_text = seed_text.strip()

    for _ in range(words_to_generate):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        if not token_list:
            break

        token_list = pad_sequences(
            [token_list],
            maxlen=input_length,
            padding="pre",
            truncating="pre",
        )

        predictions = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_prediction(predictions, temperature, top_k)
        predicted_word = index_word.get(predicted_index)

        if not predicted_word:
            break

        generated_words.append(predicted_word)
        current_text = f"{current_text} {predicted_word}"

    return current_text, generated_words


st.title("LSTM Next Word Generator")
st.caption("Type a starting phrase and let your trained LSTM continue it.")

try:
    model, tokenizer, index_word, max_len, pad_sequences = load_artifacts()
    artifacts_loaded = True
except Exception as exc:
    artifacts_loaded = False
    st.error(str(exc))
    st.info(
        "Expected files in this folder: lstm_model.h5, tokenizer.pickle, max_len.pickle"
    )


if artifacts_loaded:
    with st.sidebar:
        st.header("Generation")
        words_to_generate = st.slider("Words", min_value=1, max_value=50, value=10)
        temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Lower values are more predictable. Higher values are more varied.",
        )
        top_k = st.slider(
            "Top-k choices",
            min_value=1,
            max_value=50,
            value=10,
            help="Samples from the most likely k words.",
        )

        st.divider()
        st.metric("Vocabulary", f"{len(tokenizer.word_index):,}")
        st.metric("Max length", max_len)

    seed_text = st.text_area(
        "Starting text",
        value="",
        placeholder="Enter a few words to begin...",
        height=130,
    )

    col_generate, col_clear = st.columns([1, 1])
    generate_clicked = col_generate.button("Generate", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if generate_clicked:
        if not seed_text.strip():
            st.warning("Please enter some starting text first.")
        else:
            with st.spinner("Generating next words..."):
                result, generated_words = generate_text(
                    seed_text=seed_text,
                    words_to_generate=words_to_generate,
                    model=model,
                    tokenizer=tokenizer,
                    index_word=index_word,
                    max_len=max_len,
                    pad_sequences=pad_sequences,
                    temperature=temperature,
                    top_k=top_k,
                )

            if generated_words:
                st.subheader("Generated Text")
                st.write(result)

                st.subheader("New Words")
                st.write(" ".join(generated_words))
            else:
                st.warning(
                    "I could not generate a word from that input. Try using words from the model's training data."
                )

#venv/bin/streamlit run app.py