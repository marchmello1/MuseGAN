import streamlit as st
import numpy as np
import tensorflow as tf
from create_generator_model import get_notes, LATENT_DIMENSION, GAN, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, SAMPLE_INTERVAL
from generate_music import generate_music, create_midi
from play_midi import play_midi

st.title("Music Generation with GANs")

# Load the trained generator model
generator_model = tf.keras.models.load_model("generator_model.keras")

# Load the processed notes and get the number of unique pitches
notes = get_notes()
n_vocab = len(set(notes))

# Function to generate music
def generate_and_play_music():
    # Generate new music sequence
    generated_music = generate_music(generator_model, LATENT_DIMENSION, n_vocab)

    # Create a MIDI file from the generated music
    create_midi(generated_music, 'generated_music')

    # Play the generated music
    play_midi('generated_music.mid')

# Sidebar for setting parameters
st.sidebar.header("GAN Parameters")
sequence_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=500, value=SEQUENCE_LENGTH, step=10)
latent_dimension = st.sidebar.slider("Latent Dimension", min_value=100, max_value=2000, value=LATENT_DIMENSION, step=100)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=EPOCHS)
batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=128, value=BATCH_SIZE, step=16)
sample_interval = st.sidebar.slider("Sample Interval", min_value=1, max_value=100, value=SAMPLE_INTERVAL)

# Train GAN Button
if st.button("Train GAN"):
    st.text("Training GAN...")
    gan = GAN(rows=sequence_length)
    gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
    st.text("Training complete!")

# Generate and Play Music Button
if st.button("Generate and Play Music"):
    st.text("Generating and playing music...")
    generated_music = generate_music(generator_model, LATENT_DIMENSION, n_vocab, notes)
    create_midi(generated_music, 'generated_music')
    play_midi('generated_music.mid')
