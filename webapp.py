"""
# Youtube Video Link
VIDEO_URL = "https://youtube.com/watch?v=hWLf6JFbZoo"
"""

import streamlit as st
from pytube import YouTube
from pydub import AudioSegment
import torch
import os
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import pipeline

# ---------------------------------------------------------------------------

torch.set_num_threads(1)

audio_chunk_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=True,
                                          onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def collect_chunks(tss: dict,
                   wav: torch.Tensor):
    chunks = []
    chunks.append(wav[tss['start']: tss['end']])
    return torch.cat(chunks)


# FOR AUDIO_TO_TEXT
# LOAD THE MODEL AND PROCESSOR
model = Wav2Vec2ForCTC.from_pretrained('./saved_model/')
processor = Wav2Vec2Processor.from_pretrained('./saved_model/')

# SUMMARIZER MODEL
summarizer = torch.load("summarizer_model/long-t5-tglobal-base-16384-book-summary.pt")

# ---------------------------------------------------------------------------


def get_audio(yt, filename):
    st.text("Generating Audio ....")

    # To query the streams that contain only the audio track and download as "filename"
    # filename = 'a2'
    audio_stream = yt.streams.filter(only_audio=True).first().download("Audios", filename=filename+".mp4")

    # CONVERT mp4 TO wav FORMAT
    load_audio = AudioSegment.from_file('Audios/'+filename+".mp4", format="mp4")
    load_audio.export("wav_format/"+filename+".wav", format="wav")

    st.text("Audio Conversion...    DONE !!")
    chunks_of_audio(filename)

def chunks_of_audio(filename):
    st.text("")
    st.text("Generating Chunks of Audio")
    SAMPLING_RATE = 16000
    # Input Audio File (.wav format)
    wav = read_audio('wav_format/'+filename+'.wav', sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, audio_chunk_model, sampling_rate=SAMPLING_RATE)

    # Saving Chunks of Audio
    st.text("Saving Chunks")
    for i in range(len(speech_timestamps)):
        os.makedirs(f'wav_format/{filename}_chunks', exist_ok=True)
        save_audio(f'wav_format/{filename}_chunks/{filename}_chunk_{i}.wav',
                   collect_chunks(speech_timestamps[i], wav),
                   sampling_rate=SAMPLING_RATE)
    st.text("Saving Chunks...   DONE !!")
    st.text("")


def audio_to_text(filename):
    st.text("Generating text from audio chunks ...")
    # AUDIO TO TEXT
    full_video_text = ""
    transcriptions = []
    path = f'wav_format/{filename}_chunks'
    for audio in os.listdir(path):
        data, samplerate = sf.read(path + "/" + audio)

        # tokenize
        input_values = processor(data,
                                 sampling_rate=16000,
                                 return_tensors="pt",
                                 padding="longest").input_values

        # retrieve logits
        logits = model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        transcriptions.append(transcription[0])
        full_video_text += " " + transcription[0]

    st.text("GENERATED TEXT : ")
    st.write(full_video_text)

    # GETTING SUMMARY
    st.text("")
    st.text("Generating Summary ...")
    result = summarizer(full_video_text)
    summarized_text = result[0]['summary_text']

    st.text("SUMMARY : ")
    st.write(summarized_text)



def main():
    """Youtube Video Summarizer App"""

    st.title("Youtube Video Summarizer App")
    st.text("Build with Streamlit")
    st.text("")
    st.text("")
    video_link = st.text_input("Enter video link")
    if (video_link != ""):
        try:
            # object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(video_link)
            print("Connection Successful !!")
            filename = 'a3'
            get_audio(yt, filename)
            audio_to_text(filename)
        except:
            print("Connection Error")


if __name__ == '__main__':
    main()
