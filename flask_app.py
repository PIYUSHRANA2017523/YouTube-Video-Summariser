from flask import Flask, request
from pytube import YouTube
from pydub import AudioSegment
import torch
import os
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import pipeline

# ---------------------------------------------------------------------------

app = Flask(__name__)

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


def get_audio(yt, filename):
    print("Generating Audio ....")

    # To query the streams that contain only the audio track and download as "filename"
    # filename = 'a2'
    audio_stream = yt.streams.filter(only_audio=True).first().download("Audios", filename=filename+".mp4")

    # CONVERT mp4 TO wav FORMAT
    load_audio = AudioSegment.from_file('Audios/'+filename+".mp4", format="mp4")
    load_audio.export("wav_format/"+filename+".wav", format="wav")

    print("Audio Conversion...    DONE !!")
    chunks_of_audio(filename)

def chunks_of_audio(filename):
    print("")
    print("Generating Chunks of Audio")
    SAMPLING_RATE = 16000
    # Input Audio File (.wav format)
    wav = read_audio('wav_format/'+filename+'.wav', sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, audio_chunk_model, sampling_rate=SAMPLING_RATE)

    # Saving Chunks of Audio
    print("Saving Chunks")
    for i in range(len(speech_timestamps)):
        os.makedirs(f'wav_format/{filename}_chunks', exist_ok=True)
        save_audio(f'wav_format/{filename}_chunks/{filename}_chunk_{i}.wav',
                   collect_chunks(speech_timestamps[i], wav),
                   sampling_rate=SAMPLING_RATE)
    print("Saving Chunks...   DONE !!")
    print("")


def audio_to_text(filename):
    print("Generating text from audio chunks ...")
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

    print("GENERATED TEXT : ")
    print(full_video_text)

    # GETTING SUMMARY
    print("")
    print("Generating Summary ...")
    result = summarizer(full_video_text)
    summarized_text = result[0]['summary_text']

    print("SUMMARY : ")
    print(summarized_text)
    return summarized_text

@app.get('/summary')
def summary_api():
    # url = request.args.get('url','')
    video_link = request.args.get('url','')
    if (video_link != ""):
        try:
            # object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(video_link)
            print("Connection Successful !!")
        except:
            print("Connection Error")
    filename = 'a3'
    get_audio(yt, filename)
    summary = audio_to_text(filename)
    return summary, 200


if __name__ == '__main__':
    app.run()


# s = "FOR EVEN AS WE CELEBRATE TO NIGHT WE KNOW THE CHALLENGES THAT TO MORROW WILL BRING ONE BASED ON MUTUAL INTEREST AND MUTUAL RESPECT JUST AS MOSLEMS DO NOT FIT A CRUDE STEREOTYPE AMERIC A IS NOT THE CRUDE STEREOTYPE OF A SELF INTERESTED EMPIRE TO SAY THAT FORCE MAY SOMETIMES BE NECESSARY IS NOT A CALL TO CYNICISM IT IS A RECOGNITION OF HISTORY THE IMPERFECTIONS OF MAN AND THE LIMITS OF REASON WE GATHER HERE TO HONOR THE COURAGE OF ORDINARY AMERICANS WILLING TO ENDURE BILLY CLUBS THE CHASTENING ROD TIER GAS AND THE TRAMPLING HOOK MEN AND WOMEN WHO DESPITE THE GUSH OF BLOOD AND SPLINTERED BONE WOULD STAY TRUE TO THEIR NORTH STAR AND KEEP MARCHING TOWARD JUSTICE BLINDED BY HATRED THE ALLEGED KILLER COULD NOT SEE THE GRACE ARE THE GREATEST OF OUR LIFETIME SURROUNDING REVEREND PICMIC AND THAT BIBLE STUDY GROUP LIGHT OF LOVE THAT SHONE AS THEY OPENED THE CHURCH DOORS AND INVITED A STRANGER TO JOIN IN THEIR PRAYER CIRCE EALLEGED KILLER COULD HAVE NEVER ANTICIPATED THE WAY THE FAMILIES OF THE FALLEN WOULD RESPOND WHEN THEY SAW HIM IN COURT IN THE MIDST OF UNSPEAKABLE GRIEF WITH WORDS OF FERGETNESS HE COULDN'T IMAGINE THAT AD WHILE THIS NATION HAS BEEN TESTED BY WAR AND HAS BEEN TESTED BY RECESSION AND ALL MANNER OF CHALLENGE I STAND BEFORE YOU AGAIN TO NIGHT AFTER ALMOST TWO TERMS AS YOUR PRESIDENT TO TELL YOU I AM MORE OPTIMISTIC ABOUT THE FUTURE OF AMERICA BENEVER BEFOLE ANDBYS HE KNEWS FANS THANKS FOR CHACKING OUT OUR EU TUBE CHANNEL SUBSCRIBE BY CLICKING"