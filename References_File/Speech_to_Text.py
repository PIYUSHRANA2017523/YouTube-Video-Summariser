from transformers import AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

access_token = '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'

model = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

import librosa

input_file = 'ytaudio.mp4'

print(librosa.get_samplerate(input_file))

# Stream over 30 seconds chunks rather than load the full file
stream = librosa.stream(
    input_file,
    block_length=30,
    frame_length=16000,
    hop_length=16000
)

import soundfile as sf

for i, speech in enumerate(stream):
    sf.write(f'{i}.wav', speech, 16000)


# CHUNK TRANSCRIPTION

audio_path = []
for a in range(i+1):
    audio_path.append(f'{a}.wav')

print(audio_path)

transcriptions = model.transcribe(audio_path)

full_transcript = ' '

for item in transcriptions:
    full_transcript += ''.join(item['transcription'])

print(full_transcript)



# TEXT SUMMARIZATION

from transformers import pipeline

summarization = pipeline('summarization')

num_iters = int(len(full_transcript) / 1000)
summarized_text = []

for i in range(0, num_iters + 1):
    start = 0
    start = i * 1000
    end = (i+1) * 1000

    out = summarization(full_transcript[start:end], min_length=5, max_length=20)
    out = out[0]
    out = out['summary_text']
    # print("Summarized text\n" + out)
    summarized_text.append(out)

print(summarized_text)