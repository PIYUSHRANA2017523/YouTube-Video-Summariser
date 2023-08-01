from flask import Flask, request, session
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from googletrans import Translator
from googletrans import LANGUAGES


app = Flask(__name__)
app.secret_key = '20021934'

@app.route('/summary',  methods=['GET','POST'])
def summary_api():
    url = request.args.get('url','')
    print(url) 
    video_id = url.split('=')[1]
    summary = get_summary(get_transcript(video_id))
    session['summarized_text']=summary
    return summary, 200


def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([d['text'] for d in transcript_list])
    print(transcript)
    return transcript

def get_summary(transcript):
    summariser = pipeline('summarization')
    summary = ''
    for i in range (0, (len(transcript)//1000)+1):
        summary_text = summariser(transcript[i*1000:(i+1)*1000])[0]['summary_text']
        print(summary_text)
        summary = summary + summary_text + ' '
    return summary



























@app.route('/translate',  methods=['GET','POST'])
def get_translate():
    para = session.get('summarized_text','')
    # para = "You ara a FOOL"
    target_language = request.args.get('language')
    translator = Translator()
    language_code = get_language_code(target_language)
    translation = translator.translate(para, dest=language_code)
    translated_para = (f"Translation ({target_language}):{translation.text}")
    return translated_para


def get_language_code(language):
    for code, name in LANGUAGES.items():
        if language.lower()==name.lower():
           return code
    return None

if __name__ == '__main__':
    app.run()

    