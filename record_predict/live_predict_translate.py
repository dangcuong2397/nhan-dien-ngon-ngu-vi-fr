import os
import sys
import time
import uuid
import argparse
import subprocess
import soundfile as sf
import sounddevice as sd
from tensorflow.keras.models import load_model

from google_apis import text_to_speech, transcribe_speech, translate_text

lib_dir = os.path.join(os.getcwd(), "../tensorflow")
sys.path.append(lib_dir)
from SpectrogramGenerator import SpectrogramGenerator
from predict import predict
from compile_model import compile_model


def count_down():
    print('ll start recording in:')
    countdown = 3
    while (countdown):
        print(countdown)
        countdown -=  1
        time.sleep(0.7)
    print('Recording!')


def play(audio_file_path):
    cmd = ["ffplay", "-nodisp", "-autoexit", audio_file_path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()


def live_predict_translate(model_dir, use_google_apis, config):
    fs = 44100 #Hz
    duration = 3 #seconds

    to_record = 'y'
    while to_record == 'y':

        count_down()
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        print('Finished recording')

        filename = 'recording_{0}.wav'.format(uuid.uuid4())
        sf.write(filename, recording, fs)
        
        print('\nIdentifying the language...')
        print('The proportions below are in the following order: Croatian, French, Spanish')
        result = predict(model_dir, filename, config)
        
        if use_google_apis:
            try:
                transcribed = transcribe_speech(filename, result)
                translated = translate_text(transcribed)
                output = 'output.mp3'
                text_to_speech(translated, output)
                print('Playing the translation!')
                play(output)
                os.remove(output)
            except Exception as e:
                print('\nThe following exception occured:')
                print(e)

                print("--------------------------\n")
                print("Note from the author: If it's not a Google API error,")
                print("it may just be that the sounddevice wasn't able to access the microphone")
                print("--------------------------\n")
            os.remove(filename)
            
        print('\nWould you like to record and identify again? [y/n]: ')
        to_record = input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', default="../tensorflow/config.yaml")
    parser.add_argument('--google_apis', dest='google_apis', default=0,
                        type=int, help="Pass '1' if you'd like to use google APIs for translation, otherwise ignore")
    args = parser.parse_args()
    model = load_model(args.model_dir)
    model = compile_model(model)
    live_predict_translate(model, args.google_apis, args.config)