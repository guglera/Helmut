#!/usr/bin/env python3

import speech_recognition as sr
import google.cloud.texttospeech as tts
from google.oauth2 import service_account
import openai
import os, sys
import pyaudio
from pyaudio import PyAudio
import wave
import asyncio
import struct
import pvporcupine

Prompts = []
Replies = []
LANGUAGE = "de"
stop_word = "Helmut"
MODELTYPE = "gpt-3.5-turbo"

# OpenAI Access - Add your credentials here:
openai.organization = ""
openai.api_key = ""
openai.Model.list()

# Google Cloud credentials - Add path to your gcloud *.json file here:
gpath = os.path.join(sys.path[0], "")

# Google Access
GOOGLE_CLOUD_SPEECH_CREDENTIALS = gpath
TTScredentials = service_account.Credentials.from_service_account_file(gpath)

selectedVoice = "de-DE-Wavenet-B"

# suppress noisy ALSA STDOUT
class pyaudio:
    """
    PyAudio is noisy af every time you initialise it, which makes reading the
    log output rather difficult.  The output appears to be being made by the
    C internals, so I can't even redirect the logs with Python's logging
    facility.  Therefore the nuclear option was selected: swallow all stderr
    and stdout for the duration of PyAudio's use.

    Lifted and adapted from StackOverflow:
      https://stackoverflow.com/questions/11130156/
    """

    def __init__(self):

        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

        self.pyaudio = None

    def __enter__(self) -> PyAudio:

        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

        self.pyaudio = PyAudio()

        return self.pyaudio

    def __exit__(self, *_):

        self.pyaudio.terminate()

        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

async def record_input():
    with pyaudio() as audio:
        r = sr.Recognizer()
        m = sr.Microphone(device_index=3)
        with m as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

    try:
        InputPrompt = r.recognize_google_cloud(audio, language='de-de', credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
        print("Google Cloud Speech thinks you said: __ " + InputPrompt)
    except sr.UnknownValueError:
        print("Google Cloud Speech could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Cloud Speech service; {0}".format(e))

    return InputPrompt

def play_audio(file: str):
    #define stream chunk   
    chunk = 1024  

    #open a wav format music  
    f = wave.open(file,"rb")  
    #instantiate PyAudio  
    # p = pyaudio.PyAudio()  
    with pyaudio() as p:
    #open stream  
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
            channels = f.getnchannels(),  
            rate = f.getframerate(),  
            output = True)  
        #read data  
        data = f.readframes(chunk)  

        #play stream  
        while data:  
            stream.write(data)  
            data = f.readframes(chunk)  

        #stop stream  
        stream.stop_stream()  
        stream.close()  

        #close PyAudio  
        p.terminate()

def write_to_log(log: str):
    text_file = open("logfile.txt", "a")
    file_out = text_file.write(log + "\n")
    text_file.close()

async def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient(credentials=TTScredentials)
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{voice_name}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

def append_prompt(text: str):
    Prompts.append({"role": "user", "content": text})

def append_reply(text: str):
    Replies.append({"role": "assistant", "content": text})

async def get_response(chat):
    OpenAiApiResp = openai.ChatCompletion.create(
        model = MODELTYPE,
        messages = chat
    )
    return OpenAiApiResp['choices'][0]['message']['content']

chatdialogue = [
    {"role": "system",
    "content": "You are a helpful assistant with a voice interface. Keep your responses succinct since the user is interacting with you through a voice interface. Your responses should be a few sentences at most. Always provide your responses in the language that corresponds to the ISO-639-1 code: {LANGUAGE}."}
    ]

async def helmut():
    play_audio("alert.wav")
    for x in range(30):
        print("Sprich mit mir ...")
        recorded_input = await record_input()
        print("recorded successfully")
        if stop_word in recorded_input:
            print("Helmut gestoppt!")
            break
        if recorded_input != " ":
            append_prompt(recorded_input)
            write_to_log("Q: " + recorded_input)
            chatdialogue.append(Prompts[x])
            response = await get_response(chatdialogue)
            print(response)
            append_reply(response)
            write_to_log("A: " + response)
            await text_to_wav(selectedVoice, response)
            chatdialogue.append(Replies[x])
            play_audio("de-DE-Wavenet-B.wav")
        else:
            print("Helmut hat nichts gehört und wird nicht aktiv!")
            append_prompt("no input")
            append_reply("no response")

    play_audio("stopped.wav")

    await main()

async def main():
    current_path = os.path.join(sys.path[0])
    # Add porcupine credentials here:
    porcupine = pvporcupine.create(access_key="",
    keyword_paths = [current_path +  "/Helmut_de_linux_v2_2_0.ppn"], 
    model_path = current_path + "/porcupine_params_de.pv")
    # recoder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)


    MicInput = sr.Microphone(device_index=4, sample_rate=porcupine.sample_rate, chunk_size=porcupine.frame_length)

    try:
        with MicInput as source:
            while True:
                pcm = source.stream.read(porcupine.frame_length)
                audio_frame = struct.unpack_from("h" * porcupine.frame_length, pcm)
                # recoder.start()
                # keyword_index = porcupine.process(recoder.read())
                keyword_index = porcupine.process(audio_frame)
                if keyword_index >= 0:
                    print(f"Detected Helmut mit Hut!")
                    # recoder.stop()
                    await helmut()

    except KeyboardInterrupt:
        # recoder.stop()
        # return
        print("Keyboard Interrupt, going to shutdown")
        loop.stop()
        # loop.close()
    # finally:
        porcupine.delete()
        print("Helmut shut down")
        # recoder.delete()
        return


# asyncio.run(main())
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
# loop.run_forever()
