import cv2 
import os
import tempfile
import requests 
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import json as js
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect
import translate
import googletrans
import faiss
import numpy as np 
import clip
import torch
import json
from PIL import Image

from hangso import capture_index, Image_Captioning_URL, ObjectDetect_URL, headers

# CAPTURE IMAGE 
def capture_image():
    cam = cv2.VideoCapture(capture_index)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame

# SAVE IMAGE PATH
def save_image(frame, folder="image"):
    with open('count.txt', 'r') as fr:
        count = int(fr.readline())
    with open('count.txt', 'w') as fr:
        fr.write(str(count + 1))
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    image_path = os.path.join(folder, "captured_image.png")
    cv2.imwrite(image_path, frame)
    image_path = os.path.join('database/Keyframes', f'image_{count}.jpg') 
    cv2.imwrite(image_path, frame)
    return image_path

# TEXT TO SPEECH 
def text_to_speech(text, lang="vi"):
    if not text:
        print()
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_filename = fp.name
    
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_filename)
    
    try:
        data, fs = sf.read(temp_filename, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"Error playing sound: {e}")
    finally:
        os.remove(temp_filename)

# GENERATE DESCRIPTION
def generate_image_description(json_folder="json"):
    def read_json_file(file_path):
        with open(file_path, 'r') as file:
            return js.load(file)

    caption_file = os.path.join(json_folder, "caption.json")
    object_file = os.path.join(json_folder, "object.json")

    translator = GoogleTranslator(source='auto', target='vi')

    paragraph = ""

    try:
        caption_data = read_json_file(caption_file)
        object_data = read_json_file(object_file)

        if caption_data and isinstance(caption_data, list) and 'generated_text' in caption_data[0]:
            english_caption = caption_data[0]['generated_text']
            vietnamese_caption = translator.translate(english_caption)
            paragraph += vietnamese_caption.capitalize() + "."
        else:
            object_counts = {}
            for obj in object_data:
                if isinstance(obj, dict) and 'label' in obj:
                    label = obj['label']
                    translated_label = translator.translate(label)
                    object_counts[translated_label] = object_counts.get(translated_label, 0) + 1
                else:
                    raise ValueError("Dữ liệu đối tượng không hợp lệ (không có 'label' hoặc không phải là dictionary).")

            if object_counts:
                object_descriptions = [f"{count} {obj}" for obj, count in object_counts.items()]
                if len(object_descriptions) > 1:
                    objects_text = ", ".join(object_descriptions[:-1]) + f" và {object_descriptions[-1]}"
                else:
                    objects_text = object_descriptions[0]
                paragraph += f"Tôi nghĩ phía trước có {objects_text}."
            else:
                paragraph = "Đã có lỗi xảy ra trong quá trình xử lý hình ảnh "
    
    except (IOError, ValueError, KeyError) as e:
        paragraph = "Đã có lỗi xảy ra trong quá trình xử lý hình ảnh "  

    return paragraph

# RECORD VOICE
def listen_and_recognize(recognizer, microphone, timeout=10):
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=None)
        try:
            text = recognizer.recognize_google(audio, language="vi-VN").lower()
            return text, "vi"
        except:
            text = recognizer.recognize_google(audio, language="en-US").lower()
            return text, "en"
    except sr.WaitTimeoutError:
        return "", ""
    except sr.UnknownValueError:
        return "", ""
    except Exception as e:
        return "", ""

def get_object(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(ObjectDetect_URL, headers=headers, data=data)
    return response.json()

def get_obj_json(filename):
    os.makedirs("json", exist_ok=True)
    object = get_object(filename)
    with open("json/object.json", "w") as fr:
        js.dump(object, fr, indent=4)
    
def get_caption(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(Image_Captioning_URL, headers=headers, data=data)
    return response.json()
def get_cap_json(filename):
    os.makedirs("json", exist_ok=True)
    caption = get_caption(filename)
    with open("json/caption.json", "w") as fr:
        js.dump(caption, fr, indent=4)


class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate.
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        return text.lower()

    def __call__(self, text):
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text


class MyFaiss:
  def __init__(self): #root_database: str, bin_file: str, json_path: str):
    self.__device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, self.preprocess = clip.load("ViT-B/32", device=self.__device)
    self.translater = Translation()

  def image_warning(self, text, image_path):
    if detect(text) == 'vi':
        text = self.translater(text)
    image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.__device)
    text = clip.tokenize([text]).to(self.__device)

    # Tạo embeddings
    with torch.no_grad():
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

    # Chuẩn hóa embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Tính toán độ tương đồng
    similarity = (image_features @ text_features.T).item()
    return similarity