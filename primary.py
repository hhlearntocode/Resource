import easyocr
from side_function import capture_image, save_image, text_to_speech, MyFaiss, listen_and_recognize
import google.generativeai as genai
from typing import List
import re
from deep_translator import GoogleTranslator
import threading
import queue
import speech_recognition as sr

cosine_faiss = MyFaiss()
# GEMINI API PREPARATION
genai.configure(api_key='AIzaSyB3FmHSpGtQSIQMhbrSQuqAtT6x8cDTUMw')
# genai.configure(api_key='AIzaSyAN7GhFw4Aaci7MupJnB8AtzT-JfiOATTA')
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 250,
#   "speedup": True,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)


def classify_command(command: str, model):
    command_types: List[str] = [
        "trích xuất văn bản",
        "tìm vật",
    ]
    
    if command == "get_info":
        return {f"feat{i+1}": (i+1, desc) for i, desc in enumerate(command_types)}
    
    prompt = (
        f"Câu lệnh: '{command}' thuộc tính năng nào trong các tính năng sau:\n"
        + "\n".join(f"{i+1}) {label}" for i, label in enumerate(command_types))
        + "\nChỉ trả lời bằng số thứ tự tương ứng."
    )
    
    response = model.generate_content(prompt)
    
    # Xử lý kết quả trả về
    response_text = response.text.strip()
    match = re.search(r'\d+', response_text)
    
    if match:
        index = int(match.group()) - 1
        if 0 <= index < len(command_types):
            return f"feat{index + 1}"
    else: 
        text_to_speech(f"Không thể phân loại câu lệnh")
        return -1
    
    raise ValueError(f"Không thể phân loại câu lệnh: {command}. Phản hồi: {response_text}")

def OCR(filename):
    try:
        reader = easyocr.Reader(['vi', 'en'])
        result = reader.readtext(filename)
        
        text_in_frame = ""
        for image_result in result:
            if isinstance(image_result, (tuple, list)) and len(image_result) == 3:
                bbox, text, prob = image_result
                text_in_frame += ' ' + text
        
        if text_in_frame.strip() == "":
            text_in_frame = " "
            text_to_speech("Đầu vào không thoả mãn. Xin hãy thử lại với đầu vào chứa văn bản")
            return
        
        print("Văn bản nhận được:", text_in_frame)

        translator = GoogleTranslator(source='auto', target='vi')
        translated_text = translator.translate(text_in_frame)
        # text_to_speech(translated_text)

        response = model.generate_content(
            f"Hãy giúp tôi dự đoán đoạn text sau với những từ chưa được xác định do sai sót từ model OCR: {translated_text}"
            # f"Xin hãy chỉnh sửa lại theo văn phong của tiếng Việt và giải nghĩa câu vừa chỉnh: {translated_text}"
        )

        clean_text = response.text.replace('*', '')
        
        return clean_text

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return "Lỗi khi xử lý văn bản."
    

class ImageProcessor(threading.Thread):
    def __init__(self, task_queue, result_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            task_type, args = task

            if task_type == "feat1":
                result = self.process_feat1()
            elif task_type == "feat2":
                result = self.process_feat2()
            self.result_queue.put((task_type, result))
            self.task_queue.task_done()



    def process_feat1(self):
        frame = capture_image()
        image_path = save_image(frame)
        return OCR(image_path)

    def process_feat2(self, object_name):
        frame = capture_image()
        image_path = save_image(frame)
        similarity = cosine_faiss.image_warning(object_name, image_path)
        return f"{object_name} đang ở trước mặt bạn" if similarity > 0.2 else "không tìm thấy"


def main():
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    audio_queue = queue.Queue()

    # Start threads
    image_processor = ImageProcessor(task_queue, result_queue)
    image_processor.start()

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    text_to_speech("Xin chào, tôi có thể giúp gì cho bạn?", "vi")

    while True:
        # command, lang = listen_and_recognize(recognizer, microphone, timeout=10)
        command = input("Nhập: ")
        print(f"Đã nhận diện: {command}")

        if not command:
            text_to_speech("Tôi không nghe rõ. Vui lòng thử lại.", "vi")
            continue

        if any(word in command.lower() for word in ["dừng", "không có gì", "thoát", "kết thúc", "stop", "nothing", "exit", "quit"]):
            text_to_speech("Hẹn gặp lại, hãy nhờ tôi khi bạn cần nhé", "vi")
            break

        task = classify_command(command, model)
        print(f"phân loại input ra : {task}")

        if task == "feat1":
            text_to_speech("Đã chụp hình, đang xử lý.", "vi")
            print("feat 1 hoàn tất!")    
            task_queue.put(("feat1", ()))

        elif task == "feat2":
            task_queue.put(("feat2", ()))
            print("feat 2 hoàn tất!")   

        elif task == "feat3":
            text_to_speech("Bạn cần tìm vật gì?", "vi")
            object_name, _ = listen_and_recognize(recognizer, microphone, timeout=10)
            # object_name = input("object name: ")
            if object_name: 
                task_queue.put(("feat3", (object_name,)))
            else:
                text_to_speech("Tôi không nghe rõ tên vật. Vui lòng thử lại.", "vi")
            print("feat 3 hoàn tất!")   

        elif task == "feat4":
            text_to_speech("đang kiểm tra nguy hiểm xung quanh.", "vi")
            task_queue.put(("feat4", ()))
            print("feat 4 hoàn tất!")   

        elif task == 'feat5':
            text_to_speech("đang kiểm tra người quen phía trước", "vi")
            task_queue.put(("feat5", ()))
            print("feat 5 hoàn tất!")

        else:
            text_to_speech("Tôi không hiểu lệnh đó. Vui lòng thử lại.", "vi")
            continue

        task_type, result = result_queue.get()
        text_to_speech(result, "vi")
        text_to_speech("Tôi đang chờ lệnh tiếp theo của bạn", "vi")

    # Stop threads
    task_queue.put(None)
    audio_queue.put(None)
    image_processor.join()