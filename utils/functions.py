import pickle
import cv2
import numpy as np
import ast
import os
from settings import LOCATION_FOLDER
from settings import HEIGHT
from settings import OCR_CHARACTER_MAPPING
from settings import EMPTY_TEXT
import itertools
from utils.tf_serving import TfServing


def save_location(location, form_id):

    location = ast.literal_eval(location)
    file_name = os.path.join(LOCATION_FOLDER, form_id + ".pkl")
    with open(file_name, "wb") as f:
        pickle.dump(location, f)


def load_location(form_id):
    file_name = os.path.join(LOCATION_FOLDER, form_id + ".pkl")
    try:
        with open(file_name, "rb") as f:
            tmp = pickle.load(f)
        if isinstance(tmp, list):
            return tmp
        else:
            return None
    except:
        return None


def decode_image(image_base64):
    image = image_base64.stream.read()
    image = np.frombuffer(image, np.uint8)
    # Gray scale image
    image = cv2.imdecode(image, 0)
    return image


def resize_image(image):
    ratio = image.shape[0] / HEIGHT
    image = cv2.resize(image, (int(image.shape[1] / ratio), HEIGHT))
    return image


def normalize_image(image):
    return image / 255.0


def prepare_ocr_image(image):
    image = resize_image(image)
    image = normalize_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image


def crop_image(image, location):
    cropped_list = []
    print(image.shape)
    for l in location:
        x = int(l["x"])
        y = int(l["y"])
        height = int(l["height"])
        width = int(l["width"])
        tmp_image = image[y : y + height, x : x + width]
        cropped_list.append(prepare_ocr_image(tmp_image))
    return cropped_list


def call_tf_serving(cropped_list):
    text = []
    tf_serving = TfServing()
    for c in cropped_list:
        ocr_result = tf_serving.call(c)
        ocr_result = process_ocr_result(ocr_result)
        text.append(ocr_result)
    return text


def process_ocr_result(ocr_result):
    def label_to_text(label):
        text = ""
        for l in label:
            if l < len(OCR_CHARACTER_MAPPING):
                text += OCR_CHARACTER_MAPPING[l]
        return text

    if len(ocr_result.shape) < 2:
        return EMPTY_TEXT
    ocr_result = np.argmax(ocr_result, axis=-1)
    ocr_result = [k for k, _ in itertools.groupby(ocr_result)]
    recognized_text = label_to_text(ocr_result)
    return recognized_text
