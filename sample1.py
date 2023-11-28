
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pyttsx3 as pts
import cv2
import  pytesseract as tess
from PIL import Image, ImageFilter, ImageEnhance #Image Processing
import os
from playsound import playsound
from gtts import gTTS
import speech_recognition as sr


from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# file = "good"
# i="0"
# app =""
# def texttospeech(text, filename):
#     filename = filename + '.mp3'
#     flag = True
#     while flag:
#         try:
#             tts = gTTS(text=text, lang='en', slow=False)
#             tts.save(filename)
#             flag = False
#         except:
#             print('Trying again')
#     playsound(filename)
#     # os.close(filename)
#     os.remove(filename)
#     return

# def speechtotext(duration):
#     global i
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         r.adjust_for_ambient_noise(source, duration=1)
#         playsound('voice_based_email_mysite_speak.mp3')
#         audio = r.listen(source, phrase_time_limit=duration)
#     try:
#         response = r.recognize_google(audio)
#     except:
#         response = 'N'
#     return response

# def view_page():
#     global i,page_title
#     text1 = "Welcome to our vision website. select pages to continue"
#     texttospeech(text1, file + i)
#     i = i + str(1)

#     flag = True
#     while (flag):
#         texttospeech("Choose the app mode", file + i)
#         i = i + str(1)
#         app = speechtotext(3)
#         if app != 'N':
#             texttospeech("You meant " + app + "  say again", file + i)
#             i = i + str(1)
#             say = speechtotext(2)
#             print("rfrifr",say)
#             if say == 'text to speech' or say == 'Text to Speech':
#                 page_title = page_title.replace('object detection', 'text to speech')
#                 break
#             else:
#                 break
#             # elif say == 'text to speech' or say == 'Text To Speech':
#             # page_title = page_title.replace('object detection', 'text to speech')
#             # flag = False

#         else:
#             texttospeech("could not understand what you meant:", file + i)
#             i = i + str(1)
#         # app = app.strip()
#         # app = app.replace(' ', '')
#     app = app.lower()
#     print(app)
#     flag= False


def main():
    # global page_title
    pages = {
        "object detection": app_object_detection,
        "text to speech": image_to_speech,
    }
    
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()


    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")




def app_object_detection():
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    @st.experimental_singleton
    def generate_label_colors():
        return np.random.uniform(0, 255, size=(len(CLASSES), 3))

    COLORS = generate_label_colors()

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    # Session-specific caching
    cache_key = "object_detection_dnn"
    if cache_key in st.session_state:
        net = st.session_state[cache_key]
    else:
        net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
        st.session_state[cache_key] = net

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    def _annotate_image(image, detections):
        # loop over the detections
        (h, w) = image.shape[:2]
        result: List[Detection] = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                name = CLASSES[idx]
                result.append(Detection(name=name, prob=float(confidence)))

                # display the prediction
                label = f"{name}: {round(confidence * 100, 2)}%"
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    image,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[idx],
                    2,
                )
        cv2.imshow("Detected Object", image)
        pts.speak(label)
        return image, result

    result_queue = (
        queue.Queue()
    )  # TODO: A general-purpose shared state object may be more useful.

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()
        annotated_image, result = _annotate_image(image, detections)

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put(result)  # TODO:

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)

def image_to_speech():
    st.title( "Extract Text from Images")

    #subtitle

    st.markdown("")

    #image uploader
    image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


    @st.cache
    def load_model(): 
        reader = tess.image_to_string(input_image)
        return reader 
    def dropShadow( input_image, offset=(5,5), background=0xffffff, shadow=0x444444, 
                border=8, iterations=3):
        # Create the backdrop image -- a box in the background colour with a 
        # shadow on it.
        totalWidth = input_image.size[0] + abs(offset[0]) + 2*border
        totalHeight = input_image.size[1] + abs(offset[1]) + 2*border
        back = Image.new(input_image.mode, (totalWidth, totalHeight), background)
        
        # Place the shadow, taking into account the offset from the image
        shadowLeft = border + max(offset[0], 0)
        shadowTop = border + max(offset[1], 0)
        back.paste(shadow, [shadowLeft, shadowTop, shadowLeft + input_image.size[0], 
            shadowTop + input_image.size[1]] )
        
        # Apply the filter to blur the edges of the shadow.  Since a small kernel
        # is used, the filter must be applied repeatedly to get a decent blur.
        n = 0
        while n < iterations:
            back = back.filter(ImageFilter.BLUR)
            n += 1
            
        # Paste the input image onto the shadow backdrop  
        imageLeft = border - min(offset[0], 0)
        imageTop = border - min(offset[1], 0)
        back.paste(input_image, (imageLeft, imageTop))
        
        return back

    if image is not None:

        input_image = Image.open(image) #read image
        input_image = input_image.convert('L')
        dropShadow(input_image).show()
        dropShadow(input_image, background=0xeeeeee, shadow=0x444444, offset=(0,5)).show()
        # input_image = input_image.filter(ImageFilter.MedianFilter())
        # enhancer = ImageEnhance.Contrast(input_image)
        # input_image = enhancer.enhance(1)
        # input_image = input_image.convert('1')
        # input_image = input_image.filter(ImageFilter.BLUR)
        # input_image = input_image.filter(ImageFilter.MinFilter(3))
        # input_image = input_image.filter(ImageFilter.MinFilter)
        st.image(input_image) #display image
        
        with st.spinner("🤖 AI is at Work! "):
            result = tess.image_to_string(input_image)
            st.write(result)
        # pts.speak(result)
        #st.success("Here you go!")
        st.balloons()
    else:
        st.write("Upload an Image")




if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)
    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    # view_page()
    main()

