
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
from PIL import Image #Image Processing
import  pytesseract as tess
from PIL import Image
import cv2 as cv
import os


from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)



RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    # st.header("WebRTC demo")

    pages = {
        "Real time object detection (sendrecv)": app_object_detection,
        "Image to Text to Speech ": image_to_speech,
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
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"  
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


    if image is not None:

        input_image = Image.open(image) #read image
        st.image(input_image) #display image

        with st.spinner("🤖 AI is at Work! "):
            

            result = tess.image_to_string(input_image)

            st.write(result)
        pts.speak(result)
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
    main()
