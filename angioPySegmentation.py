import os
import numpy as np
import os.path
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import pydicom
import glob
import mpld3
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import tifffile
from streamlit_plotly_events import plotly_events
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import predict
import angioPyFunctions
import scipy
import cv2
import ssl
import pooch

# ----------------- SEPARATE LOGGER SETUP -----------------
import logging

logger = logging.getLogger("AngioPyLogger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
# ----------------- END LOGGER SETUP ---------------------

ssl._create_default_https_context = ssl._create_unverified_context

logger.debug("Setting Streamlit page config.")
st.set_page_config(page_title="AngioPy Segmentation", layout="wide")

if 'stage' not in st.session_state:
    st.session_state.stage = 0
    logger.debug("Initialized st.session_state.stage to 0.")

logger.debug("Retrieving segmentation model weights via pooch.")
segmentationModelWeights = pooch.retrieve(
    url="doi:10.5281/zenodo.13848135/modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth",
    known_hash="md5:bf893ef57adaf39cfee33b25c7c1d87b",
)
logger.debug("Segmentation model weights retrieved.")

def normalize_frame(frame, window_center=None, window_width=None):
    if window_center is not None and window_width is not None:
        if not isinstance(window_center, (list, tuple)):
            window_center = [window_center]
        if not isinstance(window_width, (list, tuple)):
            window_width = [window_width]
        
        wc = window_center[0]
        ww = window_width[0]
        
        frame_min = wc - ww / 2
        frame_max = wc + ww / 2
        frame_normalized = np.clip(frame, frame_min, frame_max)
        frame_normalized = (frame_normalized - frame_min) / (frame_max - frame_min) * 255.0
    else:
        if np.max(frame) == np.min(frame):
            logger.warning("Frame contains identical values; normalization impossible.")
            return np.zeros_like(frame, dtype=np.uint8)
        frame_normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255.0
    
    return frame_normalized.astype(np.uint8)

@st.cache_data
def selectSlice(slice_ix, pixelArray, fileName):
    logger.debug(f"selectSlice called with slice_ix={slice_ix}, fileName={fileName}.")
    outputPath = "OutputPath"
    os.makedirs(outputPath, exist_ok=True)
    tifffile.imwrite(f"{outputPath}/{fileName}", pixelArray[slice_ix, :, :])
    logger.info(f"Selected frame {slice_ix} saved as {fileName} in {outputPath}.")
    st.session_state.btnSelectSlice = True
    logger.debug("Button state updated to True for btnSelectSlice.")

logger.debug("Creating dictionaries for DICOMs and PNGs.")

### NEW/CHANGED: Directories for each filetype
DicomFolder = "Dicoms"
PngFolder   = "Pngs"

exampleFiles = {}
### DICOM
dicomPaths = sorted(glob.glob(os.path.join(DicomFolder, "*.dcm")))
for path in dicomPaths:
    exampleFiles[os.path.basename(path)] = path

### PNG
pngPaths = sorted(glob.glob(os.path.join(PngFolder, "*.png")))
for path in pngPaths:
    exampleFiles[os.path.basename(path)] = path

if len(exampleFiles) == 0:
    logger.warning("No DICOM or PNG files found!")
else:
    logger.info(f"Found {len(exampleFiles)} files in total (DICOM and PNG).")

st.markdown("<h1 style='text-align: center;'>AngioPy Segmentation</h1>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center;'> Welcome to <b>AngioPy Segmentation</b>, "
    "an AI-driven, coronary angiography segmentation tool.</h1>",
    unsafe_allow_html=True
)
st.markdown("")

### The user chooses a file from one combined dictionary
selectedFile = st.sidebar.selectbox(
    "Select DICOM or PNG:",
    options = list(exampleFiles.keys()),
    key="fileSelectBox"
)
filePath = exampleFiles[selectedFile]
logger.info(f"User selected file: {filePath}")

stepOne = st.sidebar.expander("STEP ONE", True)
stepTwo = st.sidebar.expander("STEP TWO", True)

tab1, tab2 = st.tabs(["Segmentation", "Analysis"])

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:16px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

pixelArray = None
slice_ix = 0
n_slices = 1  # Default

### Detect if user selected .dcm or .png
ext = os.path.splitext(filePath)[1].lower()

if ext == ".dcm":
    # ---- Handle DICOM ----
    try:
        logger.info(f"Attempting to load DICOM: {filePath}")
        dcm = pydicom.dcmread(filePath, force=True)
        pixelArray = dcm.pixel_array
        logger.debug(f"Pixel array loaded with shape: {pixelArray.shape}")

        if len(pixelArray.shape) == 4:
            logger.debug("Detected a 4D pixel array. Taking first channel.")
            pixelArray = pixelArray[:, :, :, 0]

        n_slices = pixelArray.shape[0]
        logger.info(f"Number of frames (n_slices): {n_slices}")

        if 'WindowCenter' in dcm and 'WindowWidth' in dcm:
            window_center = dcm.WindowCenter
            window_width = dcm.WindowWidth
            logger.debug(f"WindowCenter: {window_center}, WindowWidth: {window_width}")
        else:
            window_center = None
            window_width = None

    except Exception as e:
        logger.error(f"Error reading DICOM file {filePath}: {e}")
        pixelArray = None
        n_slices = 0
else:
    # ---- Handle PNG ----
    logger.info(f"Attempting to load PNG: {filePath}")
    try:
        pngImage = Image.open(filePath).convert("L")  # Force grayscale
        pngArray = np.array(pngImage)
        # We treat it as [slice, height, width] with slice=1
        pixelArray = np.expand_dims(pngArray, axis=0)
        n_slices = 1

        # For PNG, no windowing
        window_center = None
        window_width = None
        logger.debug(f"PNG array shape: {pixelArray.shape}")
    except Exception as e:
        logger.error(f"Error reading PNG file {filePath}: {e}")
        pixelArray = None
        n_slices = 0

if pixelArray is not None and n_slices > 0:
    with tab1:
        with stepOne:
            st.write(
                "Select frame for annotation. "
                "Aim for an end-diastolic frame with good visualisation of the artery of interest."
            )
            # If there's more than 1 slice, show slider. For PNG, n_slices=1, so slider is trivial
            if n_slices > 1:
                slice_ix = st.slider(
                    'Frame',
                    0, n_slices - 1,
                    int(n_slices / 2) if n_slices > 1 else 0,
                    key='sliceSlider'
                )
            else: 
                slice_ix = 0
            logger.info(f"User selected slice index {slice_ix}.")
            predictedMask = numpy.zeros_like(pixelArray[slice_ix, :, :])

        with stepTwo:
            selectedArtery = st.selectbox(
                "Select artery for annotation:",
                ['LAD', 'CX', 'RCA', 'LM', 'OM', 'AM', 'D'],
                key="arteryDropMenu"
            )
            logger.info(f"Artery selected: {selectedArtery}.")
            st.write(
                "Beginning with the desired start point and finishing at the desired end point, "
                "click along the artery aiming for ~5-10 points."
            )

            stroke_color = angioPyFunctions.colourTableList[selectedArtery]
            logger.debug(f"Using stroke_color {stroke_color} for the selected artery.")

        col1, col2 = st.columns((15, 15))
        with col1:
            col1a, col1b, col1c = st.columns((1, 10, 1))
            with col1b:
                leftImageText = (
                    "<p style='text-align: center; color: white;'>"
                    "Beginning with the desired <u><b>start point</b></u> and "
                    "finishing at the desired <u><b>end point</b></u>, click along the "
                    "artery aiming for ~5-10 points. Segmentation is automatic.</p>"
                )

                st.markdown("<h5 style='text-align: center; color: white;'>Selected frame</h5>", unsafe_allow_html=True)
                st.markdown(leftImageText, unsafe_allow_html=True)

                selectedFrame = pixelArray[slice_ix, :, :]
                # Resize to 512x512 for the canvas
                selectedFrame = cv2.resize(selectedFrame, (512, 512))
                selectedFrame = normalize_frame(selectedFrame, window_center, window_width)

                annotationCanvas = st_canvas(
                    fill_color="red",
                    stroke_width=1,
                    stroke_color="red",
                    background_color='black',
                    background_image=Image.fromarray(selectedFrame),
                    update_streamlit=True,
                    height=512,
                    width=512,
                    drawing_mode="point",
                    point_display_radius=2,
                    key=str(st.session_state.fileSelectBox)+"annotation",
                )
                logger.debug("Canvas for annotation created.")

                if annotationCanvas.json_data is not None:
                    objects = pd.json_normalize(annotationCanvas.json_data["objects"])
                    if len(objects) != 0:
                        logger.info(f"{len(objects)} points detected on annotation canvas.")
                        for col in objects.select_dtypes(include=['object']).columns:
                            objects[col] = objects[col].astype("str")

                        logger.debug("Running artery segmentation with annotated points.")
                        predictedMask = angioPyFunctions.arterySegmentation(
                            slice_ix=slice_ix,
                            pixelArray=pixelArray,
                            groundTruthPoints=objects[['top', 'left']],
                            segmentationModel=segmentationModelWeights,
                            window_center=window_center,
                            window_width=window_width
                        )
                        logger.debug("Segmentation model returned a predicted mask.")

        with col2:
            col2a, col2b, col2c = st.columns((1, 10, 1))
            with col2b:
                st.markdown("<h5 style='text-align: center; color: white;'>Predicted mask</h1>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: center; color: white;'>"
                    "If the predicted mask has errors, restart and select more points "
                    "to help the segmentation model. </p>", 
                    unsafe_allow_html=True
                )

                stroke_color = "rgba(255, 255, 255, 255)"
                maskCanvas = st_canvas(
                    fill_color=angioPyFunctions.colourTableList[selectedArtery],
                    stroke_width=0,
                    stroke_color=stroke_color,
                    background_color='black',
                    background_image=Image.fromarray(predictedMask),
                    update_streamlit=True,
                    height=512,
                    width=512,
                    drawing_mode="freedraw",
                    point_display_radius=3,
                    key=str(st.session_state.fileSelectBox)+"mask",
                )
                logger.debug("Canvas for predicted mask displayed.")

                if np.sum(predictedMask) > 0 and 'objects' in locals() and len(objects) > 4:
                    logger.info("Predicted mask is non-empty and sufficient points were annotated.")
                    b_channel, g_channel, r_channel = cv2.split(predictedMask)
                    a_channel = np.full_like(predictedMask[:, :, 0], fill_value=255)
                    predictedMaskRGBA = cv2.merge((predictedMask, a_channel))
                    logger.debug("Merged predicted mask into RGBA format.")

                    with tab2:
                        tab2Col1, tab2Col2 = st.columns([20, 10])
                        with tab2Col1:
                            st.markdown("<h5 style='text-align: center; color: white;'><br>Artery profile</h5>", unsafe_allow_html=True)
                            EDT = scipy.ndimage.distance_transform_edt(
                                cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2GRAY)
                            )
                            logger.debug("Computed distance transform EDT of predicted mask.")

                            skel = angioPyFunctions.skeletonise(predictedMaskRGBA)
                            tck = angioPyFunctions.skelSplinerWithThickness(skel=skel, EDT=EDT)
                            logger.debug("Skeleton and spline computed for centerline thickness.")

                            splinePointsY, splinePointsX, splineThicknesses = scipy.interpolate.splev(
                                np.linspace(0.0, 1.0, 1000),
                                tck
                            )
                            logger.debug("Spline evaluated at 1000 points.")

                            clippingLength = 20
                            vesselThicknesses = splineThicknesses[clippingLength:-clippingLength] * 2

                            fig = px.line(
                                x=np.arange(1, len(vesselThicknesses) + 1),
                                y=vesselThicknesses,
                                labels=dict(x="Centreline point", y="Thickness (pixels)"),
                                width=800
                            )
                            fig.update_traces(line_color='rgb(31, 119, 180)', line={'width': 4})
                            fig.update_xaxes(showline=True, linewidth=2, linecolor='white', showgrid=False)
                            fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='white')
                            fig.update_layout(yaxis_range=[0, np.max(vesselThicknesses) * 1.2])
                            fig.update_layout(font_color="white", title_font_color="white")
                            fig.update_layout({
                                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                            })

                            selected_points = plotly_events(fig)
                            logger.debug("Displayed thickness plot via Plotly.")

                        with tab2Col2:
                            st.markdown("<h5 style='text-align: center; color: white;'><br>Contours</h5>", unsafe_allow_html=True)
                            selectedFrameRGBA = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGBA)

                            contour = angioPyFunctions.maskOutliner(
                                labelledArtery=predictedMaskRGBA[:, :, 0],
                                outlineThickness=1
                            )
                            selectedFrameRGBA[contour, :] = [
                                angioPyFunctions.colourTableList[selectedArtery][2],
                                angioPyFunctions.colourTableList[selectedArtery][1],
                                angioPyFunctions.colourTableList[selectedArtery][0],
                                255
                            ]

                            fig2 = px.imshow(selectedFrameRGBA)
                            fig2.update_xaxes(visible=False)
                            fig2.update_yaxes(visible=False)
                            fig2.update_layout(margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0})
                            fig2
