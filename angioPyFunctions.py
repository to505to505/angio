import numpy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.interpolate
import skimage.filters
import skimage.morphology
import scipy.ndimage
import scipy.optimize
import predict
from PIL import Image
from fil_finder import FilFinder2D
import astropy.units as u
from tqdm import tqdm
import streamlit as st

# ----------------- SEPARATE LOGGER SETUP -----------------
import logging

logger = logging.getLogger("angioPyFunctionsLogger")
logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if this file is re-imported
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.debug("angioPyFunctions module loaded and logger initialized.")
# ---------------------------------------------------------

colourTableHex = {
    'LAD': "#f03b20",
    'D':   "#fd8d3c",
    'CX':  "#31a354",
    'OM':  "#74c476",
    'RCA': "#08519c",
    'AM':  "#3182bd",
    'LM':  "#984ea3",
}

colourTableList = {}
for item in colourTableHex.keys():
    # Convert hex color to a list [B,G,R]
    # (Note: reversed because of how cv2 uses color ordering)
    colourTableList[item] = [
        int(colourTableHex[item][5:7], 16),  # Blue
        int(colourTableHex[item][3:5], 16),  # Green
        int(colourTableHex[item][1:3], 16)   # Red
    ]

logger.debug(f"Initialized colourTableList with {len(colourTableList)} items.")


def skeletonise(maskArray):
    """
    Takes a 3-channel or single-channel maskArray (e.g. RGBA) and returns a skeleton (2D).
    """
    logger.debug("Entered skeletonise()")

    # Convert to grayscale if it's multi-channel
    logger.debug(f"Original maskArray shape: {maskArray.shape}, dtype: {maskArray.dtype}")
    maskGray = cv2.cvtColor(maskArray, cv2.COLOR_BGR2GRAY)
    logger.debug(f"Converted to grayscale. New shape: {maskGray.shape}, dtype: {maskGray.dtype}")

    # Skeletonize
    skeleton = skimage.morphology.skeletonize(maskGray.astype('bool'))
    logger.debug("Skeletonization done.")

    # Use FilFinder2D to find and prune the main skeleton
    fil = FilFinder2D(
        skeleton.astype('uint8'),
        distance=250 * u.pc,
        mask=skeleton,
        beamwidth=10.0 * u.pix
    )
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(
        branch_thresh=400 * u.pix,
        skel_thresh=10 * u.pix,
        prune_criteria='length'
    )
    logger.debug("FilFinder2D skeleton analysis complete.")

    # fil.skeleton is a processed skeleton
    skel = fil.skeleton.astype('<u1') * 255
    logger.debug("Returning final skeleton array from skeletonise().")

    return skel


def skelEndpoints(skel):
    """
    Find endpoints of a skeleton (start and end).
    """
    logger.debug("Entered skelEndpoints()")

    skel = numpy.uint8(skel > 0)
    logger.debug(f"Binary skeleton shape: {skel.shape}")

    # Convolve with kernel to detect endpoints
    kernel = numpy.uint8([
        [1,  1, 1],
        [1, 10, 1],
        [1,  1, 1]
    ])
    filtered = cv2.filter2D(skel, -1, kernel)
    out = numpy.zeros_like(skel)
    out[numpy.where(filtered == 11)] = 1

    endCoords = numpy.where(filtered == 11)
    endCoords = list(zip(*endCoords))

    # Just a naive approach to picking the first two
    if len(endCoords) < 2:
        logger.warning("Less than 2 endpoints found. Skeleton may not have clear endpoints.")
        startPoint = (0, 0)
        endPoint = (0, 0)
    else:
        startPoint = endCoords[0]
        endPoint = endCoords[-1]

    logger.debug(f"Endpoints found: start={startPoint}, end={endPoint}")
    return startPoint, endPoint


def skelPointsInOrder(skel, startPoint=None):
    """
    Return all points of the skeleton in an ordered path, starting from 'startPoint'
    if provided. If not, the function automatically finds endpoints.
    """
    logger.debug("Entered skelPointsInOrder()")

    if startPoint is None:
        startPoint, _ = skelEndpoints(skel)
        logger.debug(f"No startPoint provided, using detected endpoint: {startPoint}")

    # Gather all skeleton coordinates
    skelXY = numpy.array(numpy.where(skel))
    skelPoints = list(zip(skelXY[0], skelXY[1]))
    logger.debug(f"Number of skeleton points found: {len(skelPoints)}")

    startPointCopy = startPoint
    orderedPoints = []

    while len(skelPoints) > 1:
        # Remove the 'current' point from the available pool
        skelPoints.remove(startPointCopy)

        # Find the closest point (by L1 distance)
        diffs = numpy.abs(numpy.array(skelPoints) - numpy.array(startPointCopy))
        dists = numpy.sum(diffs, axis=1)
        closest_point_index = numpy.argmin(dists)
        closestPoint = skelPoints[closest_point_index]
        orderedPoints.append(closestPoint)

        startPointCopy = closestPoint

    orderedPoints = numpy.array(orderedPoints)
    logger.debug(f"Finished ordering points. Total ordered points: {len(orderedPoints)}")

    return orderedPoints


def skelSplinerWithThickness(skel, EDT, smoothing=50, order=3, decimation=2):
    """
    Spline fit the skeleton, using the distance transform (EDT) as thickness data.
    Returns the interpolation 'tck' from splprep.
    """
    logger.debug("Entered skelSplinerWithThickness()")

    startPoint, endPoint = skelEndpoints(skel)
    logger.debug(f"startPoint={startPoint}, endPoint={endPoint}")

    orderedPoints = skelPointsInOrder(skel, startPoint)
    logger.debug(f"orderedPoints shape: {orderedPoints.shape}")

    # Unzip ordered points to x,y arrays
    x = orderedPoints[:, 1].ravel()
    y = orderedPoints[:, 0].ravel()

    # Decimate
    x = x[::decimation]
    y = y[::decimation]

    t = EDT[y, x]
    logger.debug("Gathered thickness values from EDT for each skeleton point.")

    # Spline can't handle the very last out-of-bounds if lengths mismatch
    x = x[0:-1]
    y = y[0:-1]
    t = t[0:-1]

    logger.debug(f"Spline fitting arrays: x.shape={x.shape}, y.shape={y.shape}, t.shape={t.shape}")

    tcko, uo = scipy.interpolate.splprep([y, x, t], s=smoothing, k=order, per=False)
    logger.debug("Completed splprep to obtain TCK spline parameters.")
    return tcko
    
def normalize_frame(frame, window_center=None, window_width=None):
    # Если есть параметры VOI LUT, используем их
    if window_center is not None and window_width is not None:
        # Преобразуем параметры в список, если они скалярные
        if not isinstance(window_center, (list, tuple)):
            window_center = [window_center]
        if not isinstance(window_width, (list, tuple)):
            window_width = [window_width]
        
        # Берем первые значения (для простоты)
        wc = window_center[0]
        ww = window_width[0]
        
        # Применяем оконное преобразование вручную
        frame_min = wc - ww / 2
        frame_max = wc + ww / 2
        frame_normalized = np.clip(frame, frame_min, frame_max)
        frame_normalized = (frame_normalized - frame_min) / (frame_max - frame_min) * 255.0
    else:
        # Ручная нормализация на основе значений кадра
        if np.max(frame) == np.min(frame):
            logger.warning("Кадр содержит одинаковые значения, нормализация невозможна")
            return np.zeros_like(frame, dtype=np.uint8)
        frame_normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255.0
    
    return frame_normalized.astype(np.uint8)

@st.cache_resource 
def arterySegmentation(slice_ix, pixelArray, groundTruthPoints, segmentationModel, window_center, window_width):
    """
    Run the segmentation model using groundTruthPoints to guide the model's input channels.
    Returns a predicted mask (numpy array).
    """
    logger.debug("Entered arterySegmentation()")
    logger.info(f"Segmenting slice {slice_ix} with {len(groundTruthPoints)} groundTruthPoints...")

    inputImage = pixelArray[slice_ix, :, :]

    inputImage = cv2.resize(inputImage, (512, 512))
    inputImage = normalize_frame(inputImage, window_center=window_center, window_width=window_width)

    imageSize = inputImage.shape
    logger.debug(f"Image resized to {imageSize}.")

    # Convert user 'top','left' columns into coordinate tuples
    groundTruthPoints = list(zip(groundTruthPoints['top'], groundTruthPoints['left'] + 3.5))
    logger.debug(f"Reformatted groundTruthPoints. Total points: {len(groundTruthPoints)}")

    n_classes = 2
    net = predict.smp.Unet(
        encoder_name='inceptionresnetv2',
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes
    )

    net = predict.nn.DataParallel(net)
    device = predict.torch.device('cuda' if predict.torch.cuda.is_available() else 'cpu')
    logger.debug(f"Using device: {device}")
    net.to(device=device)

    predict.cudnn.benchmark = True
    net.load_state_dict(predict.torch.load(segmentationModel, map_location=device))
    logger.debug("Segmentation model loaded.")

    orig_image = Image.fromarray(inputImage)

    # Create an RGB image from the original grayscale
    image = predict.Image.new('RGB', imageSize, (0, 0, 0))
    image.paste(orig_image, (0, 0))

    imageArray = numpy.array(image).astype('uint8')

    # Clear last two channels
    imageArray[:, :, -1] = 0
    imageArray[:, :, -2] = 0

    # Mark endpoints distinctly
    startPoint = groundTruthPoints[0]
    endPoint = groundTruthPoints[-1]

    for point in [startPoint, endPoint]:
        y = int(point[0])
        x = int(point[1])
        imageArray[y - 2:y + 2, x - 2:x + 2, -2] = 255

    # Mark other points in the third channel
    for point in groundTruthPoints[2:-1]:
        y = int(point[0])
        x = int(point[1])
        imageArray[y - 2:y + 2, x - 2:x + 2, -1] = 255

    logger.debug("Created model input with annotated channels for groundTruthPoints.")

    image = Image.fromarray(imageArray.astype(numpy.uint8))
    mask = predict.predict_img(
        net=net,
        dataset_class=predict.CoronaryDataset,
        full_img=image,
        scale_factor=1,
        device=device
    )

    logger.debug("Model inference completed.")
    result = predict.CoronaryDataset.mask2image(mask)
    result = result.crop((0, 0, imageSize[0], imageSize[1]))
    resultsArray = numpy.asarray(result)

    logger.debug(f"Returning segmentation mask array with shape {resultsArray.shape}.")
    return resultsArray


def maskOutliner(labelledArtery, outlineThickness=3):
    """
    Finds the contour of a binary mask and returns a boolean array
    indicating where the outline is.
    """
    logger.debug("Entered maskOutliner()")

    contours, _ = cv2.findContours(labelledArtery, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tmp = numpy.zeros_like(labelledArtery)

    boundary = cv2.drawContours(tmp, contours, -1, (255, 255, 255), outlineThickness)
    boundary = boundary > 0

    logger.debug("Contours drawn, returning boolean outline array.")
    return boundary
