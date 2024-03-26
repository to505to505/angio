import os
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
# from streamlit_image_coordinates import streamlit_image_coordinates
import predict
import angioPyFunctions
import scipy
import cv2

st.set_page_config(layout="wide")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     background-color: #0099ff;
#     color:#ffffff;
# }
# div.stButton > button:hover {
#     background-color: #00ff00;
#     color:#ff0000;
#     }
# </style>""", unsafe_allow_html=True)

outputPath = "Output"

segmentationModelWeights = "SegmentationModel/modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth"
# segmentationModelWeights = "SegmentationModel/modelWeights-FAMEallArteries-inceptionresnetv2-fold1-e40-b10.pth"

# Make output folder
os.makedirs(name=outputPath, exist_ok=True)

# arteryDictionary = {
#     'LAD':       {'colour': "#f03b20"},
#     'CX':        {'colour': "#31a354"},
#     'OM':    {'colour' : "#74c476"},
#     'RCA':       {'colour': "#08519c"},
#     'AM':   {'colour' : "#3182bd"},
#     'LM':        {'colour' : "#984ea3"},
# }

def file_selector(folder_path='.'):
    fileNames = [file for file in glob.glob(f"{folder_path}/*")]
    selectedDicom = st.sidebar.selectbox('Select a DICOM file:', fileNames)
    if selectedDicom is None:
        return None
    
    return selectedDicom


def selectSlice(slice_ix, pixelArray, fileName):

    # Save the selected frame 
    tifffile.imwrite(f"{outputPath}/{fileName}", pixelArray[slice_ix, :, :])

    # Set the button as clicked
    st.session_state.btnSelectSlice = True


# def reset():

#     st.session_state.sliceSlider = 1

def set_stage(stage):
    st.session_state.stage = stage

# def segment():
    
#     if annotationCanvas.json_data is not None:

#         objects = pd.json_normalize(annotationCanvas.json_data["objects"]) # need to convert obj to str because PyArrow

#         if len(objects) != 0:

#             for col in objects.select_dtypes(include=['object']).columns:
#                 objects[col] = objects[col].astype("str")

            
#             # Run segmentation model on the selected from, and the chosen groundtruth points
#             predictedMask = angioPyFunctions.arterySegmentation(slice_ix=slice_ix, pixelArray=pixelArray, groundTruthPoints = objects[['top', 'left']], segmentationModel=segmentationModelWeights)
            
#             # Save the predicted mask
#             tifffile.imwrite(f"{outputPath}/mask.tif", predictedMask)



# Main text
st.markdown("<h1 style='text-align: center; color: white;'>AngioPy segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: white;'> Welcome to <b>AngioPy segmentation</b>, an AI-driven, coronary angiography segmentation tool.</h1>", unsafe_allow_html=True)
st.markdown("")

# Build the sidebar
selectedDicom = st.sidebar.file_uploader("Upload DICOM file:",type=["dcm"], accept_multiple_files=False)

st.sidebar.expander("STEP ONE", expanded =False)

stepOne = st.sidebar.expander("STEP ONE", True)
stepTwo = st.sidebar.expander("STEP TWO", True)
stepThree = st.sidebar.expander("STEP THREE", True)
stepFour = st.sidebar.expander("STEP FOUR", True)

# slicer = st.expander("", True)
# segmenter = st.expander("", True)

# annotationExpander = st.expander("")
# Create tabs 
tab1, tab2 = st.tabs(["Artery segmentation", "Artery analysis"])

# Increase tab font size
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:16px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)



# Once a file is uploaded, the following annotation sequence is initiated 
if selectedDicom is not None:
    try:
        
        dcm = pydicom.dcmread(selectedDicom, force=True)

        handAngle = dcm.PositionerPrimaryAngle
        headAngle = dcm.PositionerSecondaryAngle
        # dcmLabel = f"{'LAO' if handAngle > 0 else 'RAO'} {numpy.abs(handAngle):04.1f}° {'CRA' if headAngle > 0 else 'CAU'} {numpy.abs(headAngle):04.1f}°"

        pixelArray = dcm.pixel_array

        n_slices = pixelArray.shape[0]

        slice_ix = 0

        with tab1:

            with stepOne:
                st.write("Select frame for annotation. Aim for an end-diastolic frame with good visualisation of the artery of interest.")
            
                slice_ix = st.slider('Frame', 0, n_slices, int(n_slices/2), key='sliceSlider')

                # stepOneCol1, stepOneCol2 = st.columns((3,3))

                # with stepOneCol1:

                #     btnSelectSlice = st.button("SAVE FRAME", on_click=selectSlice(slice_ix=slice_ix, pixelArray=pixelArray, fileName=f"{os.path.splitext(selectedDicom.name)[0]}-{slice_ix}.tif"), use_container_width=True)
                
                # with stepOneCol2:

                #     btnReset = st.button("RESET", use_container_width=True)



                predictedMask = numpy.zeros_like(pixelArray[slice_ix, :, :])
                # combinedMaskRGBA = cv2.cvtColor(predictedMask, cv2.COLOR_GRAY2RGBA)                
                # a_channel = numpy.full_like(predictedMask[:,:,0], fill_value=255)
                # combinedMaskRGBA = cv2.merge((predictedMask, a_channel))


            # if st.session_state.btnSelectSlice == True:

            with stepTwo:

                selectedArtery = st.selectbox("Select artery for annotation:",
                        ['LAD', 'CX', 'RCA', 'LM', 'OM', 'AM', 'D'],
                        key="arteryDropMenu"
                    )

                st.write("Beginning with the desired start point and finishing at the desired end point, click along the artery aiming for ~5-10 points, then click SEGMENT")


                
                # stepTwoCol1, stepTwoCol2, stepTwoCol3 = st.columns((1,3,1))

                # with stepTwoCol2:

                #     btnSegment = st.button("SAVE MASK", on_click=tifffile.imwrite(f"{outputPath}/{selectedArtery}.tif", predictedMask), args=[1], use_container_width=True)
                
                # with stepTwoCol2:
                #     btnSave = st.button("SAVE MASK", on_click=selectSlice(slice_ix=slice_ix, pixelArray=pixelArray, fileName="selectedFrame.tif"), use_container_width=True)
            
                # drawing_mode = st.selectbox(
                #     "Mask correction:", ("Erase", "Draw")
                # )

                # stroke_width = 3
                stroke_color = angioPyFunctions.colourTableList[selectedArtery]


            # st.markdown(f"<h5 style='text-align: center; color: white;'>Acquisition angle<br>{dcmLabel}</h5>", unsafe_allow_html=True)

            col1, col2 = st.columns((15,15))

            with col1:

                col1a, col1b, col1c = st.columns((1,10,1))

                with col1b:

                    leftImageText = "<p style='text-align: center; color: white;'>Beginning with the desired <u><b>start point</b></u> and finishing at the desired <u><b>end point</b></u>, click along the artery aiming for ~5-10 points, then click SEGMENT</p>"

                    st.markdown(f"<h5 style='text-align: center; color: white;'>Selected frame</h5>", unsafe_allow_html=True)

                    st.markdown(leftImageText, unsafe_allow_html=True)

                    selectedFrame = pixelArray[slice_ix, :, :]
                    selectedFrame = cv2.resize(selectedFrame, (512,512))

                    # Create a canvas component
                    annotationCanvas = st_canvas(
                        fill_color="red",  # Fixed fill color with some opacity
                        stroke_width=1,
                        stroke_color="red",
                        background_color='black',
                        background_image= Image.fromarray(selectedFrame),
                        update_streamlit=True,
                        height=512,
                        width=512,
                        drawing_mode="point",
                        point_display_radius=2,
                        key="annotationCanvas",
                    )


                    # Do something interesting with the image data and paths
                    # if st.session_state.stage == 0:
                    if annotationCanvas.json_data is not None:
                        objects = pd.json_normalize(annotationCanvas.json_data["objects"]) # need to convert obj to str because PyArrow

                        if len(objects) != 0:

                            for col in objects.select_dtypes(include=['object']).columns:
                                objects[col] = objects[col].astype("str")

                                # ys = numpy.array(objects['top'])
                                # xs = numpy.array(objects['left'])

                            
                            # Run segmentation model on the selected from, and the chosen groundtruth points
                            predictedMask = angioPyFunctions.arterySegmentation(slice_ix=slice_ix, pixelArray=pixelArray, groundTruthPoints = objects[['top', 'left']], segmentationModel=segmentationModelWeights)
                            
                            # Save the predicted mask
                            # tifffile.imwrite(f"{outputPath}/mask.tif", predictedMask)


        
            with col2:  

                col2a, col2b, col2c = st.columns((1,10,1))

                with col2b:

                    st.markdown(f"<h5 style='text-align: center; color: white;'>Predicted mask</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: white;'>If the predicted mask has errors, restart and select more points to help the segmentation model. </p>", unsafe_allow_html=True)


                    # Define correct colour for canvas (erase vs drawing)
                    # if drawing_mode == "Erase":
                    #     stroke_color = "rgba(0, 0, 0, 255)"
                    # else:
                    #     stroke_color = "rgba(255, 255, 255, 255)"

                    stroke_color = "rgba(255, 255, 255, 255)"


                    maskCanvas = st_canvas(
                        fill_color=angioPyFunctions.colourTableList[selectedArtery],  # Fixed fill color with some opacity
                        stroke_width=0,
                        stroke_color=stroke_color,
                        background_color='black',
                        background_image= Image.fromarray(predictedMask),
                        update_streamlit=True,
                        height=512,
                        width=512,
                        drawing_mode="freedraw",
                        point_display_radius=3,
                        key="maskCanvas",
                    )


                    # Check that the mask array is not blank
                    if numpy.sum(predictedMask) > 0 and len(objects)>4:

                        # add alpha channel to predict mask in order to merge
                        b_channel, g_channel, r_channel = cv2.split(predictedMask)
                        a_channel = numpy.full_like(predictedMask[:,:,0], fill_value=255)

                        predictedMaskRGBA = cv2.merge((predictedMask, a_channel))

                        # Combine the predictedMask with any changes made
                        # erasedCanvas = numpy.zeros_like(maskCanvas.image_data[:,:,0])


                        # for ch in [0,1,2,3]:
                        #     print(numpy.sum(maskCanvas.image_data[:,:,ch]))

                        # combinedMaskRGBA = numpy.where((maskCanvas.image_data > 0), maskCanvas.image_data, predictedMaskRGBA)
                        # combinedMaskRGBA = numpy.where((predictedMask > 0), predictedMask, predictedMaskRGBA)

                        
                        # Save combined mask
                        # tifffile.imwrite(f"{outputPath}/combinedMaskRGBA.tif", predictedMaskRGBA)

                        # Drop the alpha channel for subsequent processing
                        # combinedMask = combinedMaskRGBA[:,:,3]
                        # tifffile.imwrite(f"{outputPath}/combinedMaskRGB.tif", combinedMask)

                        # with tab2:

                        #     # Create a canvas component
                        #     catheterAnnotationCanvas = st_canvas(
                        #         fill_color=arteryDictionary[selectedArtery]['colour'],  # Fixed fill color with some opacity
                        #         stroke_width=1,
                        #         stroke_color=arteryDictionary[selectedArtery]['colour'],
                        #         background_color='black',
                        #         background_image= Image.fromarray(selectedFrame),
                        #         update_streamlit=True,
                        #         height=512,
                        #         width=512,
                        #         drawing_mode="point",
                        #         point_display_radius=2,
                        #         key="catheterAnnotationCanvas",
                        #     )


                        #     # Do something interesting with the image data and paths
                        #     if annotationCanvas.json_data is not None:
                        #         objects = pd.json_normalize(annotationCanvas.json_data["objects"]) # need to convert obj to str because PyArrow

                        #         if len(objects) != 0:

                        #             for col in objects.select_dtypes(include=['object']).columns:
                        #                 objects[col] = objects[col].astype("str")

                        #                 # ys = numpy.array(objects['top'])
                        #                 # xs = numpy.array(objects['left'])

                                    
                        #             # Run segmentation model on the selected from, and the chosen groundtruth points
                        #             predictedCatheterMask = angioPyFunctions.arterySegmentation(slice_ix=slice_ix, pixelArray=pixelArray, groundTruthPoints = objects[['top', 'left']], segmentationModel=segmentationModelWeights)
                                    
                        #             # Save the predicted mask
                        #             tifffile.imwrite(f"{outputPath}/catheterMask.tif", predictedCatheterMask)

                        # with stepThree:

                        #     selectedArtery = st.selectbox("Select catheter size:", ['Select catheter...', '4Fr', '5Fr', '6Fr'], key="catheterDropMenu")

                        #     if selectedArtery != 'Select catheter...':

                        #         #  leftImageExplanation = "sdfsdfdsfdsf"

                        #         #  test = st.markdown(f"{leftImageExplanation}", unsafe_allow_html=True)
                        #         leftImageTextContainer.text("sdfsfdsfsdf")



                        with tab2:

                            # combinedMask = cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2RGB)

                            # print(combinedMask.shape)
                            # tifffile.imwrite(f"{outputPath}/test.tif", combinedMask)


                            # tab2Col1, tab2Col2, tab2Col3 = st.columns([1,15,1])
                            tab2Col1, tab2Col2 = st.columns([20,10])

                            with tab2Col1:

                                st.markdown(f"<h5 style='text-align: center; color: white;'><br>Artery profile</h5>", unsafe_allow_html=True)
                                
                                # Extract thickness information from mask
                                # EDT = scipy.ndimage.distance_transform_edt(cv2.cvtColor(maskCanvas.image_data, cv2.COLOR_BGR2GRAY))
                                EDT = scipy.ndimage.distance_transform_edt(cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2GRAY))

                                # Skeletonise, get a list of ordered centreline points, and spline them
                                skel = angioPyFunctions.skeletonise(predictedMaskRGBA)
                                tck = angioPyFunctions.skelSplinerWithThickness(skel=skel, EDT=EDT)
                                
                                # Save the centreline
                                # tifffile.imwrite(f"{outputPath}/centreline.tif", skel)

                                # Interogate the spline function over 1000 points
                                splinePointsY, splinePointsX, splineThicknesses = scipy.interpolate.splev(
                                numpy.linspace(
                                    0.0,
                                    1.0,
                                    1000), 
                                    tck)
                                
                                clippingLength = 20
                                
                                vesselThicknesses = splineThicknesses[clippingLength:-clippingLength]*2

                                fig = px.line(x=numpy.arange(1,len(vesselThicknesses)+1),y=vesselThicknesses, labels=dict(x="Centreline point", y="Thickness (pixels)"), width=800)
                                # fig.update_layout(showlegend=False, xaxis={'showgrid': False, 'zeroline': True})
                                fig.update_traces(line_color='rgb(31, 119, 180)', textfont_color="white", line={'width':4})
                                fig.update_xaxes(showline=True, linewidth=2, linecolor='white', showgrid=False,gridcolor='white')
                                fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='white')
                                
                                
                                fig.update_layout(font_color="white",title_font_color="white")
                                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
                                
                                
                                selected_points = plotly_events(fig)


                                # if len(selected_points) > 0:

                                #     index = selected_points[0]['x']
                                #     x = splinePointsX[index]
                                #     y = splinePointsY[index]
                                    
                                #     # plot = st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit", margin=dict(l=0, r=0, t=0, b=0))

                                #     # fig2, ax = plt.subplots()

                                #     # ax.imshow(pixelArray[slice_ix,:,:])
                                #     # ax.scatter(x,y)


                                #     img = px.imshow(pixelArray[slice_ix,:,:], height=600, width=600, aspect = 'equal', binary_string=True, labels=dict(x="Width (pixels)", y="Height (pixels)", color="Pixel intensity"))

                                #     # img = go.Figure()
                                #     # img = go.Figure(data=go.Image(pixelArray[slice_ix, :,:]), layout={'height': pixelArray[1], 'width': pixelArray[2]})

                                #     img.add_trace(go.Scatter(x=[x], y=[y], mode='markers'))
                                #     # img.add_layout_image(dict(source=Image.fromarray(pixelArray[slice_ix, :,:])))

                                #     plot = st.plotly_chart(img, use_container_width=False, sharing="streamlit", theme="streamlit", margin=dict(l=0, r=0, t=0, b=0))

                                
                            with tab2Col2:

                                st.markdown(f"<h5 style='text-align: center; color: white;'><br>Contours</h5>", unsafe_allow_html=True)


                                selectedFrameRGBA = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGBA)

                                # imgOverlayRGBA = numpy.zeros_like(predictedMaskRGBA)
                                contour = angioPyFunctions.maskOutliner(labelledArtery=predictedMaskRGBA[:,:,0], outlineThickness=1)

                                selectedFrameRGBA[contour, :] =    [angioPyFunctions.colourTableList[selectedArtery][2],
                                                                    angioPyFunctions.colourTableList[selectedArtery][1],
                                                                    angioPyFunctions.colourTableList[selectedArtery][0],
                                                                    255]
                                
                                fig2 = px.imshow(selectedFrameRGBA)


                                fig2.update_xaxes(visible=False)
                                fig2.update_yaxes(visible=False)
                                fig2.update_layout(margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0},) #remove margins
                                # fig2.coloraxis(visible=False)

                                fig2.update_traces(dict(
                                    showscale=False, 
                                    coloraxis=None, 
                                    colorscale='gray'), selector={'type':'heatmap'})
                                
                                fig2.add_trace(go.Scatter(x=splinePointsX[clippingLength:-clippingLength], y=splinePointsY[clippingLength:-clippingLength], line=dict(width=1)))

                                st.plotly_chart(fig2, use_container_width=True)




    except RuntimeError:
        
        st.text('This does not look like a DICOM folder!')