# AngioPy Segmentation

## Online Example
Please visit https://imaging.epfl.ch/angiopy-segmentation/ for a live demo of this code on some example DICOM images

![](illustration.mp4)

## Description
This software allows single arteries to be segmented given a few clicks on a single time frame with a PyTorch 2 Deep Learning model.

## Installing and running
 - Install dependencies: ` pip install -r requirements.txt`
 - Launch Streamlit Web Interface: `streamlit run interface.py --server.fileWatcherType none`

 ...a website should pop up in your browser!

 You need to create a /Dicom folder and put some angiography DICOMs in there
