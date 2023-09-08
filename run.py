import streamlit as st
import os
import cv2
import numpy as np
import torch
from model import build_unet  # Import your U-Net model architecture
from PIL import Image
from utils import set_background

# Function to perform segmentation
def perform_segmentation(uploaded_image):

     # Construct the file path where the uploaded image is saved
    file_path = f"inference/{uploaded_image.name}"  # You can specify the directory where you want to save the image

    # Save the uploaded image to the specified file path
    with open(file_path, "wb") as f:
        f.write(uploaded_image.getvalue())

    # Load the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Load the model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model_checkpoint_path = "files/checkpoint.pth"  # Replace with your model checkpoint path
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Convert the uploaded image to a numpy array
    image_np = np.array(image)

    # Resize and preprocess the input image
    image_np = cv2.resize(image_np, (512, 512))
    x = np.transpose(image_np, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x).to(device)

    # Perform inference
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5

    # Create a blank mask with three channels
    blank_mask = np.zeros((512, 512, 3), dtype=np.uint8)

    # Copy the predicted mask into one of the channels
    blank_mask[:, :, 0] = (pred_y * 255).astype(np.uint8)  # Assuming the predicted mask is binary

    # Visualize and return the segmented result
    # segmented_image = cv2.addWeighted(image_np, 0.6, pred_y.astype(np.uint8), 0.4, 0)
    
    # Visualize and return the segmented result
    segmented_image = cv2.addWeighted(image_np, 0.6, blank_mask, 0.4, 0)

    # Ensure that pred_y has the same dimensions as image_np
    if image_np.shape != segmented_image.shape:
        print("image shape",image_np.shape, "pred image shape: ",pred_y.shape)
        st.warning("Warning: Image and segmented mask dimensions do not match.")
        return image_np  # Return the original image without segmentation


    return segmented_image


# Main Streamlit app
# def main():

#     set_background('presentation/cover1.png')
#     # Header
#     # st.markdown("<h1 style='text-align: center;'>Retina Blood Vessel Segmentation App</h1>", unsafe_allow_html=True)

#     # Title and description
#     with open('web/style.css') as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#         st.markdown("<h1 style='text-align: center;'>RetinaVision: Precise Vessel Segmentation</h1>", unsafe_allow_html=True)
#     # st.title("Image Segmentation App")
#     st.write("Upload an image and see the segmented result!")

#     # Upload image
#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

#     if uploaded_image is not None:
#         # Display the uploaded image
#         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#         # Perform segmentation on the uploaded image
#         segmented_image = perform_segmentation(uploaded_image)

#         # Display the segmented image
#         st.image(segmented_image, caption="Segmented Image", use_column_width=True)


# Main Streamlit app
def main():

    set_background('presentation/cover1.png')
    
    ## Title and description
    with open('web/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>RetinaVision: Precise Vessel Segmentation</h1>", unsafe_allow_html=True)
    # st.title("Image Segmentation App")
    st.write("Upload an image and see the segmented result!")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg",".tif"])

    if uploaded_image is not None:

        # Display the uploaded image
        st.image(uploaded_image, use_column_width=True)
        st.markdown("<h2 style='text-align: center; color: red;'><b>Uploaded Image</b></h2>", unsafe_allow_html=True)

        # Perform segmentation on the uploaded image
        segmented_image = perform_segmentation(uploaded_image)

        # Create two columns for original and segmented images
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(uploaded_image,  use_column_width=True)
            st.markdown("<h2 style='text-align: center; color: red;'><b>Original Image</b></h2>", unsafe_allow_html=True)


        # Display the segmented image in the second column
        with col2:
            st.image(segmented_image, use_column_width=True)
            st.markdown("<h2 style='text-align: center; color: red;'><b>Segmented Image</b></h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
