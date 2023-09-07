import os
import cv2
import numpy as np
import torch
from model import build_unet  # Assuming you have the model architecture defined in "model.py"

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

def perform_segmentation(image_path, output_path, model_checkpoint):
    # Load the model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess the input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image from path: {input_image_path}")

    image = cv2.resize(image, (512, 512))
    x = np.transpose(image, (2, 0, 1))
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

    # Visualize and save the results
    segmented_mask = mask_parse(pred_y)
    segmented_image = cv2.addWeighted(image.astype(np.uint8), 0.6, segmented_mask.astype(np.uint8), 0.4, 0)

    cv2.imwrite(output_path, segmented_image)

if __name__ == "__main__":
    input_image_path = "inference/15_test_0.png"  # Replace with the actual path of the input image
    output_image_path = "inference/segmented_image.png"  # Path to save the segmented output image
    model_checkpoint_path = "files/checkpoint.pth"  # Replace with the actual path of the model checkpoint

    perform_segmentation(input_image_path, output_image_path, model_checkpoint_path)
