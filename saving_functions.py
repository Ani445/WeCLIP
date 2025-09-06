import cv2
import numpy as np
import torch
import torch.nn.functional as F
import re
import os

def save_some_cams(cam, annotation_path, cam_idx, threshold = 0.5):
    annotation_path = annotation_path.replace("\\", "/")
    image_name = re.sub(
        r".*/VOC2012/SegmentationClassAug/(.*).png", r"\1", annotation_path
    )
    jpeg_image_path = re.sub(
        r"(.*/VOC2012/).*", r"\1JPEGImages/" + image_name + ".jpg", annotation_path
    )
    original_image = cv2.imread(jpeg_image_path, cv2.IMREAD_COLOR)

    # with open(f'./imgs.txt', 'a') as f:
    #     f.write(image_name + '\n')

    # if image_name in selected_image_names:
    # if True:
        # Ensure cam is in the correct format

        # with open(f'./imgs.txt', 'a') as f:
        #     f.write(str(cam))

    cam = (cam * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # Save heatmap over original image
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    os.makedirs("./initial_cams", exist_ok=True)
    cv2.imwrite(f"./initial_cams/{image_name}_{cam_idx}.jpg", superimposed_img)



# VOC_COLORMAP = np.array([
#     [0, 0, 0],        # 0=background
#     [128, 0, 0],      # 1=aeroplane
#     [0, 128, 0],      # 2=bicycle
#     [128, 128, 0],    # 3=bird
#     [0, 0, 128],      # 4=boat
#     [128, 0, 128],    # 5=bottle
#     [0, 128, 128],    # 6=bus
#     [128, 128, 128],  # 7=car
#     [64, 0, 0],       # 8=cat
#     [192, 0, 0],      # 9=chair
#     [64, 128, 0],     # 10=cow
#     [192, 128, 0],    # 11=diningtable
#     [64, 0, 128],     # 12=dog
#     [192, 0, 128],    # 13=horse
#     [64, 128, 128],   # 14=motorbike
#     [192, 128, 128],  # 15=person
#     [0, 64, 0],       # 16=potted plant
#     [128, 64, 0],     # 17=sheep
#     [0, 192, 0],      # 18=sofa
#     [128, 192, 0],    # 19=train
#     [0, 64, 128],     # 20=tv/monitor
# ])

def voc_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap

VOC_COLORMAP = voc_colormap()

def save_refined_cams(cam, annotation_path, cam_idx, threshold = 0.5):
    annotation_path = annotation_path.replace("\\", "/")
    image_name = re.sub(
        r".*/VOC2012/SegmentationClassAug/(.*).png", r"\1", annotation_path
    )
    jpeg_image_path = re.sub(
        r"(.*/VOC2012/).*", r"\1JPEGImages/" + image_name + ".jpg", annotation_path
    )
    original_image = cv2.imread(jpeg_image_path, cv2.IMREAD_COLOR)

    cam = cam.astype(np.uint8)
    pseudolabel = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_image = np.zeros((pseudolabel.shape[0], pseudolabel.shape[1], 3), dtype=np.uint8)
    for class_id in np.unique(pseudolabel):
        color_image[pseudolabel == class_id] = VOC_COLORMAP[class_id]
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    os.makedirs("./all_cams/new", exist_ok=True)
    success = cv2.imwrite(f"./all_cams/new/{image_name}_{cam_idx}.png", color_image)
    # if success:
    #     print("Saved pseudolabel!")