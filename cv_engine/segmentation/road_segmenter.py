import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'deeplabv3_resnet50',
    pretrained=True
)
model.eval()
def segment_road(frame):
    frame = cv2.resize(frame,(320,180))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        output = model(input_batch)["out"][0]
    end = time.time()
    print("Inference:", end - start)

    with torch.no_grad():
        output = model(input_batch)["out"][0]
    prediction = output.argmax(0).byte().cpu().numpy()
    print(np.unique(prediction))
    road_mask = (prediction == 7).astype(np.uint8)*255
    return road_mask    