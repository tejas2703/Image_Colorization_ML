import cv2
import numpy as np
from tabulate import tabulate

# Path to your model files
PROTOTXT = "./model/colorization_deploy_v2.prototxt"
MODEL = "./model/colorization_release_v2.caffemodel"
POINTS = "./model/pts_in_hull.npy"

# Load the pre-trained model using OpenCV's DNN module
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Load the cluster centers (for color prediction)
pts = np.load(POINTS)
pts = pts.transpose().reshape(2, 313, 1, 1)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Collecting model layer information
layer_names = net.getLayerNames()
layers_info = []

for i, layer_name in enumerate(layer_names):
    layer = net.getLayer(i + 1)
    layer_type = layer.type
    num_params = len(layer.blobs)
    layers_info.append([i + 1, layer_name, layer_type, num_params])

# Print model architecture in tabular format
print(tabulate(layers_info, headers=["Layer No.", "Layer Name", "Type", "No. of Parameters (Blobs)"], tablefmt="grid"))
