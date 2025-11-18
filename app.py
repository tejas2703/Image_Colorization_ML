from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model paths (same as before)
PROTOTXT = "./model/colorization_deploy_v2.prototxt"
MODEL = "./model/colorization_release_v2.caffemodel"
POINTS = "./model/pts_in_hull.npy"

# Load model (same as before)
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Process uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Save uploaded file
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Colorize image (same as before)
            image = cv2.imread(filepath)
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)

            colorized = (255 * colorized).astype("uint8")
            output_filename = f"colorized_{filename}"
            output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
            cv2.imwrite(output_filepath, colorized)

            return jsonify({'colorized_image': output_filename}), 200

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
