
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
import io


from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = load_model("C:/Users/hadia/OneDrive/Desktop/new/TEST1/sign.h5")

# Compile the model with your desired settings
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Create a mapping for class labels
class_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

@app.route("/") 
def home():
    return render_template("index.html")

@app.route("/index") 
def index():
    return render_template("index.html")

@app.route("/about") 
def about():
    return render_template("about.html")


@app.route("/tryItNow") 
def tryItNow():
    return render_template("tryItNow.html")





@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'frame' not in data:
        return jsonify({'error': 'No frame part in the request'}), 400

    # Decode the Base64-encoded image data
    base64_img = data['frame'].split(",")[1]
    img_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(img_bytes))

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Resize the frame to match the input size expected by your model
    frame_resized = cv2.resize(image_np, (100, 100))

    # Expand dimensions to create a batch-sized image
    img_array = np.expand_dims(frame_resized, axis=0)
    img_array = img_array.astype(np.float32) / 255.0

    # Make predictions using your model
    predictions = model.predict(img_array)

    # Assuming a single-class prediction
    predicted_class = np.argmax(predictions)
    predicted_label = class_mapping[predicted_class]
    
    # Print the predicted label to the terminal
    print(f'Predicted Label: {predicted_label}')

    return jsonify({'label': predicted_label})


    


if __name__ == "__main__":
    app.run(host='0.0.0.0')