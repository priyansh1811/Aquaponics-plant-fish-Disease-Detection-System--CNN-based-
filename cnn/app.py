import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Debugging: Print template directory
print(f"🔍 Flask is looking for templates in: {os.path.abspath('templates')}")

# ✅ Allow file uploads to be stored temporarily
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create upload directory if not exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load Trained Model
MODEL_PATH = "/Users/priyansh18/Desktop/farmhelp/aquaponics/plant_disease_detection_cnn_fixed.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# ✅ Load Class Names from Dataset Folder
data_dir = "/Users/priyansh18/Downloads/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
class_names = sorted([cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))])

# ✅ Function to Preprocess Image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model
    return img_array

# ✅ Route for Homepage (Main Webpage)
@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("frontend.html")
    except Exception as e:
        print(f"❌ Template Error: {e}")
        return jsonify({"error": "Template not found!"})

# ✅ Route for Handling Image Upload & Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file!"})

        # ✅ Save Uploaded File
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # ✅ Preprocess and Predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0]

        # ✅ Get top prediction
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = f"{prediction[predicted_index] * 100:.2f}%"

        return jsonify({"class": predicted_class, "confidence": confidence, "image_url": file_path})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Internal Server Error! Check Flask logs for details."})

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)