from flask import Flask, render_template, request
import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image

# ---------------------------------------------------------
# LIGHTWEIGHT YOLO MODEL (NO DOWNLOAD, NO OOM)
# ---------------------------------------------------------
MODEL_PATH = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt"

# ---------------------------------------------------------
# CLASS DICTIONARY
# ---------------------------------------------------------
classes_dict = {
    0: 'Adho Mukha Svanasana', 1: 'Adho Mukha Vrksasana', 2: 'Alanasana', 3: 'Anjaneyasana',
    4: 'Ardha Chandrasana', 5: 'Ardha Matsyendrasana', 6: 'Ardha Navasana',
    7: 'Ardha Pincha Mayurasana', 8: 'Ashta Chandrasana', 9: 'Baddha Konasana',
    10: 'Bakasana', 11: 'Balasana', 12: 'Bitilasana', 13: 'Camatkarasana',
    14: 'Dhanurasana', 15: 'Eka Pada Rajakapotasana', 16: 'Garudasana', 17: 'Halasana',
    18: 'Hanumanasana', 19: 'Malasana', 20: 'Marjaryasana', 21: 'Navasana',
    22: 'Padmasana', 23: 'Parsva Virabhadrasana', 24: 'Parsvottanasana',
    25: 'Paschimottanasana', 26: 'Phalakasana', 27: 'Pincha Mayurasana',
    28: 'Salamba Bhujangasana', 29: 'Salamba Sarvangasana',
    30: 'Setu Bandha Sarvangasana', 31: 'Sivasana', 32: 'Supta Kapotasana',
    33: 'Trikonasana', 34: 'Upavistha Konasana', 35: 'Urdhva Dhanurasana',
    36: 'Urdhva Mukha Svsnssana', 37: 'Ustrasana', 38: 'Utkatasana',
    39: 'Uttanasana', 40: 'Utthita Hasta Padangusthasana',
    41: 'Utthita Parsvakonasana', 42: 'Vasisthasana', 43: 'Virabhadrasana One',
    44: 'Virabhadrasana Three', 45: 'Virabhadrasana Two', 46: 'Vrksasana'
}

# ---------------------------------------------------------
# POSE CLASSIFIER
# ---------------------------------------------------------
class YogaClassifier(torch.nn.Module):
    def __init__(self, num_classes, input_length):
        super(YogaClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_length, 64)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.layer2 = torch.nn.Linear(64, 64)
        self.outlayer = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        return self.outlayer(x)


def load_classifier():
    """Load classification model safely."""
    model_pose = YogaClassifier(num_classes=len(classes_dict), input_length=32)
    state_dict = torch.load("best.pth", map_location="cpu", weights_only=True)
    model_pose.load_state_dict(state_dict)
    model_pose.eval()
    return model_pose


# ---------------------------------------------------------
# LOAD MODELS ONCE
# ---------------------------------------------------------
print("Loading YOLOv8n-pose (Render Safe)...")
yolo_model = YOLO(MODEL_PATH)
classifier_model = load_classifier()
print("Models loaded successfully!")

# ---------------------------------------------------------
# PREDICTION PIPELINE
# ---------------------------------------------------------
def make_prediction(image_path):
    results = yolo_model.predict(image_path, verbose=False)

    for r in results:
        im_array = r.plot()

        keypoints = r.keypoints.xyn.cpu().numpy()[0]
        keypoints = keypoints.reshape(1, -1)[0].tolist()

        keypoints_tensor = torch.tensor(keypoints[2:], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_logits = classifier_model(keypoints_tensor)
            pred = torch.softmax(pred_logits, dim=1).argmax().item()

        prediction = classes_dict[pred]

        # Plot image with prediction text
        image_bytes = io.BytesIO()
        plt.imshow(im_array[..., ::-1])
        plt.title(f"Prediction: {prediction}", color="green")
        plt.savefig(image_bytes, format="png")
        plt.close()

        image_bytes.seek(0)
        base64_img = base64.b64encode(image_bytes.read()).decode("utf-8")

        return base64_img, prediction


# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict():
    image_file = request.files["file"]
    image_path = "temp.png"
    image_file.save(image_path)

    plot_base64, prediction = make_prediction(image_path)

    os.remove(image_path)

    return render_template(
        "prediction.html",
        prediction=prediction,
        plot_base64=plot_base64,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

