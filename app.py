from flask import Flask, render_template, request
import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import base64
import io
import gdown

# -------------------------------
# GOOGLE DRIVE YOLO MODEL DOWNLOAD (USING GDOWN)
# -------------------------------
def download_model():
    model_path = "models/yolov8x-pose-p6.pt"
    FILE_ID = "1Up8eKUQHsNiEU7naRx5RW3OkLvmPTgy_"  # your file id

    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(model_path):
        print("\nDownloading YOLO model using gdown...\n")

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        gdown.download(url, model_path, quiet=False, fuzzy=True)

        print("\nModel download completed.\n")
    else:
        print("YOLO model already exists.")

# Download model at startup
download_model()


# -------------------------------
# CLASS DICTIONARY
# -------------------------------
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


# -------------------------------
# POSE CLASSIFIER MODEL
# -------------------------------
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


def load_model():
    model_pose = YogaClassifier(num_classes=len(classes_dict), input_length=32)
    model_pose.load_state_dict(torch.load("best.pth", map_location=torch.device("cpu")))
    model_pose.eval()
    return model_pose


# -------------------------------
# YOLO + CLASSIFIER PREDICTION
# -------------------------------
def make_prediction(model, image_path):

    yolo_model = YOLO("models/yolov8x-pose-p6.pt")

    results = yolo_model.predict(image_path, verbose=False)

    for r in results:
        im_array = r.plot()

        keypoints = r.keypoints.xyn.cpu().numpy()[0]
        keypoints = keypoints.reshape((1, keypoints.shape[0] * keypoints.shape[1]))[0].tolist()

        keypoints_tensor = torch.tensor(keypoints[2:], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logit = model(keypoints_tensor)
            pred = torch.softmax(logit, dim=1).argmax(dim=1).item()
            prediction = classes_dict[pred]

        image = io.BytesIO()
        plt.imshow(im_array[..., ::-1])
        plt.title(f"Prediction: {prediction}", color="green")
        plt.savefig(image, format='png')
        plt.close()

        image.seek(0)
        plot_base64 = base64.b64encode(image.read()).decode('utf-8')

        return plot_base64, prediction


# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict():
    image_file = request.files['file']
    image_path = 'temp.png'
    image_file.save(image_path)

    model = load_model()
    plot_base64, prediction = make_prediction(model, image_path)

    os.remove(image_path)

    return render_template('prediction.html',
                           prediction=prediction,
                           plot_base64=plot_base64)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
