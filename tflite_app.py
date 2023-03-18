from flask import Flask, jsonify, request, render_template, redirect, url_for
from PIL import Image
import io
import uuid
import os
import tensorflow as tflite
import numpy as np

# Use GPU if it's available
interpreter = tflite.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath('static')

# model = torch.hub.load("ultralytics/yolov5", 'yolov5m.pt')

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for the homepage
@app.route('/')
def home():
    return redirect(url_for('classify'))

# Define a route for image classification
# @app.route('/classify', methods=['POST'])
# Define a route for image classification
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        # Get the image file from the request
        img = request.files['file'].read()

        # Load the image using PIL
        img = Image.open(io.BytesIO(img)).convert('RGB')

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = np.array(img)

        img_array = (img_array - 127.5) / 127.5
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        img_array = np.transpose(img_array, (0, 3, 1, 2))


        # Make a prediction using the model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        class_index = np.argmax(output[0])
        # prob = output[0][0]
        # get the probability of the predicted class between 0 and 1
        logits = output[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        prob = probs[class_index]               

        # Return the predicted class and probabilities
        if class_index != 0:
            result = 'Real'
            probability_real = round(float(prob) * 100, 2)
            probability_fake = round(100 - probability_real, 2)
        else:
            result = 'Fake'
            probability_fake = round(float(prob) * 100, 2)
            probability_real = round(100 - probability_fake, 2)

        # Save the image with a unique filename
        filename = str(uuid.uuid4()) + '.jpg'
        file_path = f"{UPLOAD_FOLDER}\{filename}"
        # print(img.show())
        img.save(file_path, format='JPEG')
        print("File saved at path:", file_path)

        # Render the classify.html template with the prediction results
        return render_template('classify.html', file_path=filename, prediction=result,
                               probability_real=probability_real, probability_fake=probability_fake)
    else:
        return render_template('classify.html')



if __name__ == '__main__':
    app.run(host='192.168.0.102', port=5500)
