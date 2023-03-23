from flask import Flask, jsonify, request, render_template, redirect, url_for, jsonify
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import io
import uuid
import os

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256,2),
                                 nn.LogSoftmax(dim=1))

# criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);


state = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath('static')

# model = torch.hub.load("ultralytics/yolov5", 'yolov5m.pt')

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for the homepage
@app.route('/')
def home():
    return redirect(url_for('classify'))

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        # Get the image file from the request
        img = request.files['file'].read()

        # Load the image using PIL
        img = Image.open(io.BytesIO(img)).convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            model.eval()
            output = model.forward(img_tensor.to(device))
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        class_index = top_class.cpu().numpy()[0][0]
        prob = top_p.cpu().numpy()[0][0]

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

# Define a route for image classification
@app.route('/api/classify', methods=['GET', 'POST'])
def apiclassify():
    if request.method == 'POST':
        # Get the image file from the request
        img = request.files['file'].read()

        # Load the image using PIL
        img = Image.open(io.BytesIO(img)).convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            model.eval()
            output = model.forward(img_tensor.to(device))
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        class_index = top_class.cpu().numpy()[0][0]
        prob = top_p.cpu().numpy()[0][0]

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
        # file_path = f"{UPLOAD_FOLDER}\{filename}"
        # # print(img.show())
        # img.save(file_path, format='JPEG')
        # print("File saved at path:", file_path)

        # Render the classify.html template with the prediction results
        # return render_template('classify.html', file_path=filename, prediction=result,
                            #    probability_real=probability_real, probability_fake=probability_fake)
        return jsonify({'result': result, 
                        'probability_real': probability_real, 
                        'probability_fake': probability_fake})
    
    # else:
    #     return render_template('classify.html')

# Define a route for the API endpoint that returns the model's class names
@app.route('/api/classnames', methods=['POST'])
def classnames():
    # Replace this with your own list of class names
    class_names = ['Real', 'Fake']
    return jsonify(class_names)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
