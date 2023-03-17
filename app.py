from flask import Flask, jsonify, request, render_template
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import io

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

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('classify.html')

# Define a route for image classification
@app.route('/classify', methods=['POST'])
def classify():
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
        output = model(img_tensor.to(device))
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        class_index = top_class.cpu().numpy()[0][0]
        prob = top_p.cpu().numpy()[0][0]
    
    # Return the predicted class and probabilities
    if class_index == 0:
        result = 'Real'
    else:
        result = 'Fake'
    return jsonify({'result': result, 'probability': prob})

if __name__ == '__main__':
    app.run()
