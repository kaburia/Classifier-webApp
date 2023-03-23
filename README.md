# Classifier-webApp

This repository contains code for a web application that performs image classification to determine if an image is real or fake. The web application is built using Python and Flask, and uses a pre-trained deep learning model for image classification. The user can upload an image, and the web application will classify the image as either real or fake, and display the result with a bar graph showing the probability of each classification. 



### Installation
1. Clone the repository
```
git clone https://github.com/kaburia/Classifier-webApp.git
cd Classifier-webApp
```
2 Setting up the environment
```
python3 -m pip install -r requirements.txt
```
3. Start the server
```
python3 app.py
```
### Usage
You can interact with the app in two ways:

1. Use the API: Send a POST request to the API endpoint localhost:5500/api/classify with an image in the request body to get a classification result. See [example.ipynb](./example.ipynb) for an example.

2. View the HTML pages: open your web browser and go to [localhost:5500](http://localhost:5500).


![image](https://user-images.githubusercontent.com/88529649/226077466-b50a2c33-7f38-4b21-87a6-9b25709b6a3d.png)

