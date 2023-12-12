# Importing essential libraries
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

model2=models.resnet18()
device='cpu'
model2.fc = nn.Linear(model2.fc.in_features, 1)
model2=model2.to(device)
model2.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        print("inside if condition")
        img=Image.open('./real2.jpeg')
        print(type(img))
        #output_string="Hello Arunima"
        output_string=predict_image(img)
        print(output_string)
    return render_template('result2.html',prediction = output_string)

def predict_image(img):
    transform_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
    
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad():
        model2.eval()  
        output =model2(img_normalized)
        output=torch.sigmoid(output)
        print(output)
        num = output.data[0]
        print((float)(num[0]))
        if(float)(num[0]) > 0.5:
            return "fake"
        else:
            return "real"


if __name__ == '__main__':
    app.run(debug=True)
    