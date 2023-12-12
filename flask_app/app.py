# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

model2=models.resnet18()
device='cpu'
model2.fc = nn.Linear(model2.fc.in_features, 1)
model2=model2.to(device)
#model2.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

# modelvgg=models.vgg19()
# modelvgg.classifier[6]=nn.Linear(modelvgg.classifier[6].in_features, 1)
# modelvgg=modelvgg.to(device)
checkpoint = torch.load("best_model_27.pth", map_location=torch.device('cpu'))
model_state_dict = checkpoint['model_state_dict']
model2.load_state_dict(model_state_dict)
# modelvgg.load_state_dict(model_state_dict,strict=False)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('newhome.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file)
        print(type(img))
        #output_string="Hello Arunima"
        output_string=predict_image(img)
        print(output_string)
    return output_string    
    #return render_template('result2.html',prediction = output_string)

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
            return "Fake"
        else:
            return "Real"
        # modelvgg.eval()  
        # output =modelvgg(img_normalized)
        # output=torch.sigmoid(output)
        # print(output)
        # num = output.data[0]
        # print((float)(num[0]))
        # if(float)(num[0] > 0.5):
        #     return "Fake"
        # else:
        #     return "Real"


if __name__ == '__main__':
    app.run(debug=True)
    