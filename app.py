import yaml
from fastapi import FastAPI, UploadFile, File
import models
import torch
import src.cal as c
import src.calMetric as m
from PIL import Image
import torchvision.transforms as transforms
import base64



# Load config file
cfg_filepath = "config.yml"
with open(cfg_filepath, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

#Req: Takes in a PIL image that has to be transformed to a tensor, returns output as 0 or 1. Threshold TBD

# Load model
"""
print("Loading model")
wt = torch.load(cfg['model_name'])
net = models.DenseNet(cfg['num_in_classes'])
net.fc = torch.nn.Linear(342,8) # This is hard coded based on number of nodes in FC layer.
net.load_state_dict(wt['state_dict'])
print("Model successfully loaded")
torch.save(net,cfg['model'])
print("Model saved")
"""



#Model loading done before app initialisaton



# Initialise app
app = FastAPI()

img_path = "temppath" #cfg['image_path']

@app.get("/")
async def get_image_path():
    return {"message": img_path}

@app.post("/predict")
async def predict_img(file: UploadFile):
        for i in range(1, 6):
            model = torch.load(f"{cfg['model']}{i}.pt")
            print(f"Fold {i} loaded.")

            img = Image.open(file.file)
            img = img.convert('RGB')

            c.test(img, model, i, epsilon=0.002, temperature=1)
        res = m.test()

        if res <= float(cfg['threshold']):   # OOD has lower values than ID
            return 'OUT', res
        else:
            return 'IN', res