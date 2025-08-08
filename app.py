import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from torchvision.models import resnet18
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import pyautogui
#I choose to go with gradio beacause it allows developer to create simple UI and write backend 
#code in single python file and easy integration of AI/ML

model=resnet18(pretrained=True)
model.eval()

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.299,0.224,0.225])
    ]
)

labels = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels=requests.get(labels).text.split('\n')

def predict(img):
    global predicted_op
    img_t=transform(img).unsqueeze(0)
    with torch.no_grad():
        pred=model(img_t).argmax().item()
        predicted_op=imagenet_labels[pred]
    return predicted_op

def automate():
    global predicted_op
    if not predicted_op.strip():
        return 
    else:
        driver = webdriver.Chrome()
        driver.get("https://www.google.com")
        time.sleep(2)
        search = driver.find_element(By.NAME,'q')
        search.send_keys(predicted_op)
        search.send_keys(Keys.ENTER)
        time.sleep(4)
        pyautogui.click(x=85, y=300, button='left') #to check im not robot
        time.sleep(20)
        pyautogui.click(x=150, y=340, button='left') #to click on images
        time.sleep(4)
        pyautogui.click(x=200, y=340, button='left') # to click on videos
        time.sleep(4)
        wikipedia = "https://en.wikipedia.org/wiki/" + predicted_op
        driver.get(wikipedia)

        return predicted_op
    
Title = "<center><h1>Object Detecion and Seeking Details</h1></center>"

description = "Upload an image in upload box it will detect the object." \
            "If you press Seek more details it will automate the process"

note = "**NOTE**:firstly run Detect Object then run Seek more details "

with gr.Blocks(title="Image Detection and Info") as demo:
    gr.HTML(Title)
    gr.Markdown(description)
    gr.Markdown(note)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label="Upload an Image")
            detect_button = gr.Button("Detect Object")
            detection_output = gr.Textbox(label="Detection Result")
        with gr.Column():
            automate_button = gr.Button("Seek more details")
            automation_output = gr.Textbox(label="Seeking Progress")

    detect_button.click(fn=predict, inputs=image_input, outputs=detection_output)
    automate_button.click(fn=automate, outputs=automation_output)

demo.launch()