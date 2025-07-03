# Loading necessary Library uded for MQTT communication.
import paho.mqtt.client as mqtt

# queue library funcionality
from collections import deque

#Loading necessary Libraries for image recognition.
import torch
from BoardRecognitionArchitectures.BoardRecognitionAI import BoardRecognitionAI
#from BoardRecognitionArchitectures.BoardRecognitionAIv2 import BoardRecognitionAIv2 #  Uncomment this line after comparing with older version.

#Loading necessary Libraries for scraping images from Raspberry Pi local host.
import requests
import cv2
from ImageHandling import ImageConverter
converter = ImageConverter()
import numpy as np
import matplotlib.pyplot as plt

IP_ADDRESS = "192.168.1.31"
WEB_CAM_HISTORY = "E:\Datasets\SOC\RaspberryPI\WebCamHistory"
def get_image_from_pi(ip_address):
    """
    Fetches an image from the Raspberry Pi's camera.
    """
    DATABASE_NAME = "chessbot_view_history"
    url = f"http://{ip_address}/capture"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print("Failed to fetch image:", response.status_code)

import time
import keyboard
import json

# Sets up the MQTT client and topics for communication with ESP32 boards.
KILL_SWITCH = "q"
SERIAL_NAME = "COM3"
TOPICS = {
    KILL_SWITCH: "topics/break",
    1: "topics/esp32/0",
    2: "topics/esp32/1"
}


class EncodersPosition:
    """
    Class for handling encoder positions.
    """
    def __init__(self, serial_name):
        self.serial_name = serial_name
        self.list = [
            {"name": "encoder1", "position": 0, "last_update": time.time()},
            {"name": "encoder2", "position": 0, "last_update": time.time()},
            {"name": "encoder3", "position": 0, "last_update": time.time()},
            {"name": "encoder4", "position": 0, "last_update": time.time()},
            {"name": "encoder5", "position": 0, "last_update": time.time()},
            {"name": "encoder6", "position": 0, "last_update": time.time()}
        ]

    def get_state_all(self):
        return self.list
    
    def get_state(self, encoder_name):
        """
        Returns the state of a specific encoder.
        """
        for encoder in self.list:
            if encoder["name"] == encoder_name:
                return encoder
        raise Exception(f"Encoder {encoder_name} not found.")


def send_command(command, reciever):
    """Sends a command to a celected ESP32 board."""
    client.publish(TOPICS[reciever], json.dumps(command))

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    for topic in TOPICS.values():
        if msg.topic == topic:
            print(f"Received command for {topic}: {payload}")
        
    print(f"Received message: {msg.topic} {msg.payload.decode()}")

client = mqtt.Client("processing_unit")
client.on_message = on_message

client.connect("localhost", 1883, 60)

client.subscribe("sensors/#")
client.publish("sensors", "Hello from the broker!")



command = None



def IK():
    """
    Simulates the Inverse Kinematics (IK) process by creating a queue with a message.
    ###This function is a placeholder for the actual IK logic that would process images and commands.

    """
    queue = deque()
    queue.append("Image received from Raspberry Pi")
    return queue

if __name__ == "__main__":
    print("Script initialized.")
    print("--------------------------------")

    model = BoardRecognitionAI()  
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    MODEL_PATH = r"E:\Datasets\SOC\ChessPositionsRenders\model.pth"
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict)

    model.eval()

    print("Press 'q' to exit.")
    client.loop_start()
    
    while True:
        if keyboard.is_pressed(KILL_SWITCH):
            client.publish(TOPICS[KILL_SWITCH], json.dumps({"command": "stop"}))
            print("Kill switch activated. Exiting...")
            break
        
        else:
            # Get Image from Raspberry Pi
            image_in_bytes = get_image_from_pi(IP_ADDRESS)
            timestamp = time.time()
            img_path = f"{WEB_CAM_HISTORY}\{timestamp}.jpg"

            with open(img_path, "wb") as f:
                f.write(image_in_bytes)

            cropped_images = converter.crop_image(converter.image_to_array(img_path), None, "state")

            images = cropped_images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            fig_logits, col_logits = model(images)
            fig_pred = fig_logits.argmax(dim=1).item()
            col_pred = col_logits.argmax(dim=1).item()
            print(f"Figure: {fig_pred}, Color: {col_pred}")

            queue = IK()
            while queue:
                instruction = queue.popleft()

        # Wait to avoid lagging the loop
        time.sleep(0.1)