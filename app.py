import paho.mqtt.client as mqtt
import time 
def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} {msg.payload.decode()}")

client = mqtt.Client()

client.on_message = on_message

client.connect("localhost", 1883, 60)  # or your IP if running remotely

client.subscribe("sesnors/#") 
client.publish("sesnors/#", "Hello from the broker!")

client.loop_start()

command = None
while True:
    time.sleep(1) 
    if command != None:
        client.publish("drivers/#", command)
        command = None
    