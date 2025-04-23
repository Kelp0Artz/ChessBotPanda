import paho.mqtt.client as mqtt
import time
import keyboard

TOPICS = {
    0: "topics/break",
    1: "topics/esp32/0",
    2: "topics/esp32/1"
}

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message

client.connect("localhost", 1883, 60)

client.subscribe("sensors/#")
client.publish("sensors", "Hello from the broker!")

client.loop_start()

command = None
while True:
    time.sleep(0.05)  # Prevent CPU overload
    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('w'):
        command = "w"
    elif keyboard.is_pressed('a'):
        command = "a"
    elif keyboard.is_pressed('s'):
        command = "s"
    elif keyboard.is_pressed('d'):
        command = "d"

    if command is not None:
        client.publish("sensors", command)
        command = None
