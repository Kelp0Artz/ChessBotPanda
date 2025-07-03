# 🤖 ChessBotPANDA

ChessBotPANDA was created as part of a personal challenge to win SOČ, a prestigious Slovak student research competition. It combines AI, robotics, and computer vision into a fully autonomous chess-playing system. The project is not only a technical showcase but also a step toward deeper research in real-time decision-making systems.
This project was named after my best friend, Panda 🐼.

---

## 🚀 Features

- 🧠 Neural network-based decision making for functional chess bot, which is mimicking human moves (LSTM / CNN)
- 📷 Camera-based object detection
- 🦾 Robotic arm with 6 degrees of freedom
- 🔌 ESP32s + MQTT integration
- 🔌 Raspberry Pi 3B + Raspberry Pi Cam V2 + HTTP integration
- 🧰 Real-time control 

---

## 🎥 Demo

- ***soonish

---

## 🧠 AI Model Overview

### ChessBot
Different versions of this model are designed to compare **accuracy**, **inference time**, and **training duration**. This helps identify the most efficient architecture for real-time chess piece recognition.

#### v1
- Model type: LSTM / CNN 
- Training dataset: sample.h5
- Input/Output: Piece positions + AverageElo → Move
- Frameworks used: PyTorch, OpenCV

#### v2
work in progress
- Model type: GAN

### ChessPiecesRecognizer
Different versions of this model are designed to compare **accuracy**, **inference time**, and **training duration**. This helps identify the most efficient architecture for real-time chess piece recognition.

#### v1
- Model type: CNN
- Training dataset: *** soon published
- Input/Output: Image of one square on board → figure typeand color
- Frameworks used: PyTorch, OpenCV

#### v2
work in progress
- Input/Output: Adding previous state of the board and the square

#### v3
work in progress
- Input/Output: Computing all squares in one go 

---

## ⚙️ Hardware Overview

| Component         | Details                           |
|-------------------|-----------------------------------|
| Microcontrollers  | 2x ESP32 WROOM-32                 |
|                   | Raspberry PI 3b                   |
| Motors            | 3x NEMA 17 + 3x NEMA 23 Stepper   |
| Drivers           | 6x DM556                          |
| Sensors           | 5x AS5048A Magnetic Encoder       |
| Power Supply      | 48V, 1500W PSU                    |
| Camera            | Raspberry Pi Cam V2               |
| GPU               | RTX 3060                          |

---

## 🛠️ Installation

- *** soon
### Datasets
- *** soon downloadable on Hugging Face
---

## 📬 Contact

If you have any questions, suggestions or want to colaborate, feel free to reach out:

- **Name:** Juraj Orisko
- **Email:** juraj.orisko007@gmail.com
- **Discord** Kelp0.py (with zero)
- **LinkedIn:** *soon*

# ChessBotPanda
This project was named after my best friend, Panda 🐼.
This repository mainly focuses on Open-sourcing my project, with wich I want to win a national competition.
