# AI-Driven-forensic-face-sketch-construction-and-recognition
# Forensic Face Generation & Criminal Identification System

## Overview

The **Forensic Face Generation & Criminal Identification System** is an AI-powered investigation platform designed to assist law enforcement agencies in generating suspect faces from witness descriptions and matching them against a criminal database.

The system combines speech recognition, natural language processing, AI image generation, facial recognition, and role-based case management into a single web application.

---

## Features

### AI-Powered Forensic Pipeline

* 🎤 **Witness Audio Processing**

  * Upload witness audio recordings
  * Automatic speech-to-text transcription using Deepgram

* 📝 **Attribute Extraction**

  * Extract facial attributes from witness descriptions
  * Detect age, gender, skin tone, facial marks, accessories, and more

* 🧠 **AI Face Generation**

  * Generate realistic suspect faces from extracted attributes
  * Support for witness sketch enhancement
  * Multiple candidate face generation

* 🔍 **Criminal Database Matching**

  * Face embedding generation using ArcFace (ONNX)
  * Similarity-based suspect identification
  * Attribute-assisted matching

---

## User Roles

### Super Admin

* Manage administrators
* View system audit logs
* Monitor system activities

### Admin

* Manage police officers
* Manage criminal database
* Rebuild face embeddings database

### Police Officer

* Create investigation cases
* Upload witness statements
* Generate suspect faces
* Search criminal database matches

---

## System Architecture

```text
Witness Audio
      │
      ▼
Speech-to-Text
      │
      ▼
Attribute Extraction
      │
      ▼
AI Face Generation
      │
      ▼
Face Matching Engine
      │
      ▼
Top Suspect Results
```

---

## Technology Stack

### Backend

* Python
* Flask
* Flask-Login
* Flask-Bcrypt
* SQLite

### Artificial Intelligence

* ONNX Runtime
* ArcFace Embeddings
* Stable Diffusion
* ControlNet
* Hugging Face Models

### Computer Vision

* OpenCV
* Pillow
* NumPy

### APIs

* Deepgram Speech-to-Text
* Hugging Face Inference API

---

## Project Structure

```text
forensic-face-generation/
│
├── app.py
├── config.py
├── requirements.txt
├── .env.example
│
├── modules/
│   ├── speech_to_text.py
│   ├── attribute_parser.py
│   ├── face_generator.py
│   ├── face_matcher.py
│   └── face_enhancer.py
│
├── data/
│   ├── criminal_db/
│   ├── generated_faces/
│   ├── uploads/
│   └── forensic.db
│
├── models/
│   └── db_embeddings_v5.pkl
│
├── database/
│   └── schema.sql
│
└── docs/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/forensic-face-generation.git

cd forensic-face-generation
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file:

```env
SECRET_KEY=your_secret_key

DEEPGRAM_API_KEY=your_deepgram_key

HF_API_TOKEN=your_huggingface_token

GENERATION_MODE=local
```

### Generation Modes

#### Local Mode

```env
GENERATION_MODE=local
```

* Fully offline after model download
* Supports sketch-to-photo generation
* Recommended for forensic investigations

#### API Mode

```env
GENERATION_MODE=api
```

* Uses Hugging Face Inference API
* Faster setup
* No local model downloads

---

## Running the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Default Credentials

```text
Username: superadmin
Password: Admin@123
```

Change the password immediately after first login.

---

## Workflow

### Step 1 – Upload Witness Audio

Police officer uploads witness audio recording.

### Step 2 – Description Analysis

System extracts structured facial attributes.

### Step 3 – Face Generation

AI generates multiple suspect face candidates.

### Step 4 – Criminal Matching

Generated faces are compared against the criminal database and ranked by similarity.

---

## Key Modules

| Module              | Purpose                      |
| ------------------- | ---------------------------- |
| speech_to_text.py   | Audio transcription          |
| attribute_parser.py | Witness description analysis |
| face_generator.py   | AI suspect generation        |
| face_matcher.py     | Face similarity matching     |
| face_enhancer.py    | Image enhancement            |
| adaface_onnx.py     | Face embeddings              |

---

## Security Features

* Role-Based Access Control (RBAC)
* Password Hashing using Bcrypt
* Session Management
* Audit Logging
* Secure File Upload Handling

---

## Future Enhancements

* Real-time CCTV integration
* Multi-language witness support
* Advanced facial aging simulation
* Cloud deployment support
* Mobile application support

---

## Disclaimer

This project is intended for educational, research, and authorized law-enforcement use only. Users must comply with applicable privacy, surveillance, biometric, and data-protection laws before deployment.

---

## License

This project is licensed under the MIT License.
