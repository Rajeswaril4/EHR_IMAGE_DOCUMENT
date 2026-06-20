# EHR AI Clinical Assistant

## Project Overview
The EHR AI Clinical Assistant is an AI-powered web application that enhances Electronic Health Records (EHR) by automating medical image analysis, clinical note generation, and ICD-10 disease coding.

The application helps healthcare professionals reduce manual work, improve documentation accuracy, and generate AI-assisted medical reports.

---

## Features

- User Authentication (Login/Register)
- JWT-based Secure Authentication
- Admin Dashboard
- User Dashboard
- Medical Image Upload
- Image Enhancement
- AI Disease Analysis
- Clinical Note Generation
- ICD-10 Code Prediction
- Bulk CSV Processing
- Report History
- Analytics Dashboard
- MySQL Database Integration
- Report Download

---

## Technologies Used

### Frontend
- HTML5
- Tailwind CSS
- JavaScript
- Chart.js

### Backend
- Python
- Flask
- JWT Authentication

### Database
- MySQL

### AI & Machine Learning
- OpenCV
- BM3D Image Denoising
- CLAHE Contrast Enhancement
- SSIM
- PSNR
- Machine Learning-based Medical Image Analysis
- Generative AI for Clinical Notes

---

## Project Architecture

User
↓
Login/Register
↓
JWT Authentication
↓
Dashboard
↓
Upload Medical Image
↓
Image Enhancement
↓
Feature Extraction
↓
ML Disease Prediction
↓
AI Clinical Note Generation
↓
ICD-10 Mapping
↓
Store Report in MySQL
↓
Analytics Dashboard

---

## Modules

1. Authentication Module
2. Image Processing Module
3. Disease Prediction Module
4. Clinical Note Generator
5. ICD-10 Coding Module
6. Report Management
7. Bulk CSV Processor
8. Admin Dashboard

---

## Dataset

The project uses:

- EHR.csv
- CT Images
- MRI Images

The dataset is used to train and validate the machine learning model.

---

## Installation

1. Clone Repository

git clone <repository_url>

2. Install Dependencies

pip install -r requirements.txt

3. Configure MySQL Database

Update database credentials inside .env

4. Run

python app.py

---

## Future Enhancements

- Multi-Hospital Support
- Appointment Scheduling
- Voice-based Report Generation
- Cloud Deployment
- Real-time Doctor Collaboration

---

## Team Members

Developed as a Final Year B.Tech Artificial Intelligence & Data Science Project.
