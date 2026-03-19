# 🏡 PropVal AI: California Real Estate Prediction

![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-blue.svg)
![Django](https://img.shields.io/badge/Backend-Django%20REST-darkgreen.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Tailwind](https://img.shields.io/badge/UI-Tailwind%20CSS-06B6D4.svg)

An AI-powered, full-stack web application that predicts California real estate prices in real-time based on location, property age, and neighborhood income using a trained Random Forest machine learning model.

> 🟢 **Live Demo:** [https://california-real-estate-ai.netlify.app](https://california-real-estate-ai.netlify.app)

## 📖 Overview
This project bridges the gap between Data Science and Full Stack Web Development. It takes a machine learning model trained on the official California Housing dataset and serves it through a robust Django REST API. The frontend is a sleek, highly responsive React dashboard featuring interactive maps, real-time data validation, and instantaneous AI valuations.

## ✨ Features
- **Real-Time AI Valuation:** Instant price predictions powered by a compressed, high-accuracy Scikit-Learn Random Forest Regressor.
- **Interactive Geospatial Mapping:** Dynamic `react-leaflet` maps that automatically fly to the exact coordinates entered by the user.
- **Smart Data Validation:** Built-in geographical bounding boxes that restrict latitude and longitude inputs strictly to California state lines to prevent model hallucinations.
- **Modern UI/UX:** A clean, B2B SaaS-inspired dashboard built with the newly released Tailwind CSS v4.
- **Decoupled Architecture:** A clean separation of concerns, utilizing CORS to allow the React frontend to communicate seamlessly with the Python backend.

## 🛠 Tech Stack
- **Frontend Framework:** React (initialized via Vite)
- **UI & Styling:** Tailwind CSS v4
- **Mapping:** React-Leaflet & OpenStreetMap
- **Backend API:** Python, Django, Django REST Framework
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
