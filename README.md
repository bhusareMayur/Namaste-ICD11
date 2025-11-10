# 🌿 NAMASTE ↔ ICD-11 Mapping Platform  

A fully functional platform for mapping **Traditional Medicine (NAMASTE)** codes to **ICD-11**, integrating **machine learning**, **FHIR standards**, and a **CSV-driven architecture**.  
This system bridges the gap between ancient medical knowledge and modern digital healthcare interoperability.  

🔗 **Live Platform:** [https://namaste-icd11.onrender.com/](https://namaste-icd11.onrender.com/)


---

## 🚀 Overview  

The NAMASTE ↔ ICD-11 Mapping Platform provides intelligent code mapping, FHIR resource generation, and ML-based suggestions for unmapped traditional medicine codes.  
Built using **Node.js (Express)** for backend, **Flask (Python)** for machine learning, and **EJS + Bootstrap + Chart.js** for the web interface.  

---
## Prerequisites

- **Node.js 16+** and npm
- **Python 3.8+** and pip
- **Git** (optional, for version control)

## Project Structure

```
namaste-icd11-mapping/
├── datasets/ # CSV data files (currently 100 records)
│ ├── icd11_100_dataset.csv
│ ├── namaste_100_dataset.csv
│ └── namaste_icd11_100_mapping.csv
├── backend/ # Node.js Express backend
│ ├── server.js
│ ├── routes/api.js
│ ├── utils/csvLoader.js
│ ├── views/ # EJS templates
│ └── public/ # Static assets (CSS, JS)
├── ml/ # Python ML microservice
│ ├── app.py
│ ├── model_training.py
│ ├── requirements.txt
│ └── models/ # Trained models (auto-generated)
├── package.json
└── README.md
```

## Installation

### 1. Install Node.js Dependencies

```bash
cd <project-root>
npm install
```

### 2. Setup Python Environment

```bash
cd ml
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

### 1. Start Python ML Service

```bash
cd ml
source venv/bin/activate
python app.py
# ML service will run on http://127.0.0.1:5001 by default
```

### 2. Start Node.js Backend + Frontend

```bash
# in project root
npm install
npm start
# Backend will run on http://localhost:3000 (or port set in env)
```

## API Testing

### Health Check
```bash
curl http://localhost:3000/api/health
```

### Autocomplete
```bash
curl "http://localhost:3000/api/autocomplete?q=Jwa&type=namaste"
```

### Get Mapping
```bash
curl "http://localhost:3000/api/map/NAMASTE_AY_001"
```

### Generate FHIR
```bash
curl -X POST http://localhost:3000/api/generate-fhir \
  -H "Content-Type: application/json" \
  -d '{"patientId":"P001","namasteCode":"NAMASTE_AY_001","clinician":"Dr. XYZ"}'
```

## Features

- **Search & Autocomplete**: Fuzzy search with intelligent suggestions
- **Code Mapping**: View relationships between NAMASTE and ICD-11 codes  
- **ML Suggestions**: TF-IDF + cosine similarity for unmapped codes
- **FHIR Generation**: Create FHIR Condition resources with mappings
- **Analytics Dashboard**: Visualize mapping statistics and performance
- **Responsive UI**: Works on desktop, tablet, and mobile devices

## System Architecture

- **Frontend**: EJS templates + Bootstrap CSS + Chart.js
- **Backend**: Node.js + Express.js (CSV data in memory)
- **ML Service**: Python Flask + scikit-learn (TF-IDF vectors)
- **Data Layer**: CSV-based (currently 100 records)
- **Planned Upgrade**: MongoDB integration for scaling to 4500+ records

## Development

For live reloading during development:

```bash
npm run dev  # Uses nodemon for backend
# And in another terminal:
cd ml && python app.py  # ML service
```

## Troubleshooting

1. **ML Service Not Responding**: Ensure Python ML service is running on port 5001
2. **CSV Loading Errors**: Check that all CSV files exist in `/datasets` directory
3. **Port Conflicts**: Change ports in server.js and ml/app.py if needed
4. **Dependencies**: Run `npm install` and `pip install -r requirements.txt` if modules are missing
 
## Future Enhancements
1. MongoDB integration for handling large-scale datasets (4500+ records)
2. Dockerization for unified deployment
3. Authentication & Admin Panel for secure access
4. Confidence Scoring & Model Evaluation Metrics

## License
This project is open-source and free to use for research and development purposes.
 
## Author
**Mayur Bhusare**<br><br>
Full Stack Developer | Machine Learning Enthusiast<br>
📍 Pune<br>
🌐 Live Demo: https://namaste-icd11.onrender.com/<br>
💼 LinkedIn: https://www.linkedin.com/in/mayur-bhusare<br>
📧 Email: mayurbhusare8262@gmail.com<br>
