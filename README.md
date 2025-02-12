# Federated Health Classification System 🏥

A distributed health classification system that uses federated learning to train models while preserving data privacy. The system enables training health classification models while keeping data private on local clients, implementing federated averaging (FedAvg) to aggregate model weights from multiple clients into a global model.

## 🌟 Features

- **Data Privacy**: Keep sensitive health data local while contributing to global model
- **Local Training**: Train models on individual client datasets
- **Federated Learning**: Aggregate models across multiple clients using FedAvg
- **Real-time Monitoring**: Track training progress and model performance
- **Interactive UI**: User-friendly web interface for data upload and visualization
- **Model Persistence**: Automatic saving and versioning of models
- **Performance Tracking**: Comprehensive metrics and experiment logging

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Machine Learning**: PyTorch, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Storage**: SQLite, File System
- **Development**: Python 3.8+

## 📁 System Architecture

## 🚀 Getting Started

### Prerequisites

bash
python 3.8+
pip
virtual environment (recommended)


### Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/FedHealth.git
cd FedHealth

2. Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

3. Install dependencies:

bash
pip install -r requirements.txt

4. Set up environment:

bash
cp .env.example .env
# Edit .env with your settings

## 🎯 Usage

1. Start the backend server:

bash
cd backend
python run.py

2. Start the frontend (new terminal):

bash
streamlit run app.py

3. Access web interface at `http://localhost:8501`

## 📊 Data Format

Required CSV columns:

heart_rate    : float (50-120 bpm)
systolic_bp   : float (90-160 mmHg)
diastolic_bp  : float (60-100 mmHg)
spo2          : float (90-100%)
healthy       : int (0 or 1)

Generate sample data:

bash
python backend/tests/generate_sample.py

## 🧠 Model Architecture

Input Layer    : 4 features
Hidden Layers  : 64 → 32 → 16 neurons (ReLU)
Output Layer   : 1 neuron (Sigmoid)
Regularization : Dropout layers
Optimizer      : Adam with LR scheduling

## 💾 Storage System

- **Models**: Saved in `/models` directory
- **Backups**: Stored in `/backups`
- **Database**: SQLite (`federated_learning.db`)
- **Experiments**: Logged in `/experiments`
- **Cloud Storage**: Optional, configure in `.env`

## 🔄 Federated Learning Process

1. **Client Upload**: Upload health data through web interface
2. **Local Training**: Train model on client's private data
3. **Weight Collection**: Server collects model weights from clients
4. **Aggregation**: FedAvg algorithm combines weights into global model
5. **Distribution**: Updated global model sent back to clients

## 🛡️ Privacy Features

- Data never leaves client environment
- Only model weights are shared
- No raw data transmission
- Secure weight aggregation

## 📈 Monitoring

- Real-time training metrics
- Model performance visualization
- Client comparison charts
- Global model analytics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.


