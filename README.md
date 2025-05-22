
# DDPG-ALM: Portfolio Optimization Using Deep Reinforcement Learning

This repository presents an implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm enhanced with the Augmented Lagrangian Multiplier (ALM) method. The primary objective is to optimize financial portfolios by maximizing returns while adhering to constraints such as risk limits and transaction costs.

---

## 🧠 Overview

Traditional portfolio optimization techniques often struggle with dynamic market conditions and complex constraints. By leveraging reinforcement learning, specifically the DDPG algorithm, this project aims to create an adaptive agent capable of making continuous investment decisions. The integration of the ALM method ensures that the agent respects predefined constraints during the optimization process.

---

## 📂 Repository Structure

```text
.
├── Cmdp+_ddpg+alm.ipynb              # Main implementation notebook
├── app.py                           # Flask application for deployment
├── rewards_plot.png                 # Training rewards visualization
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup script
├── react-app/                       # React frontend application
├── src/                             # Frontend source files
└── static/                          # Static assets (CSS, JS, etc.)
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7 or above
- pip (Python package manager)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Prabhat-Kr-Sahu/DDPG-ALM.git
cd DDPG-ALM
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Flask app**

```bash
python app.py
```

Visit `http://localhost:5000/` in your browser to access the application interface.

---

## 🧪 Usage

- Use the Jupyter notebooks to explore how DDPG-ALM is trained and how constraints are handled.
- Launch the web app (`app.py`) to visualize agent behavior and evaluate its performance.

---

## 📊 Results

The training rewards over episodes are visualized in `rewards_plot.png`. This gives insight into how the agent improves its strategy over time.

---

## 🧠 Technical Highlights

- **Reinforcement Learning**: Deep Deterministic Policy Gradient (DDPG)
- **Constraints Handling**: Augmented Lagrangian Multiplier (ALM) Method
- **Continuous Action Space**: Ideal for portfolio allocation tasks
- **Flask + React Integration**: Seamless backend-frontend workflow for visualization

---

## 🤝 Contributing

Contributions are welcome! If you’d like to improve the project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📫 Contact

For feedback, suggestions, or collaboration:

- GitHub: [Prabhat Kumar Sahu](https://github.com/Prabhat-Kr-Sahu) and [Honey Gupta](https://github.com/honey1205)
- Email: *[b23281@students.iitmandi.ac.in]

---

> ⚡ Made with passion for intelligent finance and algorithmic strategies.
