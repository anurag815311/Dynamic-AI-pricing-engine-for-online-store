



```markdown
# 🚀 Dynamic AI Pricing Engine

An intelligent pricing system that uses machine learning to dynamically adjust product prices based on demand, competition, and market trends.

---

## 📌 Project Overview

The Dynamic AI Pricing Engine predicts optimal prices by analyzing:

- Demand patterns  
- Competitor pricing  
- Seasonal trends  
- Customer behavior  

It combines data science, backend logic, and frontend visualization to simulate real-world pricing systems used in e-commerce platforms.

---

## 🧠 Key Features

- 📊 Demand prediction using XGBoost  
- 💰 Dynamic price optimization  
- 📈 Exploratory Data Analysis (EDA)  
- 🧹 Data preprocessing & feature engineering  
- 🌐 Web interface using Flask / Streamlit  
- ⚙️ Modular and scalable design  

---

## 📁 Repository Structure

```

├── dynamic-pricing/        # Test_purpose
├── dynamic_pricing/        # Streamlit(has its own README)
├── Dataset_AI_Price_Engine.xlsx
├── final_dynamic_pricing_dataset.csv
├── pricing_dataset.csv
├── EDA.ipynb
├── preprocessing.ipynb
├── model_test.ipynb
├── .gitignore

````

> Note: Detailed explanations of modules inside `dynamic_pricing/` are available in their respective READMEs.

---

## 🔄 Workflow

1. Data Generation / Collection  
   Synthetic dataset simulating products, demand, and competitor pricing  

2. Data Preprocessing  
   Encoding, feature engineering, and cleaning  

3. Exploratory Data Analysis  
   Demand trends, price distribution, correlations  

4. Model Training  
   XGBoost model for demand prediction  

5. Price Optimization  
   Price adjusted using demand + competition + margin  

6. Deployment  
   Flask backend and Streamlit frontend  

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib, Seaborn  
- Flask  
- Streamlit  
- Jupyter Notebook  

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/dynamic-ai-pricing-engine.git
cd dynamic-ai-pricing-engine
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend (Flask)

```bash
cd dynamic_pricing
python app.py
```

### 4. Run Frontend (Streamlit)

```bash
streamlit run app.py
```

---

## 📊 Datasets

* pricing_dataset.csv → Raw dataset
* final_dynamic_pricing_dataset.csv → Processed dataset
* Dataset_AI_Price_Engine.xlsx → Structured dataset

---

## 🎯 Use Cases

* E-commerce pricing optimization
* Demand forecasting
* Competitive pricing analysis
* AI-based decision systems

---

## 🔮 Future Enhancements

* Real-time competitor price scraping
* Reinforcement learning pricing model
* Database integration (MongoDB / PostgreSQL)
* User authentication system
* API-based pricing engine

---

## 👨‍💻 Author

**Anurag Singh**
MCA Student | Data Science & Full Stack Developer

---




