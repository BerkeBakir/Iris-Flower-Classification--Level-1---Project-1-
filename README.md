# Iris Species Classifier (KNN Implementation)

This project is an interactive Machine Learning application that identifies the species of an Iris flower based on its physical measurements. It utilizes the **K-Nearest Neighbors (KNN)** algorithm to classify samples into one of three categories: *Setosa*, *Versicolor*, or *Virginica*.

## 🧠 Model Architecture
The system follows a standard supervised learning pipeline:
- **Dataset:** Scikit-learn's built-in Iris dataset (150 samples, 4 features).
- **Algorithm:** K-Nearest Neighbors ($k=3$).
- **Data Split:** 80% Training, 20% Testing to evaluate performance on unseen data.
- **Evaluation:** Accuracy score calculation using Scikit-learn metrics.

## ✨ Key Features
- **Exploratory Data Analysis (EDA):** Visualizes the distribution of species based on sepal length and width using **Seaborn** scatter plots.
- **Interactive Prediction System:** A real-time command-line interface that allows users to input their own measurements (Sepal/Petal length and width) to get instant classification results.
- **Robust Error Handling:** Validates user inputs to ensure only numerical values are processed, preventing system crashes during live predictions.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** 
  - `Pandas` & `NumPy` (Data Manipulation)
  - `Scikit-learn` (Machine Learning & Datasets)
  - `Matplotlib` & `Seaborn` (Data Visualization)

## 📂 Installation & Usage
### Prerequisites
- Python 3.x installed.
- Required libraries: `pip install pandas seaborn scikit-learn matplotlib numpy`

### Execution
Run the script to train the model and start the interactive predictor:
```bash
python iris_classifier.py
