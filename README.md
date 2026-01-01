# Explainable AI (XAI) with LIME & SHAP

**Copyright (c) 2026 Shrikara Kaudambady**

This project provides a Jupyter notebook demonstrating how to use popular Explainable AI (XAI) techniques to interpret the predictions of a machine learning model. Specifically, it uses **LIME (Local Interpretable Model-agnostic Explanations)** and **SHAP (SHapley Additive exPlanations)** on a Random Forest Classifier trained on the classic Iris dataset.

## Solution Explanation

In modern machine learning, it's often not enough to simply have a model that makes accurate predictions. We also need to understand *why* it makes those predictions. This is crucial for debugging, ensuring fairness, building trust, and complying with regulations. This notebook tackles this challenge using two powerful XAI libraries.

### 1. The Model
The notebook starts by training a `RandomForestClassifier` on the Iris dataset. This model serves as the "black box" that we want to interpret. While a Random Forest is not the most complex model, the principles shown here apply to any model, including deep neural networks.

### 2. LIME (Local Interpretable Model-agnostic Explanations)
LIME answers the question: "Why did the model make this specific prediction for this single data point?"

- **How it Works:** LIME works by creating a simpler, interpretable "surrogate" model (like a weighted linear regression) in the local vicinity of the prediction we want to explain. It generates a new dataset of small variations (perturbations) of the instance, gets the model's predictions for them, and uses this information to learn the local behavior of the model.
- **In the Notebook:** We use LIME to explain a single prediction for an Iris flower. The output is a simple, intuitive bar chart showing which feature values pushed the prediction towards a specific class (e.g., "petal width <= 1.55 cm was the most important factor for predicting 'versicolor'").

### 3. SHAP (SHapley Additive exPlanations)
SHAP is a more advanced, game theory-based approach to explainability. It calculates "Shapley values" for each feature, which represent the feature's average marginal contribution to the prediction across all possible feature combinations.

- **How it Works:** SHAP provides a mathematically sound way to fairly distribute the prediction outcome among the features. It guarantees properties like local accuracy (the sum of feature contributions equals the prediction) and consistency.
- **In the Notebook:**
    - **Global Explanations:** We generate a SHAP summary plot. This plot provides a powerful overview of the entire model, showing the most important features and the distribution of their impact. For example, it might show that `petal width` and `petal length` are the most significant features overall.
    - **Local Explanations:** We use a SHAP "force plot" to explain the same instance we analyzed with LIME. This plot provides a more detailed and interactive visualization, showing features as "forces" that either push the prediction higher (red) or lower (blue).

## How to Use

Follow these steps to run the notebook and explore the XAI techniques.

### 1. Clone the Repository
If this project is on a Git repository, clone it to your local machine:
```bash
git clone <repository-url>
cd gemini-cli-projects/xai-explainable-ai-notebook
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
Start the Jupyter Notebook server.
```bash
jupyter notebook
```
This will open a new tab in your web browser.

### 5. Run the Notebook
In the browser, click on the `explainable_ai_with_lime_shap.ipynb` file to open it. You can then run the cells sequentially to see the model training and explanation generation process.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
