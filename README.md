# Zoo Multi-Class ML Classification Task

Machine Learning project for multi-class classification of animals using the Zoo Dataset.

## Overview

This project aims to build a machine learning model capable of predicting the class type of an animal from biological features.

The problem is formulated as a **multi-class classification task** with 7 classes:

- **Class 1 (Mammals)**
- **Class 2 (Birds)**
- **Class 3 (Reptiles)**
- **Class 4 (Fish)**
- **Class 5 (Amphibians)**
- **Class 6 (Insects)**
- **Class 7 (Invertebrates)**

Due to the class imbalance present in the dataset, special attention was given to model evaluation and selection.

---

## Dataset

The project uses the **Zoo Dataset**, composed of:

- 101 samples
- 16 input features (15 boolean, 1 numeric)
- 1 multi-class target variable

### Features

| Feature | Description | Admissible Values |
|----------|------------|-------------------|
| hair | Presence of hair | Boolean (0, 1) |
| feathers | Presence of feathers | Boolean (0, 1) |
| eggs | Lays eggs | Boolean (0, 1) |
| milk | Produces milk | Boolean (0, 1) |
| airborne | Is airborne | Boolean (0, 1) |
| aquatic | Is aquatic | Boolean (0, 1) |
| predator | Is a predator | Boolean (0, 1) |
| toothed | Has teeth | Boolean (0, 1) |
| backbone | Has a backbone | Boolean (0, 1) |
| breathes | Breathes air | Boolean (0, 1) |
| venomous | Is venomous | Boolean (0, 1) |
| fins | Has fins | Boolean (0, 1) |
| legs | Number of legs | Numeric (0, 2, 4, 5, 6, 8) |
| tail | Has a tail | Boolean (0, 1) |
| domestic | Is domestic | Boolean (0, 1) |
| catsize | Is cat-sized or larger | Boolean (0, 1) |

Target:

- **type** = Numeric (integer values in range [1,7]) representing the class type of the animal.

---

## Class Imbalance

The dataset is highly imbalanced:

- Class 1: ~40.6% (41 instances)
- Class 2: ~19.8% (20 instances)
- Class 3: ~5.0% (5 instances)
- Class 4: ~12.9% (13 instances)
- Class 5: ~4.0% (4 instances)
- Class 6: ~7.9% (8 instances)
- Class 7: ~9.9% (10 instances)

Because of this, standard accuracy alone is not an appropriate metric.

The following strategies were adopted:

- Stratified train/test split
- Balanced Accuracy as the primary evaluation metric
- `class_weight='balanced'` where supported

---

## Data Preprocessing

### Train/Test Split

- 70% Training Set (70 samples)
- 30% Test Set (31 samples)
- `random_state=0`
- Stratified sampling

Using a 70/30 split ensures a larger test set compared to the initial 75/25 split, providing a more realistic and robust evaluation of model generalization.

### Feature Scaling

Features are standardized using:

```python
StandardScaler()
```

The scaler is fitted only on the training set to avoid data leakage.

---

## Models Evaluated

Four machine learning models were compared using Grid Search with 3-Fold Cross Validation. The choice of cv=3 is dictated by the class sizes (minority classes have exactly 3 samples in the training set), ensuring that every CV fold contains at least one sample of each minority class for stable evaluation metrics.

### 1. Logistic Regression

Hyperparameters:

- Penalty: L1, L2
- C ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1}

Best configuration:

```text
C = 5e-05
penalty = l2
```

Best CV Balanced Accuracy:

```text
0.860
```

---

### 2. Support Vector Machine (SVM)

Hyperparameters:

- Kernel: Linear, RBF
- C ∈ {0.1, 1, 10, 100}
- Gamma ∈ {scale, auto, 0.01, 0.1}

Best configuration:

```text
kernel = rbf
C = 1
gamma = 0.01
```

Best CV Balanced Accuracy:

```text
0.884
```

---

### 3. Random Forest

Hyperparameters:

- n_estimators ∈ {50, 100, 150}
- max_depth ∈ {3, 5, 7, None}
- min_samples_split ∈ {2, 5}

Best configuration:

```text
n_estimators = 100
max_depth = 3
min_samples_split = 2
```

Best CV Balanced Accuracy:

```text
0.865
```

---

### 4. Multi-Layer Perceptron (MLP)

Hyperparameters:

- hidden_layer_sizes ∈ {n/2, n, 2n} (where n = 16)
- alpha ∈ {0.0001, 0.001, 0.01}
- learning_rate_init ∈ {0.001, 0.01, 0.1}

Best configuration:

```text
hidden_layer_sizes = 16 (or 8)
alpha = 0.01
learning_rate_init = 0.1
```

Best CV Balanced Accuracy:

```text
0.884
```

---

## Model Selection

Comparison of cross-validation results:

| Model | Balanced Accuracy |
|---------|---------|
| SVM | 0.884 |
| MLP | 0.884 |
| Random Forest | 0.865 |
| Logistic Regression | 0.860 |

**Justification:**
Both SVM and MLP achieved the highest validation performance of 0.884. We selected the **Support Vector Machine (SVM)** because it is a simpler model with fewer parameters compared to the MLP, reducing the risk of overfitting on a small dataset (70 training samples).

---

## Final Test Results

Test set size:

```text
31 samples
```

Classification Report:

| Class | Precision | Recall | F1-score | Support |
|---------|---------|---------|---------|---------|
| Class 1 | 1.00 | 1.00 | 1.00 | 13 |
| Class 2 | 1.00 | 1.00 | 1.00 | 6 |
| Class 3 | 1.00 | 0.50 | 0.67 | 2 |
| Class 4 | 1.00 | 1.00 | 1.00 | 4 |
| Class 5 | 0.50 | 1.00 | 0.67 | 1 |
| Class 6 | 1.00 | 1.00 | 1.00 | 2 |
| Class 7 | 1.00 | 1.00 | 1.00 | 3 |

Overall metrics:

```text
Accuracy           = 0.968
Balanced Accuracy  = 0.929
```

Confusion Matrix:

```text
[[13  0  0  0  0  0  0]
 [ 0  6  0  0  0  0  0]
 [ 0  0  1  0  1  0  0]
 [ 0  0  0  4  0  0  0]
 [ 0  0  0  0  1  0  0]
 [ 0  0  0  0  0  2  0]
 [ 0  0  0  0  0  0  3]]
```

Exactly 1 sample belonging to Class 3 (Reptiles) was misclassified as Class 5 (Amphibians).

### Overfitting/Underfitting Analysis

- **Underfitting:** The validation performance (Balanced Accuracy = 0.884) is high, indicating that the models are not underfitting.
- **Overfitting:** The test set performance (Accuracy = 0.968, Balanced Accuracy = 0.929) remains high and is well-aligned with validation metrics. Since test accuracy does not degrade, there is no evidence of overfitting. The high accuracy is expected given the clean separability of class features in the Zoo dataset.

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- FPDF2 (for PDF report generation)

Main Scikit-Learn components:

- StandardScaler
- GridSearchCV
- LogisticRegression
- SVC
- RandomForestClassifier
- MLPClassifier

---

## Project Structure

```text
.
├── Index
├── ML_Zoo_MultiClass_Classification_Task.pdf
├── README.md
├── classification_zoo.py
├── correlation_matrix.png
├── generate_report_pdf.py
├── train_zoo.py
├── zoo.data
└── zoo.names
```

---

## How to Run

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn fpdf2
```

Run the training script:

```bash
python train_zoo.py
```

The script will:

1. Load the dataset
2. Split train/test data
3. Scale features
4. Perform Grid Search on all models
5. Select the best model
6. Train the final classifier
7. Evaluate on the test set

---

## Conclusions

Despite the small dataset size and class imbalance, animal traits provide a highly discriminative feature space, allowing for robust classification.

Among the tested approaches, an **RBF Support Vector Machine** provided the best balance, achieving a **Balanced Accuracy of 0.929** on the test set.

---

## Author

**Daniele Primavera**

Bachelor's Degree in Computer Engineering  
University of Modena and Reggio Emilia (UNIMORE)