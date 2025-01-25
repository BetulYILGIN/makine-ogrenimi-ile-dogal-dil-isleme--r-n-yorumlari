# Text Classification with K-Fold Cross Validation and TF-IDF

This project implements a text classification system that analyzes positive and negative reviews using machine learning techniques. The pipeline includes data preprocessing, TF-IDF feature extraction, feature selection, and model evaluation using K-Fold Cross Validation.

## Features
- **Data Preprocessing:** Reads positive and negative reviews from text files and splits them into training and test sets.
- **TF-IDF Feature Extraction:** Computes term importance using Term Frequency-Inverse Document Frequency (TF-IDF) scores.
- **Feature Selection:** Reduces the dimensionality of the feature space using the Chi-Square method.
- **Model Training and Evaluation:** Trains a Logistic Regression model using K-Fold Cross Validation and evaluates its performance based on accuracy and F1-score.

---

## Project Structure
```
.
├── pozitif_yorumlar.txt          # Positive reviews
├── negatif_yorumlar.txt          # Negative reviews
├── folds/                        # Directory to save K-Fold data splits and results
├── main.py                       # Main script for text classification
└── README.md                     # Project documentation
```

---

## Workflow

### 1. Data Preparation
- Reads positive and negative reviews from `pozitif_yorumlar.txt` and `negatif_yorumlar.txt`.
- Splits the data into `k` folds using K-Fold Cross Validation.
- Saves the training and test datasets for each fold in separate directories.

### 2. TF-IDF Feature Extraction
- Calculates TF-IDF scores for all terms across the entire dataset.
- Generates TF-IDF vectors for each review.

### 3. Feature Selection
- Uses the Chi-Square method to select the most significant features for classification.
- Supports feature counts of 250, 500, 1000, 2500, and 5000.

### 4. Model Training and Evaluation
- Trains a Logistic Regression model on the training data for each fold.
- Evaluates the model's performance using accuracy and F1-score on the test data.
- Outputs fold-wise and overall performance metrics.

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Ensure that the following Python libraries are installed:
   ```bash
   pip install numpy scikit-learn
   ```

3. Place your `pozitif_yorumlar.txt` and `negatif_yorumlar.txt` files in the project directory.

4. Run the main script:
   ```bash
   python main.py
   ```

5. Results will be saved in the `folds/` directory.

---

## Results
- **Performance Metrics:**
  - Accuracy and F1-score are calculated for each fold.
  - The overall performance is summarized by averaging metrics across all folds.

- **Feature Reduction:**
  - Reduced feature datasets are saved in `folds/` with varying feature counts (e.g., `train_250_features.txt`).

---

## Dependencies
- Python 3.7+
- NumPy
- scikit-learn
- csv (built-in)
- os (built-in)

---

## Example Output
```
TF-IDF scores calculated
Fold 1 - Accuracy: 0.85, F1-score: 0.84
Fold 2 - Accuracy: 0.88, F1-score: 0.87
Fold 3 - Accuracy: 0.86, F1-score: 0.85
...
Overall Accuracy: 0.86
Overall F1-score: 0.85
```

---

## Contribution
Feel free to open issues or submit pull requests to improve the project!

---

## License
This project is open-source and available under the [MIT License](LICENSE).
