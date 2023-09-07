# Reviews Sentiment Classification Project
# Overview
This project is designed to classify reviews as either "Positive" or "Negative" based on their content. It includes two main components:

# Training Script (`train.py`):
This script is responsible for training a sentiment classifier model using a dataset of reviews and their corresponding sentiment labels. It utilizes the XGBoost algorithm along with Count Vectorization for feature extraction. The trained model and vectorizer are saved to disk for later use.

# Inference Script (`inference.py`):
This script is used to make predictions on new review data. It loads the pre-trained model and vectorizer and applies them to classify reviews provided in a test dataset. The predicted sentiment labels are saved to an output file.

# Prerequisites
Before running the project, ensure you have the following Python packages installed (or install them with `pip install -r requirements.txt`):
- pandas
- scikit-learn
- xgboost
# Usage
# Training Script (train.py)
To train the sentiment classifier model and create the necessary model files (`vectorizer.pkl` and `xgb_classifier.pkl`), follow these steps:

Place your training data in two separate CSV files: `reviews.csv` containing the review text and `labels.csv` containing the corresponding sentiment labels ("Positive" or "Negative").

Run the training script:

```
python train.py
```
This script will preprocess the data, train the model, and save the trained model and vectorizer to disk.

# Inference Script (inference.py)
To use the pre-trained model to classify new reviews, follow these steps:

1. Prepare a CSV file containing the test reviews that you want to classify. Each row should contain a 'text' column with the review text.

2. Run the inference script, specifying the path to the test reviews file and the desired output file for the predicted labels:

```
python inference.py test_reviews.csv test_labels_pred.csv
```
This script will load the pre-trained model and vectorizer, classify the test reviews, and save the results to the specified output file.

# Evaluation Metrics
The training script (`train.py`) includes calculations for the following evaluation metrics:

- Balanced Accuracy
- Accuracy
- Macro-average F1 Score
  
A classification report is also given.

# Data Format
- Training Data (`reviews.csv` and `labels.csv`): The training data should be in CSV format. The `reviews.csv` file should contain a 'text' column with the review text. The `labels.csv` file should contain a 'sentiment' column with sentiment labels ("Positive" or "Negative").
- Test Data (`test_reviews.csv`): The test data file should be in CSV format with a 'text' column containing the review text.

# Author
*Bohdan-Yarema Dekhtiar*
# Acknowledgments
Special thanks to It-Jim for providing the training data.
# Contact Information
For any questions or inquiries, please contact yarema.dekhtiar@gmail.com.
 
