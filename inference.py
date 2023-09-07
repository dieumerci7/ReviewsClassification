import argparse
import pickle
import pandas as pd


def main(test_reviews_file, test_labels_pred_file):
    test_reviews = pd.read_csv(test_reviews_file)

    with open('vectorizer.pkl', 'rb') as model_file:
        loaded_vectorizer = pickle.load(model_file)
    with open('xgb_classifier.pkl', 'rb') as model_file:
        loaded_xgb_classifier = pickle.load(model_file)

    X_test = test_reviews['text']
    y_pred = loaded_xgb_classifier.predict(loaded_vectorizer.transform(X_test))

    test_labels_pred = pd.DataFrame({'sentiment': y_pred})
    mapping = {1: 'Positive', 0: 'Negative'}
    test_labels_pred['sentiment'] = test_labels_pred['sentiment'].map(mapping)
    test_labels_pred.to_csv(test_labels_pred_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("test_reviews_file", help="Path to the test reviews file")
    parser.add_argument("test_labels_pred_file", help="Path to the test labels prediction file")
    args = parser.parse_args()

    main(args.test_reviews_file, args.test_labels_pred_file)
