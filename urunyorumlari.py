import os
import math
import numpy as np
import csv
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def write_file(file_path, reviews):
    with open(file_path, 'w', encoding='utf-8') as output_file:
        for review in reviews:
            output_file.write(review + '\n')

def write_csv(file_path, headers, rows):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

def write_txt(file_path, headers, rows):
    with open(file_path, 'w', encoding='utf-8') as txtfile:
        txtfile.write('\t'.join(headers) + '\n')
        for row in rows:
            txtfile.write('\t'.join(map(str, row)) + '\n')

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        rows = [list(map(float, row)) for row in reader]
    return headers, rows

def split_into_k_folds(reviews, k):
    kf = KFold(n_splits=k)
    folds = []
    for train_index, test_index in kf.split(reviews):
        train_folds = [reviews[i] for i in train_index]
        test_folds = [reviews[i] for i in test_index]
        folds.append((train_folds, test_folds))
    return folds

def create_folds_directories(base_path, k):
    for i in range(k):
        fold_path = os.path.join(base_path, f'fold_{i+1}')
        os.makedirs(fold_path, exist_ok=True)

def save_folds(folds, base_path, label):
    for i, (train, test) in enumerate(folds):
        fold_path = os.path.join(base_path, f'fold_{i+1}')
        train_path = os.path.join(fold_path, f'train_{label}.txt')
        test_path = os.path.join(fold_path, f'test_{label}.txt')
        write_file(train_path, train)
        write_file(test_path, test)

def compute_tfidf(reviews):
    term_freq = {}
    doc_freq = Counter()
    num_docs = len(reviews)

    for review in reviews:
        terms = review.split()
        term_counts = Counter(terms)
        for term in term_counts.keys():
            if term not in term_freq:
                term_freq[term] = 0
            term_freq[term] += 1
        doc_freq.update(set(terms))

    tfidf = {}
    for term in term_freq.keys():
        idf = math.log10(num_docs / doc_freq[term])
        tfidf[term] = idf

    return tfidf

def compute_tfidf_per_review(reviews, tfidf_scores):
    review_tfidf_values = []
    all_terms = sorted(tfidf_scores.keys())

    for review in reviews:
        terms = review.split()
        tfidf_vector = [tfidf_scores[term] if term in terms else 0.0 for term in all_terms]
        review_tfidf_values.append(tfidf_vector)

    return all_terms, review_tfidf_values

positive_reviews_file_path = "pozitif_yorumlar.txt"
negative_reviews_file_path = "negatif_yorumlar.txt"
folds_base_path = "folds"

positive_reviews = read_file(positive_reviews_file_path)
negative_reviews = read_file(negative_reviews_file_path)

k = 5
positive_folds = split_into_k_folds(positive_reviews, k)
negative_folds = split_into_k_folds(negative_reviews, k)

create_folds_directories(folds_base_path, k)

save_folds(positive_folds, folds_base_path, 'positive')

save_folds(negative_folds, folds_base_path, 'negative')

all_reviews = positive_reviews + negative_reviews

tfidf_scores = compute_tfidf(all_reviews)
print("TF-IDF scores calculated")

for fold_index in range(k):
    fold_num = fold_index + 1
    fold_path = os.path.join(folds_base_path, f'fold_{fold_num}')
    
    train_reviews = positive_folds[fold_index][0] + negative_folds[fold_index][0]
    test_reviews = positive_folds[fold_index][1] + negative_folds[fold_index][1]
    
    train_labels = ['positive'] * len(positive_folds[fold_index][0]) + ['negative'] * len(negative_folds[fold_index][0])
    test_labels = ['positive'] * len(positive_folds[fold_index][1]) + ['negative'] * len(negative_folds[fold_index][1])
    
    train_terms, train_tfidf_values = compute_tfidf_per_review(train_reviews, tfidf_scores)
    test_terms, test_tfidf_values = compute_tfidf_per_review(test_reviews, tfidf_scores)
    
    train_tfidf_values_with_labels = [tfidf + [1.0 if label == 'positive' else 0.0] for tfidf, label in zip(train_tfidf_values, train_labels)]
    test_tfidf_values_with_labels = [tfidf + [1.0 if label == 'positive' else 0.0] for tfidf, label in zip(test_tfidf_values, test_labels)]
    
    csv_headers = train_terms + ['class_label']
    train_csv_path = os.path.join(fold_path, 'train.csv')
    test_csv_path = os.path.join(fold_path, 'test.csv')
    write_csv(train_csv_path, csv_headers, train_tfidf_values_with_labels)
    write_csv(test_csv_path, csv_headers, test_tfidf_values_with_labels)

print("CSV files for all folds have been created.")

feature_counts = [250, 500, 1000, 2500, 5000]
for fold_index in range(k):
    fold_num = fold_index + 1
    fold_path = os.path.join(folds_base_path, f'fold_{fold_num}')
    
    train_csv_path = os.path.join(fold_path, 'train.csv')
    headers, train_data = read_csv(train_csv_path)
    
    X_train, y_train = np.array(train_data)[:, :-1], np.array(train_data)[:, -1]
    
    for count in feature_counts:
        selector = SelectKBest(score_func=chi2, k=count)
        X_new = selector.fit_transform(X_train, y_train)
        
        selected_indices = selector.get_support(indices=True)
        selected_headers = [headers[i] for i in selected_indices] + ['class_label']
        
        reduced_train_data = np.hstack((X_new, y_train.reshape(-1, 1)))
        
        reduced_train_txt_path = os.path.join(fold_path, f'train_{count}_features.txt')
        write_txt(reduced_train_txt_path, selected_headers, reduced_train_data)

print("Reduced feature text files have been created.")

accuracies = []
f1_scores = []

for fold_index in range(k):
    fold_num = fold_index + 1
    fold_path = os.path.join(folds_base_path, f'fold_{fold_num}')
    
    train_csv_path = os.path.join(fold_path, 'train.csv')
    test_csv_path = os.path.join(fold_path, 'test.csv')
    
    headers, train_data = read_csv(train_csv_path)
    _, test_data = read_csv(test_csv_path)
    
    X_train, y_train = np.array(train_data)[:, :-1], np.array(train_data)[:, -1]
    X_test, y_test = np.array(test_data)[:, :-1], np.array(test_data)[:, -1]
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracies.append(accuracy)
    f1_scores.append(f1)
    
    print(f'Fold {fold_num} - Accuracy: {accuracy}, F1-score: {f1}')

overall_accuracy = np.mean(accuracies)
overall_f1 = np.mean(f1_scores)

print(f'Overall Accuracy: {overall_accuracy}')
print(f'Overall F1-score: {overall_f1}')
