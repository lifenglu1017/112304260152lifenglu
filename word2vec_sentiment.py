import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
negative_words = {'not', 'no', 'never', 'nor', 'none', 'nobody', 'nowhere', 'nothing', 'neither', 'hardly', 'scarcely', 'barely', 'seldom'}
stop_words = stop_words - negative_words

def clean_text(text):
    text = re.sub(r'<br />', ' ', text)
    text = text.lower()
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return words

print("Loading labeled training data...")
labeled_train = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)

print("Loading unlabeled training data...")
try:
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', quoting=3)
except FileNotFoundError:
    try:
        unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', quoting=3)
    except FileNotFoundError:
        print("Warning: unlabeledTrainData.tsv not found, using only labeled data")
        unlabeled_train = pd.DataFrame()

print("Loading test data...")
try:
    test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)
except FileNotFoundError:
    try:
        test_data = pd.read_csv('testData.tsv', sep='\t', quoting=3)
    except FileNotFoundError:
        print("Warning: testData.tsv not found, will create dummy test data")
        test_data = labeled_train.sample(1000).copy()
        test_data['sentiment'] = test_data['sentiment'].copy()
        test_data = test_data.drop('sentiment', axis=1)

print("Preprocessing text...")
labeled_train['clean_review'] = labeled_train['review'].apply(clean_text)
if not unlabeled_train.empty:
    unlabeled_train['clean_review'] = unlabeled_train['review'].apply(clean_text)
test_data['clean_review'] = test_data['review'].apply(clean_text)

print("Training Word2Vec model...")
all_reviews = list(labeled_train['clean_review'])
if not unlabeled_train.empty:
    all_reviews.extend(list(unlabeled_train['clean_review']))

w2v_model = Word2Vec(
    sentences=all_reviews,
    vector_size=300,
    window=5,
    min_count=5,
    sg=0,
    epochs=10,
    workers=4,
    seed=42
)

def get_sentence_vector(words):
    vectors = []
    for word in words:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

print("Generating sentence vectors...")
X_train = np.array([get_sentence_vector(words) for words in labeled_train['clean_review']])
y_train = labeled_train['sentiment'].values
X_test = np.array([get_sentence_vector(words) for words in test_data['clean_review']])

print("Training Logistic Regression model...")
lr_model = LogisticRegression(
    C=4.0,
    max_iter=2000,
    random_state=42,
    solver='liblinear'
)

print("Cross-validation AUC score:")
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Mean AUC: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")

lr_model.fit(X_train, y_train)

print("Predicting on test set...")
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

print("Generating submission file...")
submission = pd.DataFrame({
    'id': test_data['id'].str.replace('"', ''),
    'sentiment': y_pred_proba
})
submission.to_csv('submission.csv', index=False, quoting=3)

print("Done! Submission file saved as submission.csv")
print(f"Sample predictions:\n{submission.head(10)}")