import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def generate_data(num_samples, num_features, perfect_case=True):
    X = np.random.rand(num_samples, num_features)
    if perfect_case:
        # Perfect: ieșirea este o funcție liniară de intrare
        y = (np.sum(X, axis=1) > num_features / 2).astype(int)
    else:
        # Chance: ieșirea este aleatorie
        y = np.random.choice([0, 1], size=num_samples)
    return X, y


def train_ai(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_ai(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def test_ai_by_learning():
    num_samples = 10000
    num_features = 20

    X_perfect, y_perfect = generate_data(num_samples, num_features, perfect_case=True)
    X_train, X_test, y_train, y_test = train_test_split(X_perfect, y_perfect, test_size=0.3, random_state=42)

    model_perfect = train_ai(X_train, y_train)
    accuracy_perfect = evaluate_ai(model_perfect, X_test, y_test)
    print(f"Accuratețea în cazul Perfect: {accuracy_perfect * 100:.2f}%")


    X_chance, y_chance = generate_data(num_samples, num_features, perfect_case=False)
    X_train, X_test, y_train, y_test = train_test_split(X_chance, y_chance, test_size=0.3, random_state=42)

    model_chance = train_ai(X_train, y_train)
    accuracy_chance = evaluate_ai(model_chance, X_test, y_test)
    print(f"Accuratețea în cazul Chance: {accuracy_chance * 100:.2f}%")

    print("\nConcluzie:")
    if accuracy_perfect > 0.9 and accuracy_chance < 0.6:
        print("AI-ul poate învăța în cazul Perfect, dar nu poate în cazul Chance.")
    else:
        print("Performanța AI-ului nu respectă diferențele așteptate între Perfect și Chance.")

# Rulează testul
test_ai_by_learning()
