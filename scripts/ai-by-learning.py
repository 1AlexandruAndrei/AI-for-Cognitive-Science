import random
import numpy as np
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
    return accuracy, predictions


def visualize_data_distribution(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(np.sum(X, axis=1), y, alpha=0.5, c=y, cmap='bwr', label="Data points")
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1, label="Decision Boundary")
    plt.title(title)
    plt.xlabel("Sum of Features")
    plt.ylabel("Output Label (y)")
    plt.legend()
    plt.show()

def visualize_model_performance(accuracy_perfect, accuracy_chance):
    plt.figure(figsize=(8, 6))
    categories = ["Perfect Case", "Chance Case"]
    accuracies = [accuracy_perfect, accuracy_chance]
    plt.bar(categories, accuracies, color=['blue', 'red'])
    plt.title("Model Performance Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

def test_ai_by_learning():
    num_samples = 1000
    num_features = 10

    X_perfect, y_perfect = generate_data(num_samples, num_features, perfect_case=True)
    X_train, X_test, y_train, y_test = train_test_split(X_perfect, y_perfect, test_size=0.3, random_state=42)

    model_perfect = train_ai(X_train, y_train)
    accuracy_perfect, predictions_perfect = evaluate_ai(model_perfect, X_test, y_test)

    print(f"Accuratețea în cazul Perfect: {accuracy_perfect * 100:.2f}%")
    visualize_data_distribution(X_perfect, y_perfect, "Distribuția Datelor - Cazul Perfect")
    ConfusionMatrixDisplay.from_estimator(model_perfect, X_test, y_test)
    plt.title("Confusion Matrix - Cazul Perfect")
    plt.show()

    print("\nPredicții pentru cazul Perfect (primele 10 exemple):")
    for i in range(10):
        print(f"Exemplu {i + 1}: Predicție = {predictions_perfect[i]}, Real = {y_test[i]}")

    plt.figure(figsize=(8, 6))
    plt.hist(predictions_perfect, bins=2, color='blue', alpha=0.7, rwidth=0.8)
    plt.title("Distribuția Predicțiilor - Cazul Perfect")
    plt.xlabel("Predicție (0 sau 1)")
    plt.ylabel("Frecvență")
    plt.show()

    X_chance, y_chance = generate_data(num_samples, num_features, perfect_case=False)
    X_train, X_test, y_train, y_test = train_test_split(X_chance, y_chance, test_size=0.3, random_state=42)

    model_chance = train_ai(X_train, y_train)
    accuracy_chance, predictions_chance = evaluate_ai(model_chance, X_test, y_test)

    print(f"Accuratețea în cazul Chance: {accuracy_chance * 100:.2f}%")
    visualize_data_distribution(X_chance, y_chance, "Distribuția Datelor - Cazul Chance")
    ConfusionMatrixDisplay.from_estimator(model_chance, X_test, y_test)
    plt.title("Confusion Matrix - Cazul Chance")
    plt.show()

    print("\nPredicții pentru cazul Chance (primele 10 exemple):")
    for i in range(10):
        print(f"Exemplu {i + 1}: Predicție = {predictions_chance[i]}, Real = {y_test[i]}")

    plt.figure(figsize=(8, 6))
    plt.hist(predictions_chance, bins=2, color='red', alpha=0.7, rwidth=0.8)
    plt.title("Distribuția Predicțiilor - Cazul Chance")
    plt.xlabel("Predicție (0 sau 1)")
    plt.ylabel("Frecvență")
    plt.show()

    visualize_model_performance(accuracy_perfect, accuracy_chance)



test_ai_by_learning()
