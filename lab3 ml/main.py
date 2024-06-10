import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = pd.read_csv("WQ-R.csv", delimiter=';')


data = data.apply(pd.to_numeric, errors='coerce')


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


def visualize_tree(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=True)
    plt.show()


visualize_tree(model, X.columns)



def print_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    return accuracy, precision, recall, f1


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train metrics:")
train_metrics = print_classification_metrics(y_train, y_train_pred)
print("\nTest metrics:")
test_metrics = print_classification_metrics(y_test, y_test_pred)



def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()


plot_confusion_matrix(y_test, y_test_pred, 'Confusion Matrix for Test Data')



def compare_criteria(X_train, y_train, X_test, y_test):
    criteria = ['gini', 'entropy']
    results = {}

    for criterion in criteria:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = print_classification_metrics(y_test, y_pred)
        results[criterion] = (accuracy, precision, recall, f1)

    return results


criteria_results = compare_criteria(X_train, y_train, X_test, y_test)
print("\nCriteria comparison results:")
print(criteria_results)



def evaluate_parameters(X_train, y_train, X_test, y_test):
    depths = range(1, 11)
    min_samples = range(1, 11)
    scores = []

    for depth in depths:
        for min_sample in min_samples:
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_sample, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            scores.append((depth, min_sample, accuracy))

    scores_df = pd.DataFrame(scores, columns=['Depth', 'Min Samples Leaf', 'Accuracy'])
    return scores_df


param_results = evaluate_parameters(X_train, y_train, X_test, y_test)

plt.figure(figsize=(12, 8))
for min_sample in range(1, 11):
    subset = param_results[param_results['Min Samples Leaf'] == min_sample]
    plt.plot(subset['Depth'], subset['Accuracy'], label=f'Min Samples Leaf: {min_sample}')

plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Effect of Tree Depth and Min Samples Leaf on Accuracy')
plt.legend()
plt.show()



def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


plot_feature_importances(model, X.columns)
