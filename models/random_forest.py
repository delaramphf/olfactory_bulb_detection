import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_random_forest(X, Y, max_depth=10, n_estimators=100, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_state)
    rf_clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    rf_clf.fit(X_train, y_train)
    return rf_clf, X_train, X_test, y_train, y_test

def get_feature_importance(rf_clf):
    importance = np.zeros(rf_clf.n_features_in_)
    for tree in rf_clf.estimators_:
        importance += tree.feature_importances_
    return importance / len(rf_clf.estimators_)

def evaluate_random_forest(rf_clf, X_test, y_test):
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

def perform_cross_validation(rf_clf, X, Y, cv=5):
    return cross_val_score(rf_clf, X, Y, cv=cv)

def get_top_features(importance, feature_names, n_top_features=20):
    idx = np.argsort(importance)[::-1]
    return [(feature_names[i], importance[i]) for i in idx[:n_top_features]]

def random_forest_pipeline(X, Y, max_depth=10, n_estimators=100, random_state=0):
    rf_clf, X_train, X_test, y_train, y_test = train_random_forest(X, Y, max_depth, n_estimators, random_state)
    importance = get_feature_importance(rf_clf)
    accuracy, conf_matrix, class_report = evaluate_random_forest(rf_clf, X_test, y_test)
    cv_scores = perform_cross_validation(rf_clf, X, Y)
    
    return {
        'classifier': rf_clf,
        'feature_importance': importance,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'cv_scores': cv_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
