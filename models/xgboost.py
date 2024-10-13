from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_xgboost(X, Y, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_state)
    xgb_clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf, X_train, X_test, y_train, y_test

def evaluate_xgboost(xgb_clf, X_test, y_test):
    y_pred = xgb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

def xgboost_pipeline(X, Y, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0):
    xgb_clf, X_train, X_test, y_train, y_test = train_xgboost(X, Y, n_estimators, learning_rate, max_depth, random_state)
    accuracy, conf_matrix, class_report = evaluate_xgboost(xgb_clf, X_test, y_test)
    cv_scores = cross_val_score(xgb_clf, X, Y, cv=5)
    
    return {
        'classifier': xgb_clf,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'cv_scores': cv_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
