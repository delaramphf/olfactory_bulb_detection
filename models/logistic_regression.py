from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_logistic_regression(X, Y, C=100, max_iter=1000000, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_state)
    lr_clf = LogisticRegression(penalty='l2', C=C, max_iter=max_iter, solver='lbfgs', multi_class='ovr', random_state=random_state)
    lr_clf.fit(X_train, y_train)
    return lr_clf, X_train, X_test, y_train, y_test

def evaluate_logistic_regression(lr_clf, X_test, y_test):
    y_pred = lr_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

def logistic_regression_pipeline(X, Y, C=100, max_iter=1000000, random_state=0):
    lr_clf, X_train, X_test, y_train, y_test = train_logistic_regression(X, Y, C, max_iter, random_state)
    accuracy, conf_matrix, class_report = evaluate_logistic_regression(lr_clf, X_test, y_test)
    cv_scores = cross_val_score(lr_clf, X, Y, cv=5)
    
    return {
        'classifier': lr_clf,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'cv_scores': cv_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
