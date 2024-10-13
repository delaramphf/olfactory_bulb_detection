from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_svm(X, Y, C=100000, kernel='linear', random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_state)
    svm_clf = SVC(C=C, kernel=kernel, random_state=random_state)
    svm_clf.fit(X_train, y_train)
    return svm_clf, X_train, X_test, y_train, y_test

def evaluate_svm(svm_clf, X_test, y_test):
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

def svm_pipeline(X, Y, C=100000, kernel='linear', random_state=0):
    svm_clf, X_train, X_test, y_train, y_test = train_svm(X, Y, C, kernel, random_state)
    accuracy, conf_matrix, class_report = evaluate_svm(svm_clf, X_test, y_test)
    cv_scores = cross_val_score(svm_clf, X, Y, cv=5)
    
    return {
        'classifier': svm_clf,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'cv_scores': cv_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }