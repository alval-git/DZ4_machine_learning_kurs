from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import  GridSearchCV
from sklearn.utils import compute_sample_weight
# Настройка GridSearchCV


def DecisionTreeClassifierModel(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def CatBoostClassifierModel(X_train, y_train, X_test, y_test):
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 5, 10],
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=1
                               )
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Accuracy на тестовых данных: {accuracy:.2f}')
    print('Classification Report:')
    print(report)