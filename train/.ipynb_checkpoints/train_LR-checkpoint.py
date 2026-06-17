from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib # Used to save your trained model

def train_and_save(X_train, y_train, save_path):
    # Standardize then train
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Save the model so you can 'pull' it later for LIME/SHAP
    joblib.dump(pipeline, save_path)
    return pipeline

    from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }