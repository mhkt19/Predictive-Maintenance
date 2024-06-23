import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from datetime import datetime
from statistics import mean

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocess the data: encode categorical variables and split into features and targets."""
    print("Columns in the dataset:", data.columns)
    data['Product ID'] = data['Product ID'].astype('category').cat.codes
    data['Type'] = data['Type'].astype('category').cat.codes
    features = data.drop(columns=['UDI', 'Target', 'Failure Type'])
    features.columns = [str(col) for col in features.columns]  # Ensure feature names are strings
    target_failure = data['Target']
    target_failure_type = data['Failure Type']
    return features, target_failure, target_failure_type

def split_data(features, target):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Train and evaluate the given model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Additional evaluation metrics
    train_acc = round(model.score(X_train, y_train) * 100, 2)
    test_acc = round(model.score(X_test, y_test) * 100, 2)

    return y_pred, cm, accuracy, mcc, report, train_acc, test_acc

def save_results(folder_path, model_name, report_failure, cm_failure, accuracy_failure, mcc_failure, report_failure_type, cm_failure_type, accuracy_failure_type, mcc_failure_type, train_acc, test_acc):
    """Save the results and plots in a specified folder."""
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, f'metrics_{model_name}.txt'), 'w') as f:
        f.write("Failure Detection:\n")
        f.write(report_failure)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm_failure))
        f.write("\nAccuracy: {}\n".format(accuracy_failure))
        f.write("Matthews Correlation Coefficient (MCC): {}\n".format(mcc_failure))
        f.write("\nTrain Accuracy: {}\n".format(train_acc))
        f.write("Test Accuracy: {}\n\n".format(test_acc))
        f.write("Failure Type Detection:\n")
        f.write(report_failure_type)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm_failure_type))
        f.write("\nAccuracy: {}\n".format(accuracy_failure_type))
        f.write("Matthews Correlation Coefficient (MCC): {}\n".format(mcc_failure_type))

    # Plot and save confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ConfusionMatrixDisplay(cm_failure).plot(ax=ax[0], cmap=plt.cm.Blues)
    ax[0].set_title(f'{model_name} - Failure Detection Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')

    ConfusionMatrixDisplay(cm_failure_type).plot(ax=ax[1], cmap=plt.cm.Blues)
    ax[1].set_title(f'{model_name} - Failure Type Detection Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')

    plt.savefig(os.path.join(folder_path, f'confusion_matrices_{model_name}.png'))

def save_descriptive_statistics(data, folder_path):
    """Save descriptive statistics to a CSV file in the specified folder."""
    os.makedirs(folder_path, exist_ok=True)
    desc_stats = data.describe().T
    null_counts = data.isnull().sum()
    desc_stats['null_count'] = null_counts
    desc_stats.to_csv(os.path.join(folder_path, 'descriptive_statistics.csv'))

def main():
    filepath = './dataset/predictive_maintenance.csv'
    data = load_data(filepath)
    features, target_failure, target_failure_type = preprocess_data(data)
    
    # Create a main results folder with timestamp
    main_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_folder_path = f'./results/{main_timestamp}'
    os.makedirs(main_folder_path, exist_ok=True)
    
    # Save descriptive statistics
    save_descriptive_statistics(data, main_folder_path)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100)
    }

    all_metrics = {
        'accuracy_failure': [],
        'mcc_failure': [],
        'accuracy_failure_type': [],
        'mcc_failure_type': [],
        'train_acc_failure': [],
        'test_acc_failure': [],
        'train_acc_failure_type': [],
        'test_acc_failure_type': []
    }

    for i in range(10):
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_folder_path = os.path.join(main_folder_path, f'run_{i+1}_{run_timestamp}')
        os.makedirs(run_folder_path, exist_ok=True)

        # Split data for failure detection
        X_train_failure, X_test_failure, y_train_failure, y_test_failure = split_data(features, target_failure)


        for model_name, model in models.items():
            print(f"\nEvaluating model: {model_name} (Run {i+1})")
            rf_failure = train_and_evaluate(model, X_train_failure, y_train_failure, X_test_failure, y_test_failure)
            y_pred_failure, cm_failure, accuracy_failure, mcc_failure, report_failure, train_acc_failure, test_acc_failure = rf_failure

            # Split data for failure type prediction
            X_train_failure_type, X_test_failure_type, y_train_failure_type, y_test_failure_type = split_data(features, target_failure_type)
           
            rf_failure_type = train_and_evaluate(model, X_train_failure_type, y_train_failure_type, X_test_failure_type, y_test_failure_type)
            y_pred_failure_type, cm_failure_type, accuracy_failure_type, mcc_failure_type, report_failure_type, train_acc_failure_type, test_acc_failure_type = rf_failure_type

            # Save the results
            model_folder_path = os.path.join(run_folder_path, model_name)
            save_results(model_folder_path, model_name, report_failure, cm_failure, accuracy_failure, mcc_failure, report_failure_type, cm_failure_type, accuracy_failure_type, mcc_failure_type, train_acc_failure, test_acc_failure)

            # Collect metrics for averaging
            all_metrics['accuracy_failure'].append(accuracy_failure)
            all_metrics['mcc_failure'].append(mcc_failure)
            all_metrics['accuracy_failure_type'].append(accuracy_failure_type)
            all_metrics['mcc_failure_type'].append(mcc_failure_type)
            all_metrics['train_acc_failure'].append(train_acc_failure)
            all_metrics['test_acc_failure'].append(test_acc_failure)
            all_metrics['train_acc_failure_type'].append(train_acc_failure_type)
            all_metrics['test_acc_failure_type'].append(test_acc_failure_type)

    # Calculate average metrics
    avg_metrics = {k: mean(v) for k, v in all_metrics.items()}

    # Save average metrics to main folder
    with open(os.path.join(main_folder_path, 'average_metrics.txt'), 'w') as f:
        for metric, value in avg_metrics.items():
            f.write(f'{metric}: {value}\n')

if __name__ == "__main__":
    main()
