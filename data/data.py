import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
import requests
import gdown

def get_data():
    # Download the dataset from the URL
    # URL of the raw CSV file on GitHub
# URL of the Google Drive file
    url = 'https://drive.google.com/uc?id=18PWgT2ZFHVu0it8AHdwXh4wbCcH8z8e9'

# Output file name
    output = 'data/Churn_Modelling.csv'

# Download the file
    gdown.download(url, output, quiet=False)
    print("Downloaded dataset")
    # Load the dataset
    data = pd.read_csv('data/Churn_Modelling.csv')

    # Select features and target variable
    X = data.iloc[:, 3:13]  # Assuming you want columns 3 to 12 as features
    y = data.iloc[:, 13]    # Assuming the target variable is in column 13

    # Encode categorical variables
    label_encoder = LabelEncoder()
    X['Geography'] = label_encoder.fit_transform(X['Geography'])
    X['Gender'] = label_encoder.fit_transform(X['Gender'])

    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Calculate the number of classes
    class_num = len(data['Exited'].unique())

    # Convert NumPy arrays to tensors
    train_data_num = torch.tensor(X_train, dtype=torch.float32)
    test_data_num = torch.tensor(X_test, dtype=torch.float32)
    train_data_global = torch.tensor(y_train.values, dtype=torch.float32)
    test_data_global = torch.tensor(y_test.values, dtype=torch.float32)

    # Create dictionaries for local data
    train_data_local_num_dict = {0: train_data_num}
    train_data_local_dict = {0: train_data_global}
    test_data_local_dict = {0: test_data_global}

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num
    )

# Usage
train_data_num, test_data_num, train_data_global, test_data_global,train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = get_data()
