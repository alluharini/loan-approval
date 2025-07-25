# app_knn.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

app = Flask(__name__)

# Load the data
data = pd.read_csv("LP.csv")
data.drop(['Loan_ID'], axis=1, inplace=True)

# Handle categorical variables using label encoding
label_encoder = preprocessing.LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_encoder.fit_transform(data[col])

data = data.fillna(data.mean())

X = data.drop(['Loan_status'], axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)


# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, Y)

# Calculate accuracy on the test set
Y_pred = knn.predict(X_test)
accuracy = 100 * metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy score (KNN):", accuracy)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        new_user_data = {
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'Property_Area': request.form['Property_Area'],
        }

        for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            if new_user_data[col] not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, new_user_data[col])
            new_user_data[col] = label_encoder.transform([new_user_data[col]])[0]

        new_user_input = pd.DataFrame([new_user_data], columns=X.columns)
        
        # Use KNN model for prediction
        prediction = knn.predict(new_user_input)[0]

        if prediction == 1:
            return render_template('result_accepted.html')
        else:
            return render_template('result_rejected.html')

if __name__ == '__main__':
    app.run(debug=True)
