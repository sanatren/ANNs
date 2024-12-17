# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import keras_tuner as kt
import pickle
import datetime

# Load Dataset
Bank_data = pd.read_csv('Churn_Modelling.csv')

# Drop Unrelated Columns
Bank_data = Bank_data.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1)

# Label Encode Gender Column
Lb_encoder_Gen = LabelEncoder()
Bank_data['Gender'] = Lb_encoder_Gen.fit_transform(Bank_data['Gender'])

# One Hot Encode Geography Column
ohe_encoder_geo = OneHotEncoder()
geo_encoder = ohe_encoder_geo.fit_transform(Bank_data[['Geography']]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=ohe_encoder_geo.get_feature_names_out(['Geography']))

# Combine Encoded Columns and Drop Geography
Bank_data = pd.concat([Bank_data.drop('Geography', axis=1), geo_encoder_df], axis=1)

# Split Features and Target
X = Bank_data.drop('Exited', axis=1)
y = Bank_data['Exited']

# Oversampling with RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_over, y_over = oversample.fit_resample(X, y)
print("Class distribution after oversampling:", Counter(y_over))

# Train-Test Split After Oversampling
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# Scale the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the Scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# Hyperparameter Tuning with KerasTuner
def build_model(hp):
    model = Sequential()

    # First Hidden Layer
    model.add(Dense(
        units=hp.Int('units_1', min_value=16, max_value=128, step=16),
        activation='relu',
        input_shape=(X_train.shape[1],)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.0, 0.5, step=0.1)))

    # Second Hidden Layer
    model.add(Dense(
        units=hp.Int('units_2', min_value=16, max_value=128, step=16),
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0.0, 0.5, step=0.1)))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer with Learning Rate
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Hyperparameter Tuning with KerasTuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='churn_tuning'
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Start Search
tuner.search(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, tensorboard_callback]
)

# Get the Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The optimal number of units in the first layer is {best_hps.get('units_1')} and 
the second layer is {best_hps.get('units_2')} with a dropout rate of {best_hps.get('dropout_1')} 
and learning rate {best_hps.get('learning_rate')}.
""")

# Train the Best Model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    callbacks=[early_stop, tensorboard_callback]
)

# Evaluate the Best Model
y_pred = (best_model.predict(X_test) > 0.5).astype(int)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
