import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import add_silvio_cut


def read_csv(data_path):
    # Load and preprocess the data
    data = pd.read_csv(data_path)
    data = data[data.iloc[: , 0] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 1] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 2] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 3] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 4] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 5] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 6] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 7] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 8] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 9] != 11111111.0] # drop if period = 11111111.0
    data = data[data.iloc[: , 10] != 11111111.0] # drop if period = 11111111.0
    data.reset_index(drop=True, inplace=True)
    return data

def get_scaler(X_train, X_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def get_class_weights(y_encoded):
    # We calculate the class weights based on the inverse frequency of each class.
    # This gives more weight to the underrepresented classes (like RRd) during training.
    # Calculate class weights
    class_counts = y_encoded.sum()
    class_weights = {i: 1.0 / count for i, count in enumerate(class_counts)}    
    return class_weights

def build_model(features, mod):
    if mod == "softmax":
        units = 3
        activation = "softmax"
        loss = "categorical_crossentropy"
    elif mod == "sigmoid":
        units = 3
        activation = "sigmoid"
        loss = "categorical_crossentropy"

    # Build the model
    model = models.Sequential([
            layers.Dense(64, activation='relu', 
                         kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=5e-2),
                         bias_regularizer=tf.keras.regularizers.L2(5e-2), 
                         input_shape=(len(features),)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32,
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=5e-2),
                        bias_regularizer=tf.keras.regularizers.L2(5e-2), 
                        activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16,
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=5e-2),
                        bias_regularizer=tf.keras.regularizers.L2(5e-2), 
                        activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units, activation=activation)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                loss=loss,
                metrics=['accuracy'])

    return model

def train_model(model, X_train_scaled, y_train, class_weights):
    # Train the model
    history = model.fit(X_train_scaled, y_train, 
                        epochs=100, 
                        batch_size=32, 
                        validation_split=0.2,
                        class_weight=class_weights,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return history

def plot_history(history, model_type):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("evaluation/" + 'history_accuracy-' + str(model_type) + '.png', format='png', dpi=300)
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("evaluation/" + 'history_loss-' + str(model_type) +  '.png', format='png', dpi=300)
    plt.close()

def plot_classification(y_true, model_type):
    
    y_true = y_true[y_true.iloc[: , 2] < 1.6] # drop if amplitude > 1.6

    mapping = {'RRab': 0, 'RRc': 1, 'RRd': 2}
    class_number = [mapping[x] for x in y_true['bestclassification']]

    color_palette = sns.color_palette('colorblind', 3)
    classification_colors = [color_palette[x] if x >= 0
                            else (0.5, 0.5, 0.5)
                            for x in class_number]

    plt.scatter(y_true['p1'], y_true['peaktopeakg'], s=.5, linewidth=0, c=classification_colors, alpha=0.85)
    add_silvio_cut(plt)
    plt.title('Classification')
    plt.xlabel('Period')
    plt.ylabel('Amplitude')
    plt.savefig("evaluation/" + 'true_classification-' + str(model_type) + '.png', format='png', dpi=1200)
    plt.close()


def plot_diff_classification(y_true, diff_classification, model_type):
    
    filtered_y = y_true[y_true['sourceid'].isin(diff_classification['sourceid'])]
    
    mapping = {'RRab': 0, 'RRc': 1, 'RRd': 2}
    class_number = [mapping[x] for x in diff_classification['predicted']]

    color_palette = sns.color_palette('colorblind', 3)
    classification_colors = [color_palette[x] if x >= 0
                            else (0.5, 0.5, 0.5)
                            for x in class_number]
    
    plt.scatter(filtered_y['p1'], filtered_y['peaktopeakg'], s=.5, linewidth=0, c=classification_colors, alpha=0.85)
    add_silvio_cut(plt)
    plt.title('Classification')
    plt.xlabel('Period')
    plt.ylabel('Amplitude')
    plt.savefig("evaluation/" + 'predicted_classification-' + str(model_type) + '.png', format='png', dpi=1200)
    plt.close()

def save_model(model, scaler, model_type):
    # Create a directory for saving models if it doesn't exist
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # Save the trained model
    model.save('saved_models/rr_lyrae_model' + str(model_type) + '.keras')
    print("Model saved")

    # Save the scaler
    joblib.dump(scaler, 'saved_models/rr_lyrae_scaler' + str(model_type) + '.joblib')
    print("Scaler saved")

# Function to load the model and scaler
def load_model_and_scaler(model_type):
    loaded_model = tf.keras.models.load_model('saved_models/rr_lyrae_model' + str(model_type) + '.keras')
    loaded_scaler = joblib.load('saved_models/rr_lyrae_scaler' + str(model_type) + '.joblib')
    return loaded_model, loaded_scaler

# Function to classify new data
def classify_star(star_data, model, scaler, model_type):
    
    classes = ['RRab', 'RRc', 'RRd']
    X = star_data.drop('sourceid', axis=1)
    star_data_scaled = scaler.transform(X)
    prediction = model.predict(star_data_scaled)
    class_index = np.argmax(prediction, axis=1)

    predictions = [[star_data['sourceid'][count], classes[idx], max(prediction[count])] for count, idx in enumerate(class_index)]
    return pd.DataFrame(predictions, columns= ['sourceid', 'bestclassification', 'probability'])

def merge_dataframes_by_highest_probability(df1, df2):
    # Concatenate the two dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Sort the combined dataframe by 'sourceid' and 'probability' in descending order
    combined_df_sorted = combined_df.sort_values(['sourceid', 'probability'], ascending=[True, False])
    # Keep the first occurrence of each 'sourceid' (which will be the one with the highest probability)
    result_df = combined_df_sorted.drop_duplicates(subset='sourceid', keep='first')
    # Reset the index of the resulting dataframe
    result_df = result_df.reset_index(drop=True)
    
    return result_df

def compare_classification(y_predicted, y_true):
    # Ensure the columns are the same
    #assert set(y_predicted.columns) == set(y_true.columns), "Dataframes must have the same columns"

    # Compare the 'bestclassification' column
    differences = []
    for idx, (cls1, cls2, source_id) in enumerate(zip(y_predicted['bestclassification'], y_true['bestclassification'], y_predicted['sourceid'])):
        if cls1 != cls2:
            differences.append((source_id, cls1, cls2))

    """
    if differences:
        print("Differences in 'bestclassification' column:")
        for source_id, cls1, cls2 in differences:
            print(f"Source ID: {source_id}, Predicted: {cls1}, True: {cls2}")
    else:
        print("No differences found in the 'bestclassification' column.")
    """
    # Create a new dataframe with only the rows that have different classifications
    diff_classification = pd.DataFrame(differences, columns=['sourceid', 'predicted', 'true'])

    return diff_classification
