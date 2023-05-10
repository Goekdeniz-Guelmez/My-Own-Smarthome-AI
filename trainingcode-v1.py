import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# Load the dataset
df = pd.read_csv('smart_home_data.csv')

# Select features
features = df[['Indoor_Temperature', 'Outdoor_Temperature', 'Light_Level']]
# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Light switches
light_status = df[['Light1_Status', 'Light2_Status']]
encoder_light = LabelEncoder()
light_status = light_status.apply(encoder_light.fit_transform)

# RGB LED
led_rgb = df[['LED_Red', 'LED_Green', 'LED_Blue']].replace('NA', np.nan).fillna(-1).astype(int)

# Outlets
outlet_status = df[['Outlet1_Status', 'Outlet2_Status']]
encoder_outlet = LabelEncoder()
outlet_status = outlet_status.apply(encoder_outlet.fit_transform)

# Split the data into training and test sets
features_train, features_test, light_train, light_test, led_train, led_test, outlet_train, outlet_test = train_test_split(features, light_status, led_rgb, outlet_status, test_size=0.2, random_state=42)

# Define the model architecture
input_layer = Input(shape=(features_train.shape[1],))

# Shared layer
shared_layer = Dense(64, activation='relu')(input_layer)

# Light switch layers
light_layer = Dense(32, activation='relu')(shared_layer)
light_output = Dense(2, activation='sigmoid', name='light')(light_layer)

# LED layers
led_layer = Dense(32, activation='relu')(shared_layer)
led_output = Dense(3, activation='relu', name='led')(led_layer)

# Outlet layers
outlet_layer = Dense(32, activation='relu')(shared_layer)
outlet_output = Dense(2, activation='sigmoid', name='outlet')(outlet_layer)

# Create the model
model = Model(inputs=input_layer, outputs=[light_output, led_output, outlet_output])

# Compile the model
model.compile(loss={'light': 'binary_crossentropy', 'led': 'mse', 'outlet': 'binary_crossentropy'},
              optimizer='adam',
              metrics={'light': 'accuracy', 'led': 'mse', 'outlet': 'accuracy'})

# Train the model
model.fit(features_train,
          {'light': light_train, 'led': led_train, 'outlet': outlet_train},
          epochs=50,
          batch_size=32,
          validation_data=(features_test, {'light': light_test, 'led': led_test, 'outlet': outlet_test}))

# Evaluate the model
score = model.evaluate(features_test, {'light': light_test, 'led': led_test, 'outlet': outlet_test})
print('Light Test loss:', score[0])
print('Light Test accuracy:', score[1])
print('LED Test loss:', score[2])
print('LED Test accuracy:', score[3])
print('Outlet Test loss:', score[4])
print('Outlet Test accuracy:', score[5])
