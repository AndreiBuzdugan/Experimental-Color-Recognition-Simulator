import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os

df = pd.read_csv('date_antrenament.csv')

df['Culoare'] = pd.Categorical(df['Culoare'])
df['Culoare'] = df['Culoare'].cat.codes

X = df[['H', 'S', 'V']].values
y = df['Culoare'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

save_dir = r'C:\Users\Andrei\Desktop\RN\Proiect\Proiect_Var2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filename_counter_file = os.path.join(save_dir, 'filename_counter.txt')
if os.path.exists(filename_counter_file):
    with open(filename_counter_file, 'r') as file:
        last_filename_counter = int(file.read())
else:
    last_filename_counter = 0

fig_filenames = []

epochs = 30
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracy, label='Acuratețe antrenament', marker='o')
plt.plot(range(1, epochs + 1), test_accuracy, label='Acuratețe test', marker='o')
plt.xlabel('Epoci')
plt.ylabel('Acuratețe')
plt.title('Acuratețea modelului în timpul antrenării')
plt.legend()
plt.grid(True)

last_filename_counter += 1
fig_filename = os.path.join(save_dir, f'acuratete_model_{last_filename_counter}.png')

plt.savefig(fig_filename)
plt.show()

with open(filename_counter_file, 'w') as file:
    file.write(str(last_filename_counter))

fig_filenames.append(fig_filename)

model.save('culoare_classifier_model')
