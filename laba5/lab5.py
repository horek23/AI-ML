import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.randint(0, 2, size=(230, 12))
sum_X = np.sum(X, axis=1)
Y = np.array([[1, 0] if s > 10 else [0, 1] for s in sum_X])

np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(8, activation='sigmoid', input_shape=(12,)),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

print(f"Accuracy: {accuracy_score(y_test_labels, y_pred_labels):.2f}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()