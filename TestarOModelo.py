from tensorflow.keras.models import load_model
import numpy as np

modelo = load_model("modelo_letras.keras")

# Exemplo: prever uma letra
pred = modelo.predict(X_test[0].reshape(1, 28, 28, 1))
letra_prevista = np.argmax(pred)

print("Letra prevista (índice):", letra_prevista)
