import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================================
# 0. CONFIGURACIÓN DE SEMILLA GLOBAL PARA REPRODUCIBILIDAD (Rúbrica: 15%)
# =========================================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================================
# 1. CARGA DE DATOS Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =========================================================================
print("1. Cargando el conjunto de datos Sign Language MNIST...")
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

print(f"Dimensiones Entrenamiento (Train): {train_df.shape}")
print(f"Dimensiones Prueba (Test):         {test_df.shape}")
print(f"Valores nulos en Train:            {train_df.isnull().sum().sum()}")
print(f"Valores nulos en Test:             {test_df.isnull().sum().sum()}")
print(f"Clases únicas detectadas:          {train_df['label'].nunique()}")

# Estadísticas de balanceo de clases y luminancia para el documento
conteo_clases = train_df['label'].value_counts()
pixeles_train = train_df.drop('label', axis=1).values
print(f"Media de muestras por clase:       {conteo_clases.mean():.2f} (std: {conteo_clases.std():.2f})")
print(f"Mínimo por clase: {conteo_clases.min()} | Máximo por clase: {conteo_clases.max()}")
print(f"Intensidad media de píxeles:       {pixeles_train.mean():.2f} (std: {pixeles_train.std():.2f})")

# Diccionario de etiquetas originales (0 a 24, sin el 9 que corresponde a la J)
mapa_original_a_letra = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'
}
nombres_letras = [mapa_original_a_letra[k] for k in sorted(mapa_original_a_letra.keys())]

# Gráfica 1: Distribución de clases en Entrenamiento (EDA)
plt.figure(figsize=(12, 5))
sns.countplot(
    x=train_df['label'].map(mapa_original_a_letra),
    order=nombres_letras,
    palette='viridis'
)
plt.title('Distribución de muestras por letra en el conjunto de entrenamiento (EDA)')
plt.xlabel('Letra del Alfabeto (ASL)')
plt.ylabel('Número de imágenes')
plt.tight_layout()
plt.savefig('eda_distribucion_clases.png', dpi=300)
plt.show()

# Gráfica 2: Visualización de muestras de ejemplo en 28x28 píxeles (EDA)
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = train_df.iloc[i, 1:].values.reshape(28, 28)
    letra = mapa_original_a_letra[train_df.iloc[i, 0]]
    plt.imshow(img, cmap='gray')
    plt.title(f"Letra: {letra}")
    plt.axis('off')
plt.suptitle('Muestras del dataset Sign Language MNIST (28x28 píxeles en escala de grises)')
plt.tight_layout()
plt.savefig('eda_muestras_ejemplo.png', dpi=300)
plt.show()

# =========================================================================
# 2. PREPROCESADO RIGUROSO Y RECODIFICACIÓN BIYECTIVA
# =========================================================================
def preprocesar_datos(df):
    """
    1. Aplica recodificación biyectiva a las etiquetas: {0..8, 10..24} -> {0..23}.
    2. Normaliza las intensidades de píxeles al rango continuo [0, 1].
    3. Reestructura cada vector plano de 784 variables en un tensor 3D (28, 28, 1).
    """
    y_orig = df['label'].values
    y_recod = np.where(y_orig > 9, y_orig - 1, y_orig)

    x_norm = df.drop('label', axis=1).values.astype('float32') / 255.0
    x_tensor = x_norm.reshape(-1, 28, 28, 1)
    return x_tensor, y_recod

print("\n2. Ejecutando preprocesado tensorial y partición estratificada (80% Train / 20% Val)...")
X_full_train, y_full_train = preprocesar_datos(train_df)
X_test, y_test = preprocesar_datos(test_df)

X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train,
    test_size=0.20,
    random_state=SEED,
    stratify=y_full_train
)

print(f"Tensor X_train: {X_train.shape} | Etiquetas y_train: {y_train.shape}")
print(f"Tensor X_val:   {X_val.shape}   | Etiquetas y_val:   {y_val.shape}")
print(f"Tensor X_test:  {X_test.shape}  | Etiquetas y_test:  {y_test.shape}")

# =========================================================================
# 3. EXPERIMENTO 0: PERCEPTRÓN MULTICAPA CLÁSICO (MLP BASELINE)
# =========================================================================
print("\n3. Entrenando Experimento 0: Perceptrón Multicapa (MLP Baseline)...")
mlp_model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(24, activation='softmax')
], name="MLP_Baseline")

mlp_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop_mlp = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_mlp = mlp_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=[early_stop_mlp],
    verbose=1
)

mlp_test_loss, mlp_test_acc = mlp_model.evaluate(X_test, y_test, verbose=0)

# =========================================================================
# 4. EXPERIMENTO 1: RED NEURONAL CONVOLUCIONAL BASE (CNN BASE)
# =========================================================================
print("\n4. Construyendo y entrenando Experimento 1: Modelo CNN Base...")
cnn_base = Sequential([
    Input(shape=(28, 28, 1)),
    # Bloque Convolucional 1
    Conv2D(32, (3, 3), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),

    # Bloque Convolucional 2
    Conv2D(64, (3, 3), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),

    # Bloque Convolucional 3
    Conv2D(128, (3, 3), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),

    # Clasificador Denso
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')
], name="CNN_Base_Entrega2")

cnn_base.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_base.summary()

early_stop_cnn = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_cnn = cnn_base.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=[early_stop_cnn],
    verbose=1
)

# Gráfica 3: Curvas de aprendizaje de la CNN Base
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Entrenamiento (Train)', linewidth=2)
plt.plot(history_cnn.history['val_accuracy'], label='Validación (Val)', linewidth=2)
plt.title('Evolución de la Exactitud (CNN Base)')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Entrenamiento (Train)', linewidth=2)
plt.plot(history_cnn.history['val_loss'], label='Validación (Val)', linewidth=2)
plt.title('Evolución de la Función de Pérdida (CNN Base)')
plt.xlabel('Época')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('curvas_entrenamiento_cnn_base.png', dpi=300)
plt.show()

# =========================================================================
# 5. EVALUACIÓN EN PRUEBA EXTERNA Y MATRIZ DE CONFUSIÓN
# =========================================================================
print("\n5. Evaluando modelos sobre el conjunto de prueba independiente (Test)...")
cnn_test_loss, cnn_test_acc = cnn_base.evaluate(X_test, y_test, verbose=0)

best_ep_mlp = np.argmin(history_mlp.history['val_loss'])
best_ep_cnn = np.argmin(history_cnn.history['val_loss'])

print("\n" + "="*70)
print("RESUMEN DE MÉTRICAS PARA LA TABLA 2 DE LA ENTREGA 2:")
print("="*70)
print(f"Exp 0 (MLP Clásico) -> Train Acc: {history_mlp.history['accuracy'][best_ep_mlp]*100:.2f}% | "
      f"Val Acc: {history_mlp.history['val_accuracy'][best_ep_mlp]*100:.2f}% | "
      f"Test Loss: {mlp_test_loss:.4f} | Test Acc: {mlp_test_acc*100:.2f}%")
print(f"Exp 1 (CNN Base)    -> Train Acc: {history_cnn.history['accuracy'][best_ep_cnn]*100:.2f}% | "
      f"Val Acc: {history_cnn.history['val_accuracy'][best_ep_cnn]*100:.2f}% | "
      f"Test Loss: {cnn_test_loss:.4f} | Test Acc: {cnn_test_acc*100:.2f}%")
print("="*70 + "\n")

# Predicciones y Matriz de Confusión de la CNN Base
y_pred_probs = cnn_base.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 11))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=nombres_letras, yticklabels=nombres_letras
)
plt.title('Matriz de Confusión en Prueba Independiente - Modelo CNN Base (Entrega 2)')
plt.ylabel('Letra Real (Ground Truth)')
plt.xlabel('Predicción de la Red Neuronal')
plt.tight_layout()
plt.savefig('matriz_confusion_cnn_base.png', dpi=300)
plt.show()

print("Reporte de Clasificación por Letra (CNN Base):")
print(classification_report(y_test, y_pred, target_names=nombres_letras, digits=4))