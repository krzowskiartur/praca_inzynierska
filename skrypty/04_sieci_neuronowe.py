import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import os

# wyłączenie TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# ustawienia wizualizacji
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

os.makedirs('../wyniki', exist_ok=True)
os.makedirs('../modele', exist_ok=True)

#seed dla powtarzalności
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#przygotowanie danych *******************************************************************************************************************************************************************************************************
dane = pd.read_csv('../dane/heart_failure_data.csv')
print(f"Wczytano dane {dane.shape[0]} pacjentów")

# Wybór cech ( = Random Forest)
cechy = ['age', 'ejection_fraction', 'serum_creatinine']
zmienna_celu = 'DEATH_EVENT'

X = dane[cechy]
y = dane[zmienna_celu]

# testowe, treningowe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"\nPodział danych:")
print(f" Trening: {X_train.shape[0]} próbek")
print(f" Test:    {X_test.shape[0]} próbek")

#standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDane przeskalowane")

#różne architektury ************************************************************************************************************************************************************************************************************************

# testowane architektury
architektury = {
    'Shallow_32': [32],
    'Shallow_64': [64],
    'Shallow_128': [128],
    'Medium_64_32': [64, 32],
    'Medium_128_64': [128, 64],
    'Deep_128_64_32': [128, 64, 32]
}

wyniki_architektur = {}

for nazwa, warstwy in architektury.items():
    print(f"\n  Trenowanie: {nazwa} {warstwy}")

    model = keras.Sequential()
    model.add(layers.Input(shape=(3,)))  # 3 cechy wejściowe



    for liczba_neuronow in warstwy:
        model.add(layers.Dense(liczba_neuronow, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # trening
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # eewaluacja
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    wyniki_architektur[nazwa] = {
        'f1': f1,
        'recall': recall,
        'precision': precision
    }
    
    print(f"F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

# najlepsze architektura
najlepsza_arch = max(wyniki_architektur.items(), key=lambda x: x[1]['f1'])
print(f"\nNajlepsza architektura: {najlepsza_arch[0]}")
print(f" F1-score: {najlepsza_arch[1]['f1']:.4f}")

# regularyzacja ***************************************************************************************************************************************************************************************************************************************************

# poziomy dropout
dropout_levels = [0.0, 0.2, 0.3, 0.5]

wyniki_dropout = {}

for dropout in dropout_levels:
    
    # budowa modelu z dropout
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # trening
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # ewaluacja
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    wyniki_dropout[dropout] = {'f1': f1, 'recall': recall}
    print(f"  F1={f1:.4f}, Recall={recall:.4f}")

najlepszy_dropout = max(wyniki_dropout.items(), key=lambda x: x[1]['f1'])
print(f"\nNajlepszy dropout: {najlepszy_dropout[0]}")
print(f"  F1-score: {najlepszy_dropout[1]['f1']:.4f}")

#trening najlepszego modelu ******************************************************************************************************************************************************************************************************************************

# budowa najlepszego modelu
model_final = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model_final.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTrening finalnego modelu (może potrwać chwilę)...")

# trening z early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history_final = model_final.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

print(f"Model wytrenowany ({len(history_final.history['loss'])} epok)")

# zapisanie modelu
model_final.save('../modele/mlp_model.h5')

# ewalucaja final model ******************************************************************************************************************************************************************************************************************************
# predykcje
y_pred_proba_final = model_final.predict(X_test_scaled, verbose=0)
y_pred_final = (y_pred_proba_final > 0.5).astype(int).flatten()

# metryki
f1_final = f1_score(y_test, y_pred_final)
recall_final = recall_score(y_test, y_pred_final)
precision_final = precision_score(y_test, y_pred_final)

print("\nWyniki MLP na zbiorze testowym:")
print(f"  F1-score:  {f1_final:.4f}")
print(f"  Recall:    {recall_final:.4f}")
print(f"  Precision: {precision_final:.4f}")

# macierz pomyłek
cm = confusion_matrix(y_test, y_pred_final)
print("\nMacierz pomyłek:")
print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")

# obrazek macierzy
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Przeżył', 'Zmarł'],
            yticklabels=['Przeżył', 'Zmarł'])
plt.title('Macierz Pomyłek - Sieć Neuronowa MLP', fontsize=14, fontweight='bold')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Przewidziana klasa')
plt.tight_layout()
plt.savefig('../wyniki/07_macierz_pomylek_mlp.png', dpi=300, bbox_inches='tight')
#print("\nWykres zapisany: 07_macierz_pomylek_mlp.png")
plt.close()

# obrzek krzywych uczenia
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history_final.history['loss'], label='Trening', linewidth=2)
plt.plot(history_final.history['val_loss'], label='Walidacja', linewidth=2)
plt.title('Funkcja straty podczas treningu', fontsize=12, fontweight='bold')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_final.history['accuracy'], label='Trening', linewidth=2)
plt.plot(history_final.history['val_accuracy'], label='Walidacja', linewidth=2)
plt.title('Dokładność podczas treningu', fontsize=12, fontweight='bold')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../wyniki/08_krzywe_uczenia_mlp.png', dpi=300, bbox_inches='tight')
plt.close()
