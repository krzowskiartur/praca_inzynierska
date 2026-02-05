import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# wyłączenie ostrzeżeń
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# ustawienia wizualizacji
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

os.makedirs('../wyniki', exist_ok=True)

# ustawienie seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#przygotowanie danych ******************************************************************************************************************************************************************************************************************************
dane = pd.read_csv('../dane/heart_failure_data.csv')
print(f"Wczytano dane {dane.shape[0]} pacjentów")

cechy = ['age', 'ejection_fraction', 'serum_creatinine']
zmienna_celu = 'DEATH_EVENT'

X = dane[cechy]
y = dane[zmienna_celu]

# sprawdzenie niezbalansowania
rozklad = y.value_counts()
print(f"\nOryginalny rozkład klas:")
print(f"  Klasa 0 (przeżył): {rozklad[0]} ({rozklad[0]/len(y)*100:.1f}%)")
print(f"  Klasa 1 (zmarł):   {rozklad[1]} ({rozklad[1]/len(y)*100:.1f}%)")
print(f"  Stosunek: {rozklad[0]/rozklad[1]:.2f}:1")

# train,test podział
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"\nPodział danych:")
print(f"  Trening: {X_train.shape[0]} próbek")
print(f"  Test:    {X_test.shape[0]} próbek")

# trening i ewalucaja modelu ******************************************************************************************************************************************************************************************************************************

def trenuj_i_ewaluuj_mlp(X_train, y_train, X_test, y_test, nazwa_scenariusza):
    
    # standaryzacja
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # budowa modelu
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    #trening
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # predykcje
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # metryki
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"    F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
    
    return {
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'cm': cm,
        'y_pred': y_pred
    }

# na niezbilansowanych danych ******************************************************************************************************************************************************************************************************************************

wyniki_baseline = trenuj_i_ewaluuj_mlp(X_train, y_train, X_test, y_test, "Baseline")

# undersampling ******************************************************************************************************************************************************************************************************************************
undersampler = RandomUnderSampler(random_state=RANDOM_SEED)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

rozklad_under = pd.Series(y_train_under).value_counts()
print(f"\nRozkład po undersampling:")
print(f"  Klasa 0: {rozklad_under[0]} próbek")
print(f"  Klasa 1: {rozklad_under[1]} próbek")
print(f"  Stosunek: {rozklad_under[0]/rozklad_under[1]:.2f}:1 (zbalansowane!)")

wyniki_undersampling = trenuj_i_ewaluuj_mlp(
    X_train_under, y_train_under, X_test, y_test, "Undersampling"
)

# SMOTE ******************************************************************************************************************************************************************************************************************************

smote = SMOTE(random_state=RANDOM_SEED)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rozklad_smote = pd.Series(y_train_smote).value_counts()
print(f"\nRozkład po SMOTE:")
print(f"  Klasa 0: {rozklad_smote[0]} próbek")
print(f"  Klasa 1: {rozklad_smote[1]} próbek")
print(f"  Stosunek: {rozklad_smote[0]/rozklad_smote[1]:.2f}:1 (zbalansowane!)")

wyniki_smote = trenuj_i_ewaluuj_mlp(
    X_train_smote, y_train_smote, X_test, y_test, "SMOTE"
)

# porównanie wyników ******************************************************************************************************************************************************************************************************************************

scenariusze = ['Baseline\n(niezbalansowane)', 'Undersampling', 'SMOTE\n(oversampling)']
wyniki_f1 = [wyniki_baseline['f1'], wyniki_undersampling['f1'], wyniki_smote['f1']]
wyniki_recall = [wyniki_baseline['recall'], wyniki_undersampling['recall'], wyniki_smote['recall']]
wyniki_precision = [wyniki_baseline['precision'], wyniki_undersampling['precision'], wyniki_smote['precision']]

# tabela
print("\nTabela porównawcza:")
print("-" * 70)
print(f"{'Scenariusz':<25} {'F1-score':<12} {'Recall':<12} {'Precision':<12}")
print("-" * 70)
for i, scen in enumerate(['Baseline', 'Undersampling', 'SMOTE']):
    print(f"{scen:<25} {wyniki_f1[i]:<12.4f} {wyniki_recall[i]:<12.4f} {wyniki_precision[i]:<12.4f}")
print("-" * 70)

# najlepszy wynik
najlepszy_idx = wyniki_f1.index(max(wyniki_f1))
print(f"\n Najlepszy scenariusz: {['Baseline', 'Undersampling', 'SMOTE'][najlepszy_idx]}")
print(f"  F1-score: {wyniki_f1[najlepszy_idx]:.4f}")

# Wobrazek metryk
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(scenariusze))
width = 0.25

bars1 = ax.bar(x - width, wyniki_f1, width, label='F1-score', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, wyniki_recall, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, wyniki_precision, width, label='Precision', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Scenariusz', fontsize=12)
ax.set_ylabel('Wartość metryki', fontsize=12)
ax.set_title('Wpływ balansowania klas na wydajność MLP', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenariusze)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.0])

# dodanie wartości na słupkach
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../wyniki/09_porownanie_balansowania.png', dpi=300, bbox_inches='tight')
print("\n✓ Wykres zapisany: 09_porownanie_balansowania.png")
plt.close()

# Obrazek macierzy pomyłek
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (wyniki, nazwa) in enumerate([
    (wyniki_baseline, 'Baseline'),
    (wyniki_undersampling, 'Undersampling'),
    (wyniki_smote, 'SMOTE')
]):
    ax = axes[idx]
    sns.heatmap(wyniki['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Przeżył', 'Zmarł'],
                yticklabels=['Przeżył', 'Zmarł'], ax=ax)
    ax.set_title(f'{nazwa}\nF1={wyniki["f1"]:.3f}', fontsize=12, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Rzeczywista klasa')
    ax.set_xlabel('Przewidziana klasa')

plt.suptitle('Porównanie macierzy pomyłek dla różnych metod balansowania', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../wyniki/10_macierze_pomylek_balansowanie.png', dpi=300, bbox_inches='tight')
print("✓ Wykres zapisany: 10_macierze_pomylek_balansowanie.png")
plt.close()



