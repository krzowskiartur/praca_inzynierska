import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, recall_score, precision_score, roc_auc_score)
import joblib
import os

# wyłączenie ostrzeżeń
import warnings
warnings.filterwarnings('ignore')

# ustawienia wizualizacji
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# create katalog na wyniki
os.makedirs('../wyniki', exist_ok=True)
os.makedirs('../modele', exist_ok=True)

print("=" * 80)
print("MODEL BAZOWY: RANDOM FOREST")
print("=" * 80)

# seed dla powtarzalności wyników
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# przygotowanie danych*************************************************************************************************************************************************************************************************************

print("\nLoading danych...")
dane = pd.read_csv('../dane/heart_failure_data.csv')
print(f"Wczytano dane {dane.shape[0]} pacjentów")

# cechy do modelowania
cechy_wybrane = ['age', 'ejection_fraction', 'serum_creatinine']
zmienna_celu = 'DEATH_EVENT'


# przygotowanie macierzy cech (X) i wektora celu (y)
X = dane[cechy_wybrane]
y = dane[zmienna_celu]
print(f"\nKształt danych:")
print(f" Cechy (X): {X.shape}")
print(f" Cel (y):   {y.shape}")

# podział na zbiór treningowy i testowy (stratyfikacja dla proporcji klas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% na test
    random_state=RANDOM_SEED,
    stratify=y  # proporcje
)

print(f"\nPodział danych:")
print(f" Zbiór treningowy: {X_train.shape[0]} próbek")
print(f"  Zbiór testowy:    {X_test.shape[0]} próbek")

# standaryzacja

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDane przygotowane i przeskalowane")

# hiperparametry ************************************************************************************************************************************************************************************************************************************
parametry_do_testowania = {
    'n_estimators': [100, 200, 300, 500],  # liczba drzew w lesie
    'max_depth': [10, 20, 30, None],  # maksymalna głębokość drzewa
    'min_samples_split': [2, 5, 10],  # minimalna liczba próbek do podziału węzła
    'min_samples_leaf': [1, 2, 4],  # minimalna liczba próbek w liściu
    'max_features': ['sqrt', 'log2']  # liczba cech do rozważenia przy podziale
}

print("\nPrzeszukiwana przestrzeń hiperparametrów:")
for param, values in parametry_do_testowania.items():
    print(f"  {param:20s}: {values}")

# model bazowy
rf_base = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

# konfiguracja walidacji krzyżowej
#  5-fold stratified CV
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Randomized Search (Grid Search wolniejszy), losowe hiperparametry
print("\nRozpoczynanie przeszukiwania (może to potrwać kilka minut)...")
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=parametry_do_testowania,
    n_iter=100,  # liczba kombinacji
    cv=cv_strategy,
    scoring='f1',  # F1-score
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=1
)

# trening
random_search.fit(X_train_scaled, y_train)

print(f"\nNajlepsze hiperparametry: ")
for param, value in random_search.best_params_.items():
    print(f"  {param:20s}: {value}")

print(f"\nNajlepszy wynik F1-score (CV): {random_search.best_score_:.4f}")


# FINAL MODEL ******************************************************************************************************************************************************************************************

model_rf = random_search.best_estimator_

model_rf.fit(X_train_scaled, y_train)
print("Model wytrenowany!")

joblib.dump(model_rf, '../modele/random_forest_model.pkl')
joblib.dump(scaler, '../modele/scaler.pkl')
print("\nModel zapisany: random_forest_model.pkl")
print(" Scaler zapisany: scaler.pkl")

# ewaluacja modelu******************************************************************************************************************************************************************************************

# predykcje na zbiorze testowym
y_pred = model_rf.predict(X_test_scaled)
y_pred_proba = model_rf.predict_proba(X_test_scaled)[:, 1]

# metryki
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\nWyniki na zbiorze testowym:")
print(f"  F1-score:  {f1:.3f}")
print(f"  Recall:    {recall:.3f} ")
print(f"  Precision: {precision:.3f} ")
print(f"  AUC-ROC:   {auc:.3f}")

print("\nSzczegółowy raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=['Przeżył', 'Zmarł']))


cm = confusion_matrix(y_test, y_pred)
print(f"  TN (True Negative):  {cm[0,0]} - poprawnie przewidziane przeżycia")
print(f"  FP (False Positive): {cm[0,1]} - fałszywe alarmy")
print(f"  FN (False Negative): {cm[1,0]} - przegapieni pacjenci (KRYTYCZNE!)")
print(f"  TP (True Positive):  {cm[1,1]} - poprawnie wykryte zagrożenia")

print(f"\nInterpretacja medyczna:")
pacjenci_wysokiego_ryzyka = cm[1,0] + cm[1,1]
wykryci = cm[1,1]
przegapieni = cm[1,0]
procent_wykrycia = (wykryci / pacjenci_wysokiego_ryzyka) * 100
print(f"  Pacjentów wysokiego ryzyka: {pacjenci_wysokiego_ryzyka}")
print(f"  Wykrytych: {wykryci} ({procent_wykrycia:.1f}%)")
print(f"  Przegapionych: {przegapieni} ({100-procent_wykrycia:.1f}%)")

# obrazek macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Przeżył', 'Zmarł'],
            yticklabels=['Przeżył', 'Zmarł'])
plt.title('Macierz Pomyłek - Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Przewidziana klasa')
plt.tight_layout()
plt.savefig('../wyniki/04_macierz_pomylek_rf.png', dpi=300, bbox_inches='tight')
print("\n✓ Wykres zapisany: 04_macierz_pomylek_rf.png")
plt.close()


# ważność cech ********************************************************************************************************************************************************************************************************************

# Pobranie ważności cech z modelu
waznosc_cech = model_rf.feature_importances_
cechy_sorted = sorted(zip(cechy_wybrane, waznosc_cech), key=lambda x: x[1], reverse=True)

print("\nWażność cech :")
for i, (cecha, waznosc) in enumerate(cechy_sorted, 1):
    print(f"  {i}. {cecha:25s}: {waznosc:.4f} ({waznosc*100:.1f}%)")

# obrazek
plt.figure(figsize=(10, 6))
cechy_nazwy = [x[0] for x in cechy_sorted]
cechy_wartosci = [x[1] for x in cechy_sorted]

plt.barh(cechy_nazwy, cechy_wartosci, color='#3498db', alpha=0.7, edgecolor='black')
plt.xlabel('Ważność cechy')
plt.title('Ważność cech w modelu Random Forest', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../wyniki/05_waznosc_cech_rf.png', dpi=300, bbox_inches='tight')
print("\nWykres zapisany: 05_waznosc_cech_rf.png")
plt.close()
