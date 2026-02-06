import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score
import os

import warnings
warnings.filterwarnings('ignore')

# #wyglad wykresów( +/-)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

os.makedirs('../wyniki', exist_ok=True)

# seed dla powtarzalności
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# przygotowanie danych*************************************************************************************************************************************************************************************************************
dane = pd.read_csv('../dane/heart_failure_data.csv')
print(f"Wczytano dane {dane.shape[0]} pacjentów")

# podstawowe cechy
cechy_bazowe = ['age', 'ejection_fraction', 'serum_creatinine']
zmienna_celu = 'DEATH_EVENT'

# train,test - podzial
X_base = dane[cechy_bazowe]
y = dane[zmienna_celu]

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"\nPodział danych:")
print(f"  Trening: {X_train_base.shape[0]} próbek")
print(f"  Test:    {X_test_base.shape[0]} próbek")

#cechy interakcyjne  **********************************************************************************************************************************************************************************************************************

print("\nTworzenie nowych cech poprzez interakcje...")

# Dodanie cech interakcyjnych do oryginalnych danych
dane_z_interakcjami = dane.copy()

# 1 wiek * kreatynina
dane_z_interakcjami['age_x_creatinine'] = dane['age'] * dane['serum_creatinine']

# 2 frakcja wyrzutowa * sód
dane_z_interakcjami['ef_x_sodium'] = dane['ejection_fraction'] * dane['serum_sodium']

# 3  wiek * frakcja wyrzutowa
dane_z_interakcjami['age_x_ef'] = dane['age'] * dane['ejection_fraction']

cechy_z_interakcjami = cechy_bazowe + ['age_x_creatinine', 'ef_x_sodium', 'age_x_ef']

print(f"Utworzono 3 nowe cechy interakcyjne")
print(f"Łączna liczba cech: {len(cechy_z_interakcjami)}")

# Przygotowanie danych
X_interactions = dane_z_interakcjami[cechy_z_interakcjami]
X_train_int, X_test_int, _, _ = train_test_split(
    X_interactions, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# skalowanie
scaler_int = StandardScaler()
X_train_int_scaled = scaler_int.fit_transform(X_train_int)
X_test_int_scaled = scaler_int.transform(X_test_int)

# trening
model_interactions = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=4, max_features='sqrt',
    random_state=RANDOM_SEED, n_jobs=-1
)
model_interactions.fit(X_train_int_scaled, y_train)

# ewaluacja
y_pred_int = model_interactions.predict(X_test_int_scaled)
f1_int = f1_score(y_test, y_pred_int)
recall_int = recall_score(y_test, y_pred_int)
precision_int = precision_score(y_test, y_pred_int)

print(f"\nWyniki z cechami interakcyjnymi:")
print(f"  F1-score:  {f1_int:.4f}")
print(f"  Recall:    {recall_int:.4f}")
print(f"  Precision: {precision_int:.4f}")

#MINMAX jako metoda scalowania *********************************************************************************************************************************************************************************************
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train_base)
X_test_minmax = scaler_minmax.transform(X_test_base)

# trening
model_minmax = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=4, max_features='sqrt',
    random_state=RANDOM_SEED, n_jobs=-1
)
model_minmax.fit(X_train_minmax, y_train)

# ewaluacja
y_pred_minmax = model_minmax.predict(X_test_minmax)
f1_minmax = f1_score(y_test, y_pred_minmax)
recall_minmax = recall_score(y_test, y_pred_minmax)
precision_minmax = precision_score(y_test, y_pred_minmax)

print(f"\nWyniki z MinMaxScaler:")
print(f"  F1-score:  {f1_minmax:.4f}")
print(f"  Recall:    {recall_minmax:.4f}")
print(f"  Precision: {precision_minmax:.4f}")

# wszystkie cechy a nie 3 (ale bez DEATH_EVENT I TIME, bo przeciek danych) *********************************************************************************************************************************************************************************************
wszystkie_cechy = dane.select_dtypes(include=[np.number]).columns.tolist()
if 'DEATH_EVENT' in wszystkie_cechy:
    wszystkie_cechy.remove('DEATH_EVENT')
if 'time' in wszystkie_cechy:
    wszystkie_cechy.remove('time')  # Wykluczamy time (target leakage)

print(f"\nLiczba wszystkich dostępnych cech: {len(wszystkie_cechy)}")
print(f"Cechy: {', '.join(wszystkie_cechy)}")

X_all = dane[wszystkie_cechy]
X_train_all, X_test_all, _, _ = train_test_split(
    X_all, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# skalowanie
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

# trening modelu
model_all = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=4, max_features='sqrt',
    random_state=RANDOM_SEED, n_jobs=-1
)
model_all.fit(X_train_all_scaled, y_train)

# ewaluacja
y_pred_all = model_all.predict(X_test_all_scaled)
f1_all = f1_score(y_test, y_pred_all)
recall_all = recall_score(y_test, y_pred_all)
precision_all = precision_score(y_test, y_pred_all)

print(f"\nWyniki ze wszystkimi cechami:")
print(f"  F1-score:  {f1_all:.4f}")
print(f" Recall:    {recall_all:.4f}")
print(f"  Precision: {precision_all:.4f}")

#model bazowy ************************************************************************************************************************************************************************************************************************************************************
scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

model_baseline = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=4, max_features='sqrt',
    random_state=RANDOM_SEED, n_jobs=-1
)
model_baseline.fit(X_train_base_scaled, y_train)

y_pred_base = model_baseline.predict(X_test_base_scaled)
f1_base = f1_score(y_test, y_pred_base)
recall_base = recall_score(y_test, y_pred_base)
precision_base = precision_score(y_test, y_pred_base)

print(f"\nWyniki modelu bazowego:")
print(f"  F1-score:  {f1_base:.4f}")
print(f"  Recall:    {recall_base:.4f}")
print(f"  Precision: {precision_base:.4f}")

#wyniki ************************************************************************************************************************************************************************************************************************************************************

eksperymenty = ['Baseline\n(3 cechy)', 'Cechy\ninterakcyjne', 
                'MinMax\nScaler', 'Wszystkie\ncechy']
wyniki_f1 = [f1_base, f1_int, f1_minmax, f1_all]
wyniki_recall = [recall_base, recall_int, recall_minmax, recall_all]
wyniki_precision = [precision_base, precision_int, precision_minmax, precision_all]

print("\nTabela porównawcza:")
print("-" * 70)
print(f"{'Eksperyment':<25} {'F1-score':<12} {'Recall':<12} {'Precision':<12}")
print("-" * 70)
for i, exp in enumerate(['Baseline', 'Interakcje', 'MinMaxScaler', 'Wszystkie cechy']):
    print(f"{exp:<25} {wyniki_f1[i]:<12.4f} {wyniki_recall[i]:<12.4f} {wyniki_precision[i]:<12.4f}")
print("-" * 70)

# znalezienie najlepszego
najlepszy_idx = wyniki_f1.index(max(wyniki_f1))
print(f"\nNajlepszy wynik: {['Baseline', 'Cechy interakcyjne', 'MinMaxScaler', 'Wszystkie cechy'][najlepszy_idx]}")
print(f"F1-score: {wyniki_f1[najlepszy_idx]:.4f}")

# obrazek z porownaniami
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(eksperymenty))
width = 0.25

bars1 = ax.bar(x - width, wyniki_f1, width, label='F1-score', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, wyniki_recall, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, wyniki_precision, width, label='Precision', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Eksperyment', fontsize=12)
ax.set_ylabel('Wartość metryki', fontsize=12)
ax.set_title('Porównanie wyników eksperymentów z inżynierią cech', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(eksperymenty)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.0])

# do wykresu
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../wyniki/06_porownanie_inzynierii_cech.png', dpi=300, bbox_inches='tight')
plt.close()
