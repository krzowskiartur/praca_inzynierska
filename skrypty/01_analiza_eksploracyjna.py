import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

#wyłączenie ostrzeżeń dla czystszego outputu
import warnings
warnings.filterwarnings('ignore')

#wyglad wykresów
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


os.makedirs('../wyniki', exist_ok=True)

# wczytywanie danych *******************************************************************************************************************************************************************************
try:
    dane = pd.read_csv('../dane/heart_failure_data.csv')
    print(f" git, wczytane !")
    print(f"  Liczba pacjentów: {dane.shape[0]}")
    print(f"  Liczba zmiennych: {dane.shape[1]}")
except FileNotFoundError:
    print("nie ma pliku (trzeba wgrać)")
    print(" plik 'heart_failure_data.csv' musi być w folderze 'dane/' !!!!!!!!")
    exit(1)


#pokazanie danych i czy ma braki
print(dane.head())
print("\nTypy danych i brakujące wartości:")
print(dane.info())

# czy braki ? ()
braki = dane.isnull().sum()
if braki.sum() == 0:
    print("\nDane kompletne")
else:
    print("\nBrakujące wartości istnieja")
    print(braki[braki > 0])


#analiza "DEATH_EVENT" *******************************************************************************************************************************************************************************

rozklad_celu = dane['DEATH_EVENT'].value_counts()
procenty = dane['DEATH_EVENT'].value_counts(normalize=True) * 100

print(f"\nRozkład pacjentów:")
print(f"  Przeżyli (0): {rozklad_celu[0]} pacjentów ({procenty[0]:.1f}%)")
print(f"  Zmarli (1):   {rozklad_celu[1]} pacjentów ({procenty[1]:.1f}%)")

stosunek = rozklad_celu[0] / rozklad_celu[1]
print(f"\nStosunek klas: {stosunek:.2f}:1")
if stosunek > 1.5:
    print("Uwaga: Dane są niezbalansowane - może to wpłynąć na wyniki modeli")

# obrazek
plt.figure(figsize=(10, 6))
kolory = ['#2ecc71', '#e74c3c']
plt.bar(['Przeżyli', 'Zmarli'], rozklad_celu.values, color=kolory, alpha=0.7, edgecolor='black')
plt.title('Rozkład zmiennej celu - Śmiertelność pacjentów', fontsize=14, fontweight='bold')
plt.ylabel('Liczba pacjentów')
plt.xlabel('Status pacjenta')

#etykiet z wartosciami
for i, v in enumerate(rozklad_celu.values):
    plt.text(i, v + 5, f'{v}\n({procenty.values[i]:.1f}%)', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../wyniki/01_rozklad_zmiennej_celu.png', dpi=300, bbox_inches='tight')
print("\nWykres zapisany: 01_rozklad_zmiennej_celu.png")
plt.close()

# statystyki opisowe *****************************************************************************************************************************************************************************************************************

# wybór cech numerycznych (bez zmiennej celu)
cechy_numeryczne = dane.select_dtypes(include=[np.number]).columns.tolist()
if 'DEATH_EVENT' in cechy_numeryczne:
    cechy_numeryczne.remove('DEATH_EVENT')


print(f"\nLiczba cech numerycznych: {len(cechy_numeryczne)}")
print("\nPodstawowe statystyki:")
print(dane[cechy_numeryczne].describe().round(2))


# korelacje ******************************************************************************************************************************************************************************************************************************

# obliczenie korelacji z DEATH_EVENT
korelacje = dane[cechy_numeryczne + ['DEATH_EVENT']].corr()['DEATH_EVENT'].drop('DEATH_EVENT')
korelacje_sorted = korelacje.abs().sort_values(ascending=False)

print("\nNajsilniejsze korelacje ze śmiertelnością:")
for i, (cecha, wartosc) in enumerate(korelacje[korelacje_sorted.index].items(), 1):
    kierunek = "zwiększa" if wartosc > 0 else "zmniejsza"
    print(f"{i}. {cecha:25s}: {wartosc:6.3f}  ({kierunek} ryzyko)")

# obrazek
plt.figure(figsize=(10, 8))
korelacje_do_wykresu = korelacje[korelacje_sorted.index][:10]  # Top 10
kolory_korelacji = ['#e74c3c' if x > 0 else '#2ecc71' for x in korelacje_do_wykresu]

plt.barh(range(len(korelacje_do_wykresu)), korelacje_do_wykresu.values, color=kolory_korelacji, alpha=0.7)
plt.yticks(range(len(korelacje_do_wykresu)), korelacje_do_wykresu.index)

plt.xlabel('Współczynnik korelacji')
plt.title('Top 10 cech skorelowanych ze śmiertelnością', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../wyniki/02_korelacje_z_celem.png', dpi=300, bbox_inches='tight')
print("\nWykres zapisany: 02_korelacje_z_celem.png")
plt.close()


# porónanie grup przeżyli i zmarli ******************************************************************************************************************************************************************************************

# wybór 4 najważniejszych cech do szczegółowej analizy, przygotowanie, porównanie
top_cechy = korelacje_sorted.index[:4].tolist()
print(f"\nAnaliza szczegółowa dla cech: {', '.join(top_cechy)}")
grupa_przezyli = dane[dane['DEATH_EVENT'] == 0]
grupa_zmarli = dane[dane['DEATH_EVENT'] == 1]

print("\nPorównanie średnich wartości:")
print("-" * 67)
for cecha in top_cechy:
    srednia_przezyli = grupa_przezyli[cecha].mean()
    srednia_zmarli = grupa_zmarli[cecha].mean()

    #t-Studenta
    t_stat, p_value = stats.ttest_ind(grupa_przezyli[cecha], grupa_zmarli[cecha])
    
    roznica = abs(srednia_zmarli - srednia_przezyli)
    procent_roznica = (roznica / srednia_przezyli) * 100
    
    print(f"\n{cecha}:")
    print(f"  Przeżyli: {srednia_przezyli:.2f}")
    print(f"  Zmarli:   {srednia_zmarli:.2f}")
    print(f"  Różnica:  {roznica:.2f} ({procent_roznica:.1f}%)")
    print(f"  p-value:  {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# obrazek porównania
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, cecha in enumerate(top_cechy):
    ax = axes[idx]
    
    # boxplot dla obu grup
    dane_do_wykresu = [grupa_przezyli[cecha], grupa_zmarli[cecha]]
    bp = ax.boxplot(dane_do_wykresu, labels=['Przeżyli', 'Zmarli'], 
                     patch_artist=True, widths=0.6)
    
    # kolorowanie
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax.set_ylabel(cecha)
    ax.set_title(f'Porównanie: {cecha}', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Porównanie kluczowych cech między grupami', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../wyniki/03_porownanie_grup.png', dpi=300, bbox_inches='tight')
print("\nWykres zapisany: 03_porownanie_grup.png")
plt.close()

# wartosci odstajace ******************************************************************************************************************************************************************************************

print("\nLiczba wartości odstających (metoda IQR):")
for cecha in top_cechy:
    Q1 = dane[cecha].quantile(0.25)
    Q3 = dane[cecha].quantile(0.75)
    IQR = Q3 - Q1
    dolna_granica = Q1 - 1.5 * IQR
    gorna_granica = Q3 + 1.5 * IQR
    
    outliers = dane[(dane[cecha] < dolna_granica) | (dane[cecha] > gorna_granica)]
    procent = (len(outliers) / len(dane)) * 100
    
    print(f"  {cecha:25s}: {len(outliers):3d} ({procent:5.2f}%)")
