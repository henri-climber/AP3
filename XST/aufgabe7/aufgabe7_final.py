"""
Aufgabe 7 - Finale Version: Robuste Peak-basierte Shift-Analyse

Fokus auf markante, gut-definierte Features (Peaks) statt Gesamtkurven-Alignment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

# ==============================================================================
# SETUP
# ==============================================================================

plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13


# ==============================================================================
# DATEN LADEN
# ==============================================================================

dfA11 = pd.read_excel('data/xst.xlsx', sheet_name='A1.1')
dfA3 = pd.read_excel('data/xst.xlsx', sheet_name='A3')

for df in [dfA11, dfA3]:
    df.dropna(subset=['beta', 'R'], inplace=True)
    df['beta'] = df['beta'].astype(float)
    df['R'] = df['R'].astype(float)
    df.sort_values('beta', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Glätten
dfA11['R_smooth'] = savgol_filter(dfA11['R'], window_length=11, polyorder=3)
dfA3['R_smooth'] = savgol_filter(dfA3['R'], window_length=9, polyorder=3)

print("Daten geladen und geglättet")
print(f"A1.1: {len(dfA11)} Punkte")
print(f"A3:   {len(dfA3)} Punkte")


# ==============================================================================
# METHODE 1: FEATURE-BASIERTE ANALYSE
# ==============================================================================

def gaussian(x, amp, mu, sigma):
    """Gauss-Funktion für Peak-Fitting."""
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


def find_peak_position(df, beta_min, beta_max):
    """
    Findet Peak-Position durch Gauss-Fit.
    """
    mask = (df['beta'] >= beta_min) & (df['beta'] <= beta_max)
    data = df[mask].copy()

    if len(data) < 5:
        # Fallback: Maximum
        idx_max = data['R_smooth'].idxmax()
        return data.loc[idx_max, 'beta'], 0.1

    # Finde Maximum als Startwert
    idx_max = data['R_smooth'].idxmax()
    beta_max_val = data.loc[idx_max, 'beta']
    R_max_val = data.loc[idx_max, 'R_smooth']

    # Gauss-Fit versuchen
    try:
        popt, pcov = curve_fit(
            gaussian,
            data['beta'],
            data['R_smooth'],
            p0=[R_max_val, beta_max_val, 0.3],
            maxfev=5000
        )
        peak_pos = popt[1]
        peak_std = np.sqrt(np.diag(pcov))[1]
        return peak_pos, peak_std
    except:
        # Fallback: verwende Maximum
        return beta_max_val, 0.1


def find_rise_position(df, threshold):
    """
    Findet Position wo Kurve einen Schwellwert überschreitet.
    """
    mask = df['R_smooth'] > threshold
    if mask.sum() == 0:
        return np.nan, np.nan

    idx = df[mask].index[0]
    # Interpoliere für bessere Genauigkeit
    if idx > 0:
        beta_before = df.loc[idx-1, 'beta']
        beta_after = df.loc[idx, 'beta']
        R_before = df.loc[idx-1, 'R_smooth']
        R_after = df.loc[idx, 'R_smooth']

        # Lineare Interpolation
        frac = (threshold - R_before) / (R_after - R_before)
        beta_interp = beta_before + frac * (beta_after - beta_before)

        return beta_interp, 0.05

    return df.loc[idx, 'beta'], 0.1


print("\n" + "=" * 80)
print("FEATURE-BASIERTE ANALYSE")
print("=" * 80)

# Feature 1: Anstiegspunkt (R > 50)
print("\n1. Anstiegspunkt (R > 50 Counts):")
beta_A11_rise, unc_A11_rise = find_rise_position(dfA11, 50)
beta_A3_rise, unc_A3_rise = find_rise_position(dfA3, 50)
shift_rise = beta_A11_rise - beta_A3_rise
print(f"   A1.1:  {beta_A11_rise:.3f}°")
print(f"   A3:    {beta_A3_rise:.3f}°")
print(f"   → Shift: {shift_rise:+.4f}° (A3 muss um {shift_rise:+.4f}° nach rechts)")

# Feature 2: Plateau-Erreichen (R > 400)
print("\n2. Plateau-Erreichen (R > 400 Counts):")
beta_A11_plat, unc_A11_plat = find_rise_position(dfA11, 400)
beta_A3_plat, unc_A3_plat = find_rise_position(dfA3, 400)
shift_plat = beta_A11_plat - beta_A3_plat
print(f"   A1.1:  {beta_A11_plat:.3f}°")
print(f"   A3:    {beta_A3_plat:.3f}°")
print(f"   → Shift: {shift_plat:+.4f}°")

# Feature 3: Haupt-Peak (6-8°)
print("\n3. Haupt-Peak Position (6-8°):")
beta_A11_peak, unc_A11_peak = find_peak_position(dfA11, 6.0, 8.0)
beta_A3_peak, unc_A3_peak = find_peak_position(dfA3, 6.0, 8.0)
shift_peak = beta_A11_peak - beta_A3_peak
print(f"   A1.1:  {beta_A11_peak:.3f}° ± {unc_A11_peak:.3f}°")
print(f"   A3:    {beta_A3_peak:.3f}° ± {unc_A3_peak:.3f}°")
print(f"   → Shift: {shift_peak:+.4f}°")

# Feature 4: Hochplateau (R > 150 und beta > 8)
print("\n4. Hochplateau (beta > 8.5°, R > 150):")
dfA11_high = dfA11[(dfA11['beta'] > 8.5) & (dfA11['beta'] < 9.5)]
dfA3_high = dfA3[(dfA3['beta'] > 8.5) & (dfA3['beta'] < 9.5)]
beta_A11_high = dfA11_high['beta'].mean()
beta_A3_high = dfA3_high['beta'].mean()
shift_high = beta_A11_high - beta_A3_high
print(f"   A1.1 Mittel-Beta:  {beta_A11_high:.3f}°")
print(f"   A3 Mittel-Beta:    {beta_A3_high:.3f}°")
print(f"   → Shift: {shift_high:+.4f}°")


# Statistik über alle Features
all_shifts = np.array([shift_rise, shift_plat, shift_peak, shift_high])
mean_shift = np.mean(all_shifts)
std_shift = np.std(all_shifts, ddof=1)
median_shift = np.median(all_shifts)

print("\n" + "=" * 80)
print("STATISTISCHE AUSWERTUNG:")
print("=" * 80)
print(f"Mittelwert:           Δβ = {mean_shift:+.4f}°")
print(f"Median:               Δβ = {median_shift:+.4f}°")
print(f"Standardabweichung:   u_β = {std_shift:.4f}°")
print(f"|Δβ|:                 {abs(mean_shift):.4f}°")
print()
print(f"Einzelwerte: {', '.join([f'{s:+.4f}°' for s in all_shifts])}")


# ==============================================================================
# METHODE 2: FENSTER-BASIERTE KORRELATIONS-ANALYSE
# ==============================================================================

print("\n" + "=" * 80)
print("FENSTER-BASIERTE KORRELATIONS-ANALYSE")
print("=" * 80)

def calculate_shift_correlation(dfA11, dfA3, beta_min, beta_max, n_test=200):
    """Berechnet Shift mit Cross-Correlation."""
    mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

    df1 = dfA11[mask1].copy()
    df3 = dfA3[mask3].copy()

    beta_grid = np.linspace(beta_min, beta_max, 500)
    f1 = interp1d(df1['beta'], df1['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    f3 = interp1d(df3['beta'], df3['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')

    shifts = np.linspace(-1.0, 1.0, n_test)
    correlations = []

    for shift in shifts:
        R1 = f1(beta_grid)
        R3 = f3(beta_grid - shift)

        valid = np.isfinite(R1) & np.isfinite(R3)
        if np.sum(valid) < 10:
            correlations.append(-1)
            continue

        R1_norm = (R1[valid] - np.mean(R1[valid])) / np.std(R1[valid])
        R3_norm = (R3[valid] - np.mean(R3[valid])) / np.std(R3[valid])

        corr = np.corrcoef(R1_norm, R3_norm)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    return shifts[best_idx], correlations[best_idx]


# Wähle nur strukturreiche Fenster mit hoher Korrelation
windows_good = [
    (3.2, 4.0),   # Anstieg
    (6.5, 7.5),   # Haupt-Peak (sehr gut)
    (8.5, 9.5),   # Hochplateau (sehr gut)
]

shifts_window = []
print("\nAusgewählte strukturreiche Fenster:")
for i, (beta_min, beta_max) in enumerate(windows_good):
    shift, corr = calculate_shift_correlation(dfA11, dfA3, beta_min, beta_max)
    shifts_window.append(shift)
    print(f"  [{beta_min:.1f}, {beta_max:.1f}]°: Δβ = {shift:+.4f}° (Korr: {corr:.4f})")

mean_shift_window = np.mean(shifts_window)
std_shift_window = np.std(shifts_window, ddof=1)

print(f"\nMittelwert:  {mean_shift_window:+.4f}°")
print(f"Std.-Abw.:   {std_shift_window:.4f}°")


# ==============================================================================
# FINALES ERGEBNIS
# ==============================================================================

# Verwende Feature-Methode (robuster)
final_shift = mean_shift
final_uncertainty = std_shift

print("\n" + "=" * 80)
print("FINALES ERGEBNIS")
print("=" * 80)
print()
print(f"Systematischer Shift (Feature-Methode):    Δβ = {final_shift:+.4f}°")
print(f"Winkelunsicherheit:                        u_β = {final_uncertainty:.4f}°")
print(f"|Δβ|:                                      {abs(final_shift):.4f}°")
print()
print("INTERPRETATION:")
print(f"  • A3 liegt systematisch bei kleineren Beta-Werten")
print(f"  • A3 muss um {final_shift:+.4f}° nach rechts verschoben werden")
print(f"  • Die Unsicherheit u_β = {final_uncertainty:.4f}° kommt von der")
print(f"    Variation zwischen verschiedenen Features/Bereichen")
print()
print("Quadratische Addition zur Gesamtunsicherheit:")
u_device = 0.05  # Beispiel
u_total = np.sqrt(u_device**2 + final_uncertainty**2)
print(f"  u_gesamt = sqrt(u_Gerät² + u_β²)")
print(f"  u_gesamt = sqrt(({u_device})² + ({final_uncertainty:.4f})²)")
print(f"  u_gesamt = {u_total:.4f}°")
print("=" * 80)


# ==============================================================================
# PLOTS
# ==============================================================================

# Plot 1: Originalkurven mit markierten Features
fig1, ax = plt.subplots(figsize=(13, 8))

mask1 = dfA11['beta'] <= 10
mask3 = dfA3['beta'] <= 10

ax.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3, label='A1.1 (geglättet)', color='#1f77b4', marker='o',
        markersize=5, markevery=4, alpha=0.9)
ax.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
        linewidth=3, label='A3 (geglättet)', color='#ff7f0e', marker='s',
        markersize=5, markevery=4, alpha=0.9)

# Markiere Features
ax.axvline(beta_A11_rise, color='blue', linestyle=':', linewidth=2, alpha=0.6,
          label=f'A1.1 Anstieg: {beta_A11_rise:.2f}°')
ax.axvline(beta_A3_rise, color='orange', linestyle=':', linewidth=2, alpha=0.6,
          label=f'A3 Anstieg: {beta_A3_rise:.2f}°')

ax.axvline(beta_A11_peak, color='blue', linestyle='--', linewidth=2.5, alpha=0.7)
ax.axvline(beta_A3_peak, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)

# Annotationen
ax.annotate(f'A1.1 Peak\n{beta_A11_peak:.2f}°',
           xy=(beta_A11_peak, dfA11[dfA11['beta'] == np.round(beta_A11_peak, 1)]['R_smooth'].values[0] if len(dfA11[dfA11['beta'] == np.round(beta_A11_peak, 1)]) > 0 else 1000),
           xytext=(beta_A11_peak + 0.5, 900), fontsize=11,
           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate(f'A3 Peak\n{beta_A3_peak:.2f}°',
           xy=(beta_A3_peak, dfA3[dfA3['beta'] == np.round(beta_A3_peak, 1)]['R_smooth'].values[0] if len(dfA3[dfA3['beta'] == np.round(beta_A3_peak, 1)]) > 0 else 1000),
           xytext=(beta_A3_peak - 0.8, 900), fontsize=11,
           arrowprops=dict(arrowstyle='->', color='orange', lw=2))

ax.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
ax.set_ylabel(r'$R$ (Counts)', fontweight='bold')
ax.set_title(r'Originalkurven mit markierten Features', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=12, ncol=2)

plt.tight_layout()
fig1.savefig('aufg7_features_marked.svg', format='svg', bbox_inches='tight', dpi=150)
print("\n✓ Plot 1 gespeichert: aufg7_features_marked.svg")


# Plot 2: Vorher/Nachher mit finalem Shift
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

mask1 = (dfA11['beta'] >= 2) & (dfA11['beta'] <= 10)
mask3 = (dfA3['beta'] >= 2) & (dfA3['beta'] <= 10)

# Vor Korrektur
ax1.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3, label='A1.1', alpha=0.9, color='#1f77b4',
        marker='o', markersize=5, markevery=3)
ax1.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
        linewidth=3, label='A3 (original)', alpha=0.9, color='#ff7f0e',
        marker='s', markersize=5, markevery=3)

ax1.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
ax1.set_ylabel(r'$R$ (Counts)', fontweight='bold')
ax1.set_title(r'VOR Shift-Korrektur', fontweight='bold')
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.legend(fontsize=14)

# Nach Korrektur
dfA3_shifted = dfA3.copy()
dfA3_shifted['beta'] = dfA3['beta'] + final_shift
mask3_shifted = (dfA3_shifted['beta'] >= 2) & (dfA3_shifted['beta'] <= 10)

ax2.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3, label='A1.1', alpha=0.9, color='#1f77b4',
        marker='o', markersize=5, markevery=3)
ax2.plot(dfA3_shifted.loc[mask3_shifted, 'beta'], dfA3_shifted.loc[mask3_shifted, 'R_smooth'],
        linewidth=3, label=f'A3 ($\Delta\\beta = {final_shift:+.4f}^\circ$)',
        alpha=0.9, color='#ff7f0e', marker='s', markersize=5, markevery=3)

ax2.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
ax2.set_ylabel(r'$R$ (Counts)', fontweight='bold')
ax2.set_title(r'NACH Shift-Korrektur', fontweight='bold')
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.legend(fontsize=14)

plt.tight_layout()
fig2.savefig('aufg7_alignment_final.svg', format='svg', bbox_inches='tight', dpi=150)
print("✓ Plot 2 gespeichert: aufg7_alignment_final.svg")


# Plot 3: Zusammenfassung der Shifts
fig3, ax = plt.subplots(figsize=(10, 7))

features = ['Anstieg\n(R>50)', 'Plateau\n(R>400)', 'Haupt-Peak\n(6-8°)', 'Hochplateau\n(8.5-9.5°)']
x_pos = np.arange(len(all_shifts))

bars = ax.bar(x_pos, all_shifts, color=['steelblue', 'coral', 'mediumseagreen', 'gold'],
             alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)

# Mittelwert
ax.axhline(mean_shift, color='red', linestyle='--', linewidth=2.5,
          label=f'Mittelwert: $\Delta\\beta = {mean_shift:+.4f}^\circ$')
ax.axhspan(mean_shift - std_shift, mean_shift + std_shift,
          alpha=0.25, color='red', label=f'$\pm 1\sigma = \pm {std_shift:.4f}^\circ$')
ax.axhline(0, color='black', linewidth=1)

# Beschriftung der Bars mit Werten
for i, (bar, shift_val) in enumerate(zip(bars, all_shifts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.04,
            f'{shift_val:+.3f}°', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=11, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(features, fontsize=13)
ax.set_ylabel(r'$\Delta\beta$ ($^\circ$)', fontweight='bold', fontsize=15)
ax.set_title('Shift-Analyse basierend auf markanten Features', fontweight='bold', fontsize=17)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.legend(fontsize=13, loc='upper right', frameon=True, framealpha=0.95, edgecolor='black')

plt.tight_layout()
fig3.savefig('aufg7_shift_by_features.svg', format='svg', bbox_inches='tight', dpi=150)
print("✓ Plot 3 gespeichert: aufg7_shift_by_features.svg")


# ==============================================================================
# ERGEBNIS-ZUSAMMENFASSUNG
# ==============================================================================

print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG")
print("=" * 80)
print()
print("Die Analyse zeigt:")
print(f"  1. A3 ist im Mittel um {final_shift:+.4f}° verschoben")
print(f"  2. Die Verschiebung variiert zwischen Features (u_β = {final_uncertainty:.4f}°)")
print(f"  3. Dies reflektiert die Unsicherheit der Kristallorientierung")
print()
print("VERWENDUNG IN WEITEREN RECHNUNGEN:")
print(f"  • Winkelunsicherheit: u_β = {final_uncertainty:.4f}°")
print(f"  • Quadratisch zu Geräteunsicherheit addieren")
print(f"  • Bei systematischer Korrektur: A3-Winkel um {final_shift:+.4f}° korrigieren")
print()
print("=" * 80)

plt.show()

