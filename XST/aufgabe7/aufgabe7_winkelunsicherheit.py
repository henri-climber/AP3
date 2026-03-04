"""
Aufgabe 7: Winkelunsicherheit aus horizontaler Verschiebung
zwischen zwei Messkurven (A1.1 vs A3) - Röntgenstrahlung an NaCl-Kristall

Methode: Global Alignment / Kurven gegeneinander verschieben
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from typing import Tuple, List, Dict

# ==============================================================================
# 1. DATEN EINLESEN UND VORVERARBEITUNG
# ==============================================================================

def load_and_preprocess_data(filepath: str, sheet_A11: str, sheet_A3: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lädt beide Datensätze und führt Vorverarbeitung durch.

    Returns:
        dfA11, dfA3: Bereinigte DataFrames
    """
    # Daten einlesen
    dfA11 = pd.read_excel(filepath, sheet_name=sheet_A11)
    dfA3 = pd.read_excel(filepath, sheet_name=sheet_A3)

    # Vorverarbeitung für beide DataFrames
    for df in [dfA11, dfA3]:
        # NaNs entfernen
        df.dropna(subset=['beta', 'R'], inplace=True)
        # Zu float konvertieren
        df['beta'] = df['beta'].astype(float)
        df['R'] = df['R'].astype(float)
        # Nach beta sortieren
        df.sort_values('beta', inplace=True)
        df.reset_index(drop=True, inplace=True)

    return dfA11, dfA3


def smooth_data(df: pd.DataFrame, window_length: int = 11, polyorder: int = 3) -> pd.DataFrame:
    """
    Glättet die R-Werte mit Savitzky-Golay Filter.

    Args:
        df: DataFrame mit 'beta' und 'R'
        window_length: Fensterbreite (muss ungerade sein)
        polyorder: Polynomgrad

    Returns:
        DataFrame mit zusätzlicher Spalte 'R_smooth'
    """
    df_copy = df.copy()

    # Sicherstellen, dass genug Datenpunkte vorhanden sind
    if len(df_copy) < window_length:
        window_length = len(df_copy) if len(df_copy) % 2 == 1 else len(df_copy) - 1
        if window_length < polyorder + 1:
            # Fallback auf einfaches rolling mean
            df_copy['R_smooth'] = df_copy['R'].rolling(window=3, center=True, min_periods=1).mean()
            return df_copy

    df_copy['R_smooth'] = savgol_filter(df_copy['R'], window_length, polyorder)
    return df_copy


def find_common_range(dfA11: pd.DataFrame, dfA3: pd.DataFrame) -> Tuple[float, float]:
    """
    Findet den gemeinsamen Beta-Bereich beider Kurven.
    """
    beta_min = max(dfA11['beta'].min(), dfA3['beta'].min())
    beta_max = min(dfA11['beta'].max(), dfA3['beta'].max())
    return beta_min, beta_max


# ==============================================================================
# 2. SHIFT-SCHÄTZUNG
# ==============================================================================

def z_normalize(data: np.ndarray) -> np.ndarray:
    """
    Z-Score Normalisierung: (x - mean) / std
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return data - mean
    return (data - mean) / std


def calculate_shift_for_window(
    dfA11: pd.DataFrame,
    dfA3: pd.DataFrame,
    beta_min: float,
    beta_max: float,
    grid_step: float = 0.002,
    shift_range: Tuple[float, float] = (-0.5, 0.5),
    use_smooth: bool = True
) -> Tuple[float, float]:
    """
    Berechnet den optimalen Shift für ein bestimmtes Beta-Fenster.

    Args:
        dfA11, dfA3: DataFrames mit Messdaten
        beta_min, beta_max: Fensterbereich
        grid_step: Schrittweite für Interpolationsgitter
        shift_range: Suchbereich für Shift (min, max) in Grad
        use_smooth: Verwende geglättete Daten

    Returns:
        optimal_shift: Bester Shift Δβ
        min_loss: Minimaler Loss-Wert
    """
    # Daten im Fenster auswählen
    mask_A11 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    mask_A3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

    df1_window = dfA11[mask_A11].copy()
    df3_window = dfA3[mask_A3].copy()

    if len(df1_window) < 5 or len(df3_window) < 5:
        return np.nan, np.inf

    # Wähle Daten (geglättet oder original)
    R_col = 'R_smooth' if use_smooth and 'R_smooth' in df1_window.columns else 'R'

    # Interpolationsfunktionen erstellen
    try:
        f1 = interp1d(df1_window['beta'], df1_window[R_col],
                      kind='cubic', bounds_error=False, fill_value='extrapolate')
        f3 = interp1d(df3_window['beta'], df3_window[R_col],
                      kind='cubic', bounds_error=False, fill_value='extrapolate')
    except:
        # Fallback auf lineare Interpolation
        f1 = interp1d(df1_window['beta'], df1_window[R_col],
                      kind='linear', bounds_error=False, fill_value='extrapolate')
        f3 = interp1d(df3_window['beta'], df3_window[R_col],
                      kind='linear', bounds_error=False, fill_value='extrapolate')

    # Gemeinsames feines Gitter
    beta_grid = np.arange(beta_min, beta_max, grid_step)

    def loss_function(shift: float) -> float:
        """
        Berechnet MSE zwischen R1(beta) und R3(beta - shift).

        Interpretation: Wenn A3 bei kleineren beta-Werten liegt als A1.1,
        brauchen wir einen positiven shift, um A3 nach rechts zu verschieben.
        Vergleich: R1(beta) mit R3(beta - shift)
        → R3 wird bei (beta - shift) ausgewertet, also bei kleineren Werten
        → entspricht Verschiebung von A3 nach rechts um +shift
        """
        # A1.1 Werte am Gitter
        R1_interp = f1(beta_grid)
        # A3 Werte am verschobenen Gitter (beta - shift)
        # Das bedeutet: Wenn shift=+0.2, dann wird R3 bei beta-0.2 ausgewertet
        # Also wird der R-Wert von kleineren beta zu größeren beta verschoben
        R3_interp = f3(beta_grid - shift)

        # Nur gültige Werte verwenden
        valid_mask = np.isfinite(R1_interp) & np.isfinite(R3_interp)
        if np.sum(valid_mask) < 5:
            return 1e10

        R1_valid = R1_interp[valid_mask]
        R3_valid = R3_interp[valid_mask]

        # Z-Score Normalisierung
        R1_norm = z_normalize(R1_valid)
        R3_norm = z_normalize(R3_valid)

        # MSE berechnen
        mse = np.mean((R1_norm - R3_norm)**2)
        return mse

    # Optimierung
    result = minimize_scalar(loss_function, bounds=shift_range, method='bounded')

    return result.x, result.fun


# ==============================================================================
# 3. MEHRERE FENSTER ANALYSIEREN
# ==============================================================================

def analyze_multiple_windows(
    dfA11: pd.DataFrame,
    dfA3: pd.DataFrame,
    windows: List[Tuple[float, float]],
    grid_step: float = 0.002,
    shift_range: Tuple[float, float] = (-0.5, 0.5),
    use_smooth: bool = True
) -> Dict:
    """
    Analysiert mehrere Beta-Fenster und berechnet Statistiken.

    Returns:
        Dictionary mit Ergebnissen
    """
    results = {
        'windows': [],
        'shifts': [],
        'losses': []
    }

    print("Analysiere Fenster:")
    print("-" * 70)

    for i, (beta_min, beta_max) in enumerate(windows):
        shift, loss = calculate_shift_for_window(
            dfA11, dfA3, beta_min, beta_max,
            grid_step=grid_step,
            shift_range=shift_range,
            use_smooth=use_smooth
        )

        results['windows'].append((beta_min, beta_max))
        results['shifts'].append(shift)
        results['losses'].append(loss)

        print(f"Fenster {i+1}: [{beta_min:.1f}°, {beta_max:.1f}°] → "
              f"Δβ = {shift:+.4f}° (Loss: {loss:.4f})")

    # Statistiken berechnen
    shifts_array = np.array(results['shifts'])
    valid_shifts = shifts_array[np.isfinite(shifts_array)]

    if len(valid_shifts) > 0:
        results['mean_shift'] = np.mean(valid_shifts)
        results['std_shift'] = np.std(valid_shifts, ddof=1) if len(valid_shifts) > 1 else 0
        results['abs_mean_shift'] = np.abs(results['mean_shift'])
        results['max_deviation'] = np.max(np.abs(valid_shifts - results['mean_shift']))
        results['conservative_uncertainty'] = max(results['std_shift'], results['max_deviation'])
    else:
        results['mean_shift'] = np.nan
        results['std_shift'] = np.nan
        results['abs_mean_shift'] = np.nan
        results['max_deviation'] = np.nan
        results['conservative_uncertainty'] = np.nan

    print("-" * 70)
    print(f"\nStatistische Auswertung:")
    print(f"  Δβ (Mittelwert):              {results['mean_shift']:+.4f}°")
    print(f"  |Δβ| (Betrag):                {results['abs_mean_shift']:.4f}°")
    print(f"  Standardabweichung (u_beta):  {results['std_shift']:.4f}°")
    print(f"  Maximale Abweichung:          {results['max_deviation']:.4f}°")
    print(f"  Konservative Unsicherheit:    {results['conservative_uncertainty']:.4f}°")

    return results


# ==============================================================================
# 4. PLOTTING-FUNKTIONEN
# ==============================================================================

def setup_plot_style():
    """
    Setzt einheitlichen Plot-Style.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11


def plot_original_curves(dfA11: pd.DataFrame, dfA3: pd.DataFrame,
                        beta_min: float = 2.0, beta_max: float = 10.0,
                        windows: List[Tuple[float, float]] = None):
    """
    Plot A: Originalkurven A1.1 und A3 im relevanten Bereich.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Daten filtern
    mask_A11 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    mask_A3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

    # Originaldaten plotten
    ax.plot(dfA11.loc[mask_A11, 'beta'], dfA11.loc[mask_A11, 'R'],
            marker='o', linewidth=1.5, markersize=4, label='A1.1 (Original)',
            alpha=0.7, color='#1f77b4')
    ax.plot(dfA3.loc[mask_A3, 'beta'], dfA3.loc[mask_A3, 'R'],
            marker='s', linewidth=1.5, markersize=4, label='A3 (Original)',
            alpha=0.7, color='#ff7f0e')

    # Geglättete Daten plotten (falls vorhanden)
    if 'R_smooth' in dfA11.columns:
        ax.plot(dfA11.loc[mask_A11, 'beta'], dfA11.loc[mask_A11, 'R_smooth'],
                linewidth=2.5, label='A1.1 (geglättet)', color='#1f77b4')
    if 'R_smooth' in dfA3.columns:
        ax.plot(dfA3.loc[mask_A3, 'beta'], dfA3.loc[mask_A3, 'R_smooth'],
                linewidth=2.5, label='A3 (geglättet)', color='#ff7f0e')

    # Fenster markieren (falls angegeben)
    if windows:
        for i, (w_min, w_max) in enumerate(windows):
            ax.axvspan(w_min, w_max, alpha=0.1, color=f'C{i+2}',
                      label=f'Fenster {i+1}' if i < 4 else '')

    ax.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
    ax.set_ylabel(r'$R$ (Counts)', fontweight='bold')
    ax.set_title(r'Originalkurven A1.1 und A3 (vor Shift-Korrektur)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black')
    plt.tight_layout()

    return fig


def plot_corrected_curves(dfA11: pd.DataFrame, dfA3: pd.DataFrame,
                         shift: float,
                         beta_min: float = 2.0, beta_max: float = 10.0):
    """
    Plot B: Kurven nach Korrektur um Δβ_mean.
    A3 wird nach rechts verschoben (shift wird zu beta addiert).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Daten filtern
    mask_A11 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    # A3: beta-Werte um shift korrigieren (nach rechts verschieben wenn shift positiv)
    dfA3_shifted = dfA3.copy()
    dfA3_shifted['beta'] = dfA3_shifted['beta'] + shift
    mask_A3 = (dfA3_shifted['beta'] >= beta_min) & (dfA3_shifted['beta'] <= beta_max)

    # Geglättete Daten verwenden (falls vorhanden)
    R_col = 'R_smooth' if 'R_smooth' in dfA11.columns else 'R'

    # Plotten
    ax.plot(dfA11.loc[mask_A11, 'beta'], dfA11.loc[mask_A11, R_col],
            marker='o', linewidth=2, markersize=4, label='A1.1',
            alpha=0.8, color='#1f77b4')
    ax.plot(dfA3_shifted.loc[mask_A3, 'beta'], dfA3_shifted.loc[mask_A3, R_col],
            marker='s', linewidth=2, markersize=4,
            label=f'A3 (korrigiert um $\\Delta\\beta = {shift:+.4f}^\\circ$)',
            alpha=0.8, color='#ff7f0e')

    ax.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
    ax.set_ylabel(r'$R$ (Counts)', fontweight='bold')
    ax.set_title(r'Kurven nach Shift-Korrektur (sollten besser \"ubereinanderliegen)',
                fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black')
    plt.tight_layout()

    return fig


def plot_window_comparison(dfA11: pd.DataFrame, dfA3: pd.DataFrame,
                          window: Tuple[float, float], shift: float,
                          window_idx: int):
    """
    Plot C: Zoom auf ein einzelnes Fenster (vorher/nachher).
    """
    beta_min, beta_max = window
    padding = 0.2  # Extra Bereich für bessere Visualisierung

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Daten filtern
    mask_A11 = (dfA11['beta'] >= beta_min - padding) & (dfA11['beta'] <= beta_max + padding)
    mask_A3 = (dfA3['beta'] >= beta_min - padding) & (dfA3['beta'] <= beta_max + padding)

    R_col = 'R_smooth' if 'R_smooth' in dfA11.columns else 'R'

    # Plot 1: Vor Korrektur
    ax1.plot(dfA11.loc[mask_A11, 'beta'], dfA11.loc[mask_A11, R_col],
            marker='o', linewidth=2, markersize=5, label='A1.1', alpha=0.8)
    ax1.plot(dfA3.loc[mask_A3, 'beta'], dfA3.loc[mask_A3, R_col],
            marker='s', linewidth=2, markersize=5, label='A3', alpha=0.8)
    ax1.axvspan(beta_min, beta_max, alpha=0.15, color='gray')
    ax1.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
    ax1.set_ylabel(r'$R$ (Counts)', fontweight='bold')
    ax1.set_title(f'Fenster {window_idx}: Vor Korrektur', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()

    # Plot 2: Nach Korrektur
    dfA3_shifted = dfA3.copy()
    dfA3_shifted['beta'] = dfA3_shifted['beta'] + shift
    mask_A3_shifted = (dfA3_shifted['beta'] >= beta_min - padding) & \
                      (dfA3_shifted['beta'] <= beta_max + padding)

    ax2.plot(dfA11.loc[mask_A11, 'beta'], dfA11.loc[mask_A11, R_col],
            marker='o', linewidth=2, markersize=5, label='A1.1', alpha=0.8)
    ax2.plot(dfA3_shifted.loc[mask_A3_shifted, 'beta'],
            dfA3_shifted.loc[mask_A3_shifted, R_col],
            marker='s', linewidth=2, markersize=5,
            label=f'A3 ($\\Delta\\beta={shift:+.4f}^\\circ$)', alpha=0.8)
    ax2.axvspan(beta_min + shift, beta_max + shift, alpha=0.15, color='gray')
    ax2.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
    ax2.set_ylabel(r'$R$ (Counts)', fontweight='bold')
    ax2.set_title(f'Fenster {window_idx}: Nach Korrektur', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_shift_summary(results: Dict):
    """
    Zusätzlicher Plot: Übersicht der Shifts pro Fenster.
    """
    windows = results['windows']
    shifts = np.array(results['shifts'])
    mean_shift = results['mean_shift']
    std_shift = results['std_shift']

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(windows))
    window_labels = [f"[{w[0]:.1f}, {w[1]:.1f}]" for w in windows]

    # Bars plotten
    bars = ax.bar(x_pos, shifts, alpha=0.7, color='steelblue', edgecolor='black')

    # Mittelwert-Linie
    ax.axhline(mean_shift, color='red', linestyle='--', linewidth=2,
              label=f'Mittelwert: {mean_shift:+.4f}$^\\circ$')

    # Unsicherheitsbereich
    ax.axhspan(mean_shift - std_shift, mean_shift + std_shift,
              alpha=0.2, color='red', label=f'$\\pm 1\\sigma$: $\\pm${std_shift:.4f}$^\\circ$')

    ax.set_xlabel(r'Fenster ($\beta$-Bereich in $^\circ$)', fontweight='bold')
    ax.set_ylabel(r'$\Delta\beta$ ($^\circ$)', fontweight='bold')
    ax.set_title('Geschätzte Shifts pro Fenster', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(window_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    return fig


# ==============================================================================
# 5. HAUPTANALYSE
# ==============================================================================

def main():
    """
    Hauptfunktion: Führt die komplette Analyse durch.
    """
    print("=" * 70)
    print("AUFGABE 7: WINKELUNSICHERHEIT AUS HORIZONTALER VERSCHIEBUNG")
    print("=" * 70)
    print()

    # Daten laden
    print("1. Lade und verarbeite Daten...")
    filepath = '../data/xst.xlsx'
    dfA11, dfA3 = load_and_preprocess_data(filepath, 'A1.1', 'A3')
    print(f"   A1.1: {len(dfA11)} Datenpunkte, Beta: {dfA11['beta'].min():.1f}° - {dfA11['beta'].max():.1f}°")
    print(f"   A3:   {len(dfA3)} Datenpunkte, Beta: {dfA3['beta'].min():.1f}° - {dfA3['beta'].max():.1f}°")

    # Gemeinsamen Bereich finden
    beta_min_common, beta_max_common = find_common_range(dfA11, dfA3)
    print(f"   Gemeinsamer Bereich: {beta_min_common:.1f}° - {beta_max_common:.1f}°")
    print()

    # Daten glätten
    print("2. Glätte Daten mit Savitzky-Golay Filter...")
    dfA11 = smooth_data(dfA11, window_length=11, polyorder=3)
    dfA3 = smooth_data(dfA3, window_length=9, polyorder=3)
    print("   ✓ Glättung abgeschlossen")
    print()

    # Fenster definieren (strukturreiche Bereiche)
    windows = [
        (3.4, 4.3),   # Steiler Anstieg
        (4.2, 5.6),   # Plateau-Übergang
        (5.6, 6.4),   # Peak-Bereich
        (6.4, 7.4),   # Abstieg
        (7.4, 8.5),   # Zweiter Peak
        (8.5, 9.5),   # Hochplateau
    ]

    print("3. Berechne Shifts für verschiedene Fenster...")
    results = analyze_multiple_windows(
        dfA11, dfA3, windows,
        grid_step=0.002,
        shift_range=(-0.5, 0.5),
        use_smooth=True
    )
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print(f"Die zusätzliche Winkelunsicherheit aus der Kristallorientierung beträgt:")
    print(f"  u_beta = {results['std_shift']:.4f}°")
    print()
    print(f"Die Kurve A3 ist um durchschnittlich {results['mean_shift']:+.4f}° verschoben.")
    print()
    print(f"Diese Unsicherheit muss quadratisch zur geräte-bedingten Unsicherheit")
    print(f"hinzugefügt werden:")
    print(f"  u_total = sqrt(u_alt² + u_beta²)")
    print()
    print(f"Beispiel: Mit u_alt = 0.05° ergibt sich:")
    u_alt = 0.05
    u_total = np.sqrt(u_alt**2 + results['std_shift']**2)
    print(f"  u_total = sqrt({u_alt}² + {results['std_shift']:.4f}²) = {u_total:.4f}°")
    print("=" * 70)
    print()

    # Plots erstellen
    print("4. Erstelle Plots...")
    setup_plot_style()

    # Plot A: Originalkurven
    print("   - Plot A: Originalkurven mit Fenstern")
    fig_A = plot_original_curves(dfA11, dfA3, beta_min=2.0, beta_max=10.0, windows=windows)
    fig_A.savefig('plot_A_originalkurven.pdf', dpi=150, bbox_inches='tight')
    fig_A.savefig('plot_A_originalkurven.svg', format='svg', bbox_inches='tight')
    print("      → Gespeichert als plot_A_originalkurven.pdf/.svg")

    # Plot B: Korrigierte Kurven
    print("   - Plot B: Kurven nach Shift-Korrektur")
    fig_B = plot_corrected_curves(dfA11, dfA3, results['mean_shift'],
                                  beta_min=2.0, beta_max=10.0)
    fig_B.savefig('plot_B_korrigierte_kurven.pdf', dpi=150, bbox_inches='tight')
    fig_B.savefig('plot_B_korrigierte_kurven.svg', format='svg', bbox_inches='tight')
    print("      → Gespeichert als plot_B_korrigierte_kurven.pdf/.svg")

    # Plot C: Fenster-Vergleiche (erste 3 Fenster als Beispiele)
    print("   - Plot C: Fenster-Vergleiche (Beispiele)")
    for i in range(min(3, len(windows))):
        fig_C = plot_window_comparison(dfA11, dfA3, windows[i],
                                      results['shifts'][i], i+1)
        fig_C.savefig(f'plot_C_fenster_{i+1}.pdf', dpi=150, bbox_inches='tight')
        fig_C.savefig(f'plot_C_fenster_{i+1}.svg', format='svg', bbox_inches='tight')
        print(f"      → Gespeichert als plot_C_fenster_{i+1}.pdf/.svg")

    # Zusatzplot: Shift-Übersicht
    print("   - Zusatzplot: Shift-Übersicht")
    fig_summary = plot_shift_summary(results)
    fig_summary.savefig('plot_shift_summary.pdf', dpi=150, bbox_inches='tight')
    fig_summary.savefig('plot_shift_summary.svg', format='svg', bbox_inches='tight')
    print("      → Gespeichert als plot_shift_summary.pdf/.svg")

    print()
    print("✓ Analyse abgeschlossen!")
    print()

    # Zeige Plots
    plt.show()

    return results, dfA11, dfA3


if __name__ == "__main__":
    results, dfA11, dfA3 = main()

