import os

import matplotlib.pyplot as plt
import numpy as np

# Non-interaktives Backend für schnelles Rendering
plt.switch_backend("Agg")


def plot_bland_altman(
    data1: np.ndarray,
    data2: np.ndarray,
    mask: np.ndarray,
    method1_name: str = "VoxelMorph",
    method2_name: str = "BSpline",
    parameter_name: str = "Regional Ventilation (RVent)",
    unit: str = "%",
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Erstellt ein Bland-Altman-Diagramm zum Vergleich zweier PREFUL-Parameterkarten

    oder Voxel-/Patienten-Datensätze.

    Parameters:
        data1 (np.ndarray): Messwerte der ersten Methode (z. B. VoxelMorph).
        data2 (np.ndarray): Messwerte der Referenzmethode (z. B. BSpline oder GOREG).
        mask (np.ndarray, optional): Binäre Lungenmaske.
        method1_name (str): Name der ersten Methode.
        method2_name (str): Name der Referenzmethode.
        parameter_name (str): Name des evaluierten Parameters (z. B. 'RVent', 'FVL-CM').
        unit (str): Einheit des Parameters (z. B. '%', 'mL/min/100mL').
        save_path (str, optional): Pfad zum Speichern des Plots.
        dpi (int): Auflösung des gespeicherten Bildes.

    Returns:
        plt.Figure: Matplotlib Figure-Objekt.
    """
    # Maskierung und Auslesen der Fließkommawerte
    if mask is not None:
        val1 = data1[mask].astype(float).flatten()
        val2 = data2[mask].astype(float).flatten()
    else:
        val1 = data1.astype(float).flatten()
        val2 = data2.astype(float).flatten()

    # Nur gültige (nicht-NaN) Werte betrachten
    valid = np.isfinite(val1) & np.isfinite(val2)
    val1 = val1[valid]
    val2 = val2[valid]

    if len(val1) == 0:
        raise ValueError("Keine gültigen Datenpunkte nach Maskierung vorhanden.")

    # Berechnungen für Bland-Altman
    means = (val1 + val2) / 2.0
    diffs = val1 - val2  # Method1 - Method2 (Systematische Abweichung / Bias)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Erstellung der Figur
    fig, ax = plt.subplots(figsize=(8, 6))

    # Punktdichte anpassen je nach Voxelanzahl
    if len(means) > 5000:
        ax.scatter(means, diffs, alpha=0.15, color="#2b5c8f", edgecolors="none", s=12)
    else:
        ax.scatter(
            means,
            diffs,
            alpha=0.5,
            color="#2b5c8f",
            edgecolors="k",
            linewidths=0.5,
            s=25,
        )

    # Nulllinie (perfekte Übereinstimmung)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)

    # Bias Line (Mittlere Differenz)
    ax.axhline(
        mean_diff,
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"Bias: {mean_diff:+.3f} {unit}",
    )

    # Limits of Agreement (LoA ± 1.96 SD)
    ax.axhline(
        loa_upper,
        color="#7570b3",
        linestyle="--",
        linewidth=1.5,
        label=f"+1.96 SD: {loa_upper:+.3f} {unit}",
    )
    ax.axhline(
        loa_lower,
        color="#7570b3",
        linestyle="--",
        linewidth=1.5,
        label=f"-1.96 SD: {loa_lower:+.3f} {unit}",
    )

    # Schattierung des 95% LoA-Bereichs
    ax.axhspan(loa_lower, loa_upper, color="#7570b3", alpha=0.08)

    # Achsen- und Titelbeschriftung
    ax.set_xlabel(
        f"Mittelwert aus {method1_name} und {method2_name} [{unit}]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Differenz ({method1_name} - {method2_name}) [{unit}]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"Bland-Altman-Analyse: {parameter_name}\n{method1_name} vs. {method2_name}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Statistikkasten in der Ecke
    stats_text = (
        f"Anzahl Voxel/Scans: {len(means):,}\n"
        f"Mean Diff (Bias): {mean_diff:+.3f} {unit}\n"
        f"Std Diff (SD): {std_diff:.3f} {unit}\n"
        f"95% LoA: [{loa_lower:+.3f}, {loa_upper:+.3f}] {unit}"
    )
    ax.text(
        0.03,
        0.05,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "gray",
        },
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_multi_bland_altman(
    data_dict: dict,
    mask: np.ndarray,
    method1_name: str = "VoxelMorph",
    method2_name: str = "BSpline",
    save_path: str = "Results/bland_altman_multi_panel.png",
    dpi: int = 300,
) -> plt.Figure:
    """Erstellt ein Multi-Panel Bland-Altman-Grid für mehrere PREFUL-Parameterkarten.

    data_dict Beispiel:
        {
            'RVent': (voxelmorph_rvent, bspline_rvent, '%'),
            'FVL-CM': (voxelmorph_fvl, bspline_fvl, '%'),
            'QQ': (voxelmorph_qq, bspline_qq, 'mL/min/100mL')
        }
    """
    num_params = len(data_dict)
    cols = min(num_params, 3)
    rows = int(np.ceil(num_params / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if num_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (param_name, (d1, d2, unit)) in enumerate(data_dict.items()):
        ax = axes[idx]

        if mask is not None:
            v1 = d1[mask].astype(float).flatten()
            v2 = d2[mask].astype(float).flatten()
        else:
            v1 = d1.astype(float).flatten()
            v2 = d2.astype(float).flatten()

        valid = np.isfinite(v1) & np.isfinite(v2)
        v1, v2 = v1[valid], v2[valid]

        m = (v1 + v2) / 2.0
        d = v1 - v2
        mb = float(np.mean(d))
        sb = float(np.std(d, ddof=1))
        hi = mb + 1.96 * sb
        lo = mb - 1.96 * sb

        if len(m) > 5000:
            ax.scatter(m, d, alpha=0.15, color="#2b5c8f", edgecolors="none", s=10)
        else:
            ax.scatter(
                m, d, alpha=0.5, color="#2b5c8f", edgecolors="k", linewidths=0.5, s=20
            )

        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(
            mb, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Bias: {mb:+.2f}"
        )
        ax.axhline(
            hi,
            color="#7570b3",
            linestyle="--",
            linewidth=1.2,
            label=f"+1.96 SD: {hi:+.2f}",
        )
        ax.axhline(
            lo,
            color="#7570b3",
            linestyle="--",
            linewidth=1.2,
            label=f"-1.96 SD: {lo:+.2f}",
        )
        ax.axhspan(lo, hi, color="#7570b3", alpha=0.08)

        ax.set_xlabel(f"Mittelwert [{unit}]", fontsize=10)
        ax.set_ylabel(
            f"Differenz ({method1_name} - {method2_name}) [{unit}]", fontsize=10
        )
        ax.set_title(f"{param_name}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    for idx in range(num_params, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        f"Bland-Altman-Analysen: {method1_name} vs. {method2_name}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
