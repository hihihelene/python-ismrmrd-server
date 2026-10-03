import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim


def image_sharpness_metric(image: np.ndarray, mask: np.ndarray) -> float:
    """Image Sharpness Measure (ISM / FM) nach
    De K & Masilamani (2013) https://doi.org/10.1016/j.proeng.2013.09.086
    unter Nutzung des Masken-Objekts als Bounding Box.

    Formel aus Abschnitt 3.1 / Gl. (1):
      1. F = 2D-FFT(Bild)
      2. Fc = fftshift(F)
      3. AF = abs(Fc)
      4. M = max(AF)
      5. thres = M / 1000.0
      6. TH = Anzahl Pixels in AF > thres
      7. FM = TH / (M * N)
    """
    # Bounding Box aus der Maske bestimmen (Min/Max Indizes der Lunge)
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0.0

    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()

    # Bild auf den durch die Maske definierten Bereich zuschneiden (Rechteck)
    cropped_image = image[ymin : ymax + 1, xmin : xmax + 1].astype(float)
    M_dim, N_dim = cropped_image.shape

    # Schritte 1 bis 3: 2D-FFT auf dem zugeschnittenen Lungenbereich durchführen,
    # Zentrierung und Betragsbildung
    F = np.fft.fft2(cropped_image)
    Fc = np.fft.fftshift(F)
    AF = np.abs(Fc)

    # Schritte 4 & 5: Maximalwert (DC-Komponente) & Schwellenwert M/1000
    M_val = np.max(AF)
    thres = M_val / 1000.0

    # Schritt 6: Anzahl der Pixel oberhalb des Schwellenwerts (T_H)
    TH = np.sum(AF > thres)

    # Schritt 7: Berechne "Image Quality measure"(FM)
    FM = TH / (M_dim * N_dim)
    return float(FM)


def coefficient_of_variation(image: np.ndarray, mask: np.ndarray) -> float:
    """Variationskoeffizient des Parenchyms (CV = std / mean).
    https://doi.org/10.1002/mrm.26526 Statistical Analysis(1)"""
    values = image[mask]
    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0
    CV = float(np.std(values) / mean_val)
    return CV


def fractional_ventilation(image_series: np.ndarray, mask: np.ndarray) -> float:
    """Fraktionelle Ventilation nach Voskrebenzev et al. 2016
    https://doi.org/10.1002/mrm.26047 Methods(2)"""
    mean_signal = image_series.mean(axis=(0, 1))

    # argmax = Exspiration (hohes Signal), argmin = Inspiration (niedriges Signal)
    s_exp = image_series[:, :, np.argmax(mean_signal)].astype(float)
    s_insp = image_series[:, :, np.argmin(mean_signal)].astype(float)

    # Formel: FV = (S_exp - S_insp) / S_exp
    s_exp_safe = np.where(s_exp == 0, 1e-6, s_exp)
    fv_map = (s_exp - s_insp) / s_exp_safe

    return float(np.mean(fv_map[mask]))


def local_normalized_cross_correlation(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray | None = None,
    win_size: int = 9,
    eps: float = 1e-5,
) -> float:
    """Berechnet die Lokale Normierte Kreuzkorrelation (LNCC) zwischen zwei 2D-Bildern.
    Anpassung der Funktion der NCC Berechnung unter Voxelmorph-Neurite:
    neurite.nn.modules.ncc

    Parameters:
        image1: Registriertes Bild I (2D numpy.ndarray)
        image2: Referenzbild J (2D numpy.ndarray)
        mask: Binäre Maske (optional, z. B. Lungenmaske)
        win_size: Fenstergröße für lokale Mittelwerte (Standard: 9)

    Returns:
        float: Mittlere LNCC im ausgewerteten Bereich (1.0 = perfekte lokale Ähnlichkeit).
    """
    I = image1.astype(np.float64)
    J = image2.astype(np.float64)

    if I.shape != J.shape:
        raise ValueError(f"Shape mismatch: {I.shape} vs {J.shape}")

    if mask is not None and mask.shape != I.shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {I.shape}")

    kernel = np.ones((win_size, win_size), dtype=np.float64)
    N = win_size**2

    # Local sums, matching the training implementation.
    I_sum = ndimage.convolve(I, kernel, mode="constant", cval=0.0)
    J_sum = ndimage.convolve(J, kernel, mode="constant", cval=0.0)
    I2_sum = ndimage.convolve(I**2, kernel, mode="constant", cval=0.0)
    J2_sum = ndimage.convolve(J**2, kernel, mode="constant", cval=0.0)
    IJ_sum = ndimage.convolve(I * J, kernel, mode="constant", cval=0.0)

    cross = IJ_sum - I_sum * J_sum / N
    I_var = I2_sum - I_sum**2 / N
    J_var = J2_sum - J_sum**2 / N

    lncc_map = cross**2 / (I_var * J_var + eps)

    if mask is not None:
        mask = mask.astype(bool)

        if not np.any(mask):
            raise ValueError("Mask contains no valid pixels.")

        return float(np.mean(lncc_map[mask]))

    return float(np.mean(lncc_map))


def dice_coefficient(
    mask1: np.ndarray,
    mask2: np.ndarray,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
) -> float:
    """Berechnet den Dice-Koeffizienten zwischen zwei binären 2D-Masken.
    Anpassung der Funktion des Dice unter Voxelmorph-Neurite:
    neurite.nn.modules.dice

    Der Dice-Koeffizient misst die räumliche Überlappung zweier binärer
    Segmentierungen:

        Dice = (2 * |A ∩ B| + smooth_numerator)
               / (|A| + |B| + smooth_denominator)

    Die Berechnung entspricht der grundlegenden Dice-Definition der
    Referenzimplementierung. Die kleinen Smoothing-Konstanten dienen der
    numerischen Stabilität und entsprechen deren Standardwerten.

    Beide Eingabemasken werden als binäre Masken interpretiert, d. h. alle
    Werte ungleich null gelten als Vordergrund.

    Für zwei leere Masken wird ein Dice-Wert von 1.0 zurückgegeben, da beide
    Segmentierungen exakt übereinstimmen.

    Args:
        mask1: Erste binäre 2D-Maske.
        mask2: Zweite binäre 2D-Maske mit derselben Form wie ``mask1``.
        smooth_numerator: Kleine Konstante für den Zähler.
            Standard ist ``1e-12`` und entspricht der
            Referenzimplementierung.
        smooth_denominator: Kleine Konstante für den Nenner.
            Standard ist ``1e-12`` und entspricht der
            Referenzimplementierung.

    Returns:
        Der Dice-Koeffizient als ``float`` im Bereich ``[0, 1]``.

    Raises:
        ValueError: Wenn die beiden Masken unterschiedliche Formen haben.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"mask shape mismatch: {mask1.shape} vs {mask2.shape}")

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    intersection = np.sum(m1 & m2)
    size1 = np.sum(m1)
    size2 = np.sum(m2)

    dice = (2.0 * intersection + smooth_numerator) / (
        size1 + size2 + smooth_denominator
    )

    return float(dice)


def structural_similarity_index(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = 1.0,
) -> float:
    """Berechnet den Structural Similarity Index (SSIM).
    Nach Wang et al. (2004) https://doi.org/10.1109/TIP.2003.819861
    (B. The SSIM Index)

    Der SSIM wird zunächst als vollständige lokale SSIM-Karte berechnet.
    Wenn eine Maske angegeben wird, wird anschließend nur der mittlere
    SSIM innerhalb der maskierten Region ausgewertet. Die Maske beeinflusst
    daher nicht die lokalen SSIM-Statistiken selbst.

    Args:
        image1: Erstes 2D-Bild, z. B. das registrierte Bild.
        image2: Zweites 2D-Bild, z. B. das Referenzbild.
        mask: Optionale binäre 2D-Maske. Wenn angegeben, wird der SSIM nur
            innerhalb der maskierten Region gemittelt.
        data_range: Dynamischer Wertebereich der Bildintensitäten.
            Für Bilder im Bereich [0, 1] sollte ``1.0`` verwendet werden.
            Für 8-Bit-Bilder beispielsweise ``255.0``.

    Returns:
        Mittlerer SSIM-Wert. Ein Wert von ``1.0`` entspricht identischen
        lokalen Strukturen; kleinere Werte zeigen zunehmende Unterschiede.

    Raises:
        ValueError: Wenn die Bildformen oder die Maskenform nicht
            übereinstimmen oder ``data_range`` nicht positiv ist.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"image shape mismatch: {image1.shape} vs {image2.shape}")

    if image1.ndim != 2:
        raise ValueError(f"Expected 2D images, got ndim={image1.ndim}")

    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    if mask is not None and mask.shape != image1.shape:
        raise ValueError(f"mask/image shape mismatch: {mask.shape} vs {image1.shape}")

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    _, ssim_map = ssim(
        image1,
        image2,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )

    if mask is not None:
        mask = mask.astype(bool)

        if np.any(mask):
            return float(np.mean(ssim_map[mask]))

    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Ebene 1 - Anatomische Registrierungsgenauigkeit
# Empfohlene Mindestanforderung (Inferenz/abschließende Bewertung): TRE, Dice, HD95
# Quelle: "Primäre Bewertungsmetriken für den Vergleich klassischer und
# Deep-Learning-basierter Registrierungen", Abschnitt 1, Tabelle
# "Bei Inferenz und abschließender Bewertung" ([4][5]).
# ---------------------------------------------------------------------------


def target_registration_error(
    fixed_landmarks: np.ndarray | None,
    warped_moving_landmarks: np.ndarray | None,
) -> float:
    """Target Registration Error (TRE).

    Mittlerer euklidischer Abstand zwischen anatomisch korrespondierenden
    Landmarken im Fixed Image und den entsprechend der Transformation phi
    deformierten Landmarken des Moving Image (siehe PDF, Ebene 1,
    "Bei Inferenz und abschließender Bewertung", Quellen [4][5]).

    TODO / Dummy-Implementierung: Es liegen aktuell keine Landmarken vor.
    Solange ``fixed_landmarks`` bzw. ``warped_moving_landmarks`` ``None``
    sind, gibt diese Funktion NaN zurück. Sobald Landmarken (z. B. manuell
    annotierte anatomische Punkte oder automatisch detektierte Keypoints)
    verfügbar sind, hier als (N, 2)-Arrays [y, x] bzw. [x, y] (konsistent zur
    übrigen Codebasis) übergeben.

    Args:
        fixed_landmarks: (N, 2)-Array der Landmarken im Zielbild, oder None.
        warped_moving_landmarks: (N, 2)-Array der deformierten Landmarken
            des Moving Image (also phi angewendet auf die Moving-Landmarken),
            oder None.

    Returns:
        Mittlerer euklidischer Abstand in Pixeln (bzw. mm bei entsprechend
        skalierten Koordinaten). NaN, falls keine Landmarken vorliegen.
    """
    if fixed_landmarks is None or warped_moving_landmarks is None:
        return float("nan")

    fixed_landmarks = np.asarray(fixed_landmarks, dtype=float)
    warped_moving_landmarks = np.asarray(warped_moving_landmarks, dtype=float)

    if fixed_landmarks.shape != warped_moving_landmarks.shape:
        raise ValueError(
            "Landmark shape mismatch: "
            f"{fixed_landmarks.shape} vs {warped_moving_landmarks.shape}"
        )

    if fixed_landmarks.size == 0:
        return float("nan")

    distances = np.linalg.norm(fixed_landmarks - warped_moving_landmarks, axis=-1)
    return float(np.mean(distances))


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """Extrahiert die Randpixel (Oberfläche) einer binären 2D-Maske."""
    mask = mask.astype(bool)
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def hausdorff_distance_95(
    mask1: np.ndarray,
    mask2: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Berechnet den HD95 (95. Perzentil des bidirektionalen Oberflächenabstands)
    zwischen zwei binären 2D-Masken.

    HD95 ist robuster als der maximale Hausdorff-Abstand, da einzelne
    Ausreißer-Randpunkte den Wert nicht dominieren (siehe PDF, Ebene 1,
    "Bei Inferenz und abschließender Bewertung", Quellen [4][5]).

    Args:
        mask1: Erste binäre 2D-Maske (z. B. registrierte Lungenmaske).
        mask2: Zweite binäre 2D-Maske (z. B. Ziel-Lungenmaske), gleiche Form.
        spacing: Pixel-/Voxel-Abstand (y, x) zur Umrechnung in physikalische
            Einheiten (z. B. mm). Standard: (1.0, 1.0) = Pixel-Einheiten.

    Returns:
        HD95 als float. NaN, falls eine der Masken leer ist.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"mask shape mismatch: {mask1.shape} vs {mask2.shape}")

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    if not np.any(m1) or not np.any(m2):
        return float("nan")

    surf1 = _surface_mask(m1)
    surf2 = _surface_mask(m2)

    # Falls eine Maske vollflächig ist (keine erodierbare Grenze), die
    # gesamte Maske als Rand verwenden.
    if not np.any(surf1):
        surf1 = m1
    if not np.any(surf2):
        surf2 = m2

    dt_from_2 = ndimage.distance_transform_edt(~surf2, sampling=spacing)
    dt_from_1 = ndimage.distance_transform_edt(~surf1, sampling=spacing)

    d_1_to_2 = dt_from_2[surf1]
    d_2_to_1 = dt_from_1[surf2]

    all_distances = np.concatenate([d_1_to_2, d_2_to_1])
    return float(np.percentile(all_distances, 95))


# ---------------------------------------------------------------------------
# Ebene 2 - Plausibilität des Deformationsfeldes
# Empfohlene Mindestanforderung (Inferenz/abschließende Bewertung):
# %negJ / nicht-positive Jacobians, SDlogJ, mittlerer Verschiebungsbetrag;
# bei Referenz-DVF zusätzlich DVF-MAE/RMSE und Vektor-Korrelation.
# Quelle: PDF, Ebene 2, "Bei Inferenz und abschließender Bewertung",
# Quellen [3][8][1][9][10].
# ---------------------------------------------------------------------------


def _displacement_components(disp_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extrahiert die x-/y-Komponenten (ux, uy) aus einem 2D-Verschiebungsfeld
    der Form (2, H, W) oder (H, W, 2)."""
    if disp_field.ndim != 3:
        raise ValueError(
            f"disp_field must have shape (2, H, W) or (H, W, 2), got {disp_field.shape}"
        )

    if disp_field.shape[0] == 2:
        return disp_field[0], disp_field[1]
    elif disp_field.shape[-1] == 2:
        return disp_field[..., 0], disp_field[..., 1]
    else:
        raise ValueError(
            f"disp_field must have shape (2, H, W) or (H, W, 2), got {disp_field.shape}"
        )


def _jacobian_determinant_map(disp_field: np.ndarray) -> np.ndarray:
    """Berechnet die Jacobian-Determinanten-Karte det(J) der Transformation
    phi(x, y) = (x + ux(x, y), y + uy(x, y)) aus einem Verschiebungsfeld.

    Wird sowohl von ``negative_jacobian_rate`` als auch von
    ``sd_log_jacobian`` verwendet.
    """
    ux, uy = _displacement_components(disp_field)

    # Spatial derivatives:
    # axis=0 -> y direction
    # axis=1 -> x direction
    dux_dx = np.gradient(ux, axis=1)
    dux_dy = np.gradient(ux, axis=0)

    duy_dx = np.gradient(uy, axis=1)
    duy_dy = np.gradient(uy, axis=0)

    jac_det = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
    return jac_det


def negative_jacobian_rate(
    disp_field: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Berechnet den Anteil negativer Jacobian-Determinanten.
    Original nach Dongyang Kuang et al. (2019)
    https://doi.org/10.48550/arXiv.1811.09243 (Experiments 3.2)

    Die Metrik quantifiziert lokale Foldings bzw. nicht-invertierbare
    Bereiche eines Deformationsfeldes. Aus dem Verschiebungsfeld

        phi(x, y) = (x + ux(x, y), y + uy(x, y))

    wird die Jacobi-Matrix der Transformation berechnet:

        J =
        [[1 + dux/dx,  dux/dy],
         [duy/dx,      1 + duy/dy]]

    und anschließend deren Determinante:

        det(J) = (1 + dux/dx)(1 + duy/dy)
                 - (dux/dy)(duy/dx)

    Eine negative Determinante weist auf eine lokale Umkehrung der
    Orientierung und damit auf ein Folding der Transformation hin.

    Die Implementierung folgt dabei der in der FAIM-Arbeit verwendeten
    Definition von Folding-Locations als Orte mit negativer
    Jacobian-Determinante.

    Wenn eine Maske angegeben wird, wird die Rate ausschließlich innerhalb
    der maskierten Region berechnet. Dies stellt eine ROI-beschränkte
    Erweiterung der globalen Folding-Rate dar.

    Args:
        disp_field: 2D-Verschiebungsfeld mit Form ``(2, H, W)`` oder
            ``(H, W, 2)``. Die beiden Komponenten entsprechen den
            Verschiebungen in x- und y-Richtung.
        mask: Optionale binäre 2D-Maske mit Form ``(H, W)``. Wenn angegeben,
            wird nur die maskierte Region ausgewertet.

    Returns:
        Anteil der Pixel mit negativer Jacobian-Determinante.
        Ohne Maske entspricht dies dem Anteil aller Pixel, mit Maske dem
        Anteil innerhalb der ROI. Der Wert liegt zwischen ``0.0`` und ``1.0``.

    Raises:
        ValueError: Wenn das Verschiebungsfeld oder die Maske eine ungültige
            Form besitzt oder die Maske nicht zum räumlichen Feld passt.
    """
    ux, _ = _displacement_components(disp_field)

    if mask is not None and mask.shape != ux.shape:
        raise ValueError(f"mask/disp_field shape mismatch: {mask.shape} vs {ux.shape}")

    jac_det = _jacobian_determinant_map(disp_field)

    # Following the FAIM definition: folding = negative Jacobian.
    folding_map = jac_det < 0.0

    if mask is not None:
        mask = mask.astype(bool)

        if not np.any(mask):
            return 0.0

        return float(np.mean(folding_map[mask]))

    return float(np.mean(folding_map))


def sd_log_jacobian(
    disp_field: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """SDlogJ - Standardabweichung von log(det(J)) für gültige (positive)
    Jacobian-Determinanten.

    Dient als Maß für die Homogenität/Glattheit des Deformationsfeldes:
    je niedriger SDlogJ, desto gleichmäßiger die lokale
    Expansion/Kompression. Negative Jacobians (Foldings) werden dabei
    ausgeschlossen, da log() dort nicht definiert ist (siehe PDF, Ebene 2,
    "Bei Inferenz und abschließender Bewertung", Quelle [8]).

    Args:
        disp_field: 2D-Verschiebungsfeld mit Form ``(2, H, W)`` oder
            ``(H, W, 2)``.
        mask: Optionale binäre 2D-Maske. Wenn angegeben, wird nur die
            maskierte Region ausgewertet.

    Returns:
        Standardabweichung von log(det(J)) innerhalb der validen (positiven)
        Jacobian-Region. NaN, falls keine positiven Jacobians vorliegen.
    """
    ux, _ = _displacement_components(disp_field)

    if mask is not None and mask.shape != ux.shape:
        raise ValueError(f"mask/disp_field shape mismatch: {mask.shape} vs {ux.shape}")

    jac_det = _jacobian_determinant_map(disp_field)
    valid = jac_det > 0.0

    if mask is not None:
        valid = valid & mask.astype(bool)

    if not np.any(valid):
        return float("nan")

    log_jac = np.log(jac_det[valid])
    return float(np.std(log_jac))


def mean_displacement_magnitude(
    disp_field: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mittlerer Verschiebungsbetrag |u| = sqrt(ux^2 + uy^2) des
    Deformationsfeldes.

    Dient zur Plausibilitätsprüfung, ob ein Verfahren unrealistisch große
    Bewegungen erzeugt (siehe PDF, Ebene 2, "Bei Inferenz und
    abschließender Bewertung", Quelle [1]). Die PDF empfiehlt zusätzlich
    Median, 95. Perzentil und Maximum zu berichten; diese lassen sich bei
    Bedarf direkt aus der zurückgegebenen Verschiebungsbetrag-Karte ableiten.

    Args:
        disp_field: 2D-Verschiebungsfeld mit Form ``(2, H, W)`` oder
            ``(H, W, 2)``.
        mask: Optionale binäre 2D-Maske. Wenn angegeben, wird nur die
            maskierte Region gemittelt.

    Returns:
        Mittlerer Verschiebungsbetrag als float.
    """
    ux, uy = _displacement_components(disp_field)
    magnitude = np.sqrt(ux**2 + uy**2)

    if mask is not None:
        mask = mask.astype(bool)
        if np.any(mask):
            return float(np.mean(magnitude[mask]))

    return float(np.mean(magnitude))


def dvf_error_against_reference(
    disp_field: np.ndarray,
    reference_disp_field: np.ndarray | None,
    mask: np.ndarray | None = None,
) -> dict:
    """DVF-Fehler (MAE/RMSE) und Vektor-Korrelation gegen ein bekanntes
    Referenz-Verschiebungsfeld (z. B. aus synthetischen Daten mit bekannter
    Ground-Truth-Deformation, vgl. ASYLUM).

    Siehe PDF, Ebene 2, "Bei Inferenz und abschließender Bewertung",
    Quelle [1].

    TODO / Dummy-Implementierung: Es liegt aktuell kein Referenz-DVF vor.
    Solange ``reference_disp_field`` ``None`` ist, gibt diese Funktion NaN
    für alle drei Kennzahlen zurück. Sobald eine Ground-Truth-Deformation
    verfügbar ist (z. B. aus einem synthetischen Validierungsdatensatz),
    hier in derselben Form wie ``disp_field`` (``(2, H, W)`` oder
    ``(H, W, 2)``) übergeben.

    Args:
        disp_field: Geschätztes 2D-Verschiebungsfeld.
        reference_disp_field: Bekanntes Referenz-Verschiebungsfeld in
            gleicher Form, oder None.
        mask: Optionale binäre 2D-Maske zur ROI-Beschränkung.

    Returns:
        dict mit den Schlüsseln ``DVF_MAE``, ``DVF_RMSE`` und
        ``DVF_Vector_Correlation``. Alle Werte sind NaN, falls kein
        Referenzfeld übergeben wurde.
    """
    if reference_disp_field is None:
        return {
            "DVF_MAE": float("nan"),
            "DVF_RMSE": float("nan"),
            "DVF_Vector_Correlation": float("nan"),
        }

    ux, uy = _displacement_components(disp_field)
    ref_ux, ref_uy = _displacement_components(reference_disp_field)

    if ux.shape != ref_ux.shape:
        raise ValueError(
            f"disp_field/reference_disp_field shape mismatch: {ux.shape} vs {ref_ux.shape}"
        )

    if mask is not None:
        mask = mask.astype(bool)
        if not np.any(mask):
            return {
                "DVF_MAE": float("nan"),
                "DVF_RMSE": float("nan"),
                "DVF_Vector_Correlation": float("nan"),
            }
        ux, uy = ux[mask], uy[mask]
        ref_ux, ref_uy = ref_ux[mask], ref_uy[mask]

    diff_magnitude = np.sqrt((ux - ref_ux) ** 2 + (uy - ref_uy) ** 2)
    mae = float(np.mean(diff_magnitude))
    rmse = float(np.sqrt(np.mean(diff_magnitude**2)))

    # Vektor-Korrelation: Pearson-Korrelation über die zusammengefassten
    # (ux, uy)-Komponenten von geschätztem und Referenzfeld.
    est_flat = np.concatenate([ux.ravel(), uy.ravel()])
    ref_flat = np.concatenate([ref_ux.ravel(), ref_uy.ravel()])

    if np.std(est_flat) == 0 or np.std(ref_flat) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(est_flat, ref_flat)[0, 1])

    return {
        "DVF_MAE": mae,
        "DVF_RMSE": rmse,
        "DVF_Vector_Correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Ebene 3 - Funktionelle Qualität der Karten
# Empfohlene Mindestanforderung ohne Referenzkarten: VDP (analog QDP für
# Perfusion), jeweils basierend auf einem Schwellenwert innerhalb der
# Lungenmaske. Quelle: PDF, Ebene 3.1/3.2, "Bei Inferenz und
# abschließender Bewertung", Quellen [1][9].
# ---------------------------------------------------------------------------


def defect_percentage(
    functional_map: np.ndarray,
    mask: np.ndarray,
    threshold: float | None = None,
) -> float:
    """Defect Percentage (VDP für Ventilation, analog QDP für Perfusion).

    Anteil der Voxel/Pixel innerhalb der Lungenmaske, deren Funktionswert
    unterhalb eines Schwellenwerts liegt (siehe PDF, Ebene 3.1 "Ventilation
    Defect Percentage, VDP" bzw. Ebene 3.2 "Perfusion Defect Percentage,
    QDP").

    TODO: In der Literatur (ASYLUM, siehe PDF Quelle [9]) wird der
    Schwellenwert i. d. R. relativ zu einem klinisch/physiologisch
    begründeten Cutoff definiert (z. B. ein fester Prozentsatz des
    Referenzmittelwerts gesunder Lunge). Da hier keine Referenzkarten
    vorliegen, wird ersatzweise der **Mittelwert der Funktionskarte
    innerhalb der Lungenmaske des jeweiligen Patienten** als Schwellenwert
    verwendet (Pixel < Mittelwert = "Defekt"). Dies ist ein pragmatischer
    Platzhalter und sollte überprüft/angepasst werden, sobald eine
    klinisch etablierte Referenzschwelle feststeht.

    Args:
        functional_map: 2D-Funktionskarte (z. B. Ventilations- oder
            Perfusionskarte).
        mask: Binäre 2D-Lungenmaske.
        threshold: Optionaler expliziter Schwellenwert. Wenn ``None``
            (Standard), wird der Mittelwert der Funktionskarte innerhalb
            der Maske verwendet (siehe TODO oben).

    Returns:
        Anteil der Pixel unterhalb des Schwellenwerts in Prozent (0-100).
        NaN, falls die Maske leer ist.
    """
    mask = mask.astype(bool)
    values = functional_map[mask]

    if values.size == 0:
        return float("nan")

    if threshold is None:
        # TODO: Platzhalter-Schwellenwert = Mittelwert der Lunge (siehe Docstring).
        threshold = float(np.mean(values))

    defect_fraction = np.mean(values < threshold)
    return float(defect_fraction * 100.0)


# ---------------------------------------------------------------------------
# Ebene 4 - Robustheit, Reproduzierbarkeit und Effizienz
# Empfohlene Mindestanforderung: Test-Retest-ICC, CoV, Fehlerrate, Laufzeit.
# Quelle: PDF, Ebene 4, "Bei Inferenz und abschließender Bewertung",
# Quellen [9].
# ---------------------------------------------------------------------------


def coefficient_of_variation_repeated(values: np.ndarray) -> float:
    """Coefficient of Variation (CoV) über Wiederholungsmessungen eines
    globalen Funktionsparameters (z. B. VDP, mittlere Ventilation) für
    denselben Patienten.

    CoV = SD / |Mittelwert| * 100 (%)

    Siehe PDF, Ebene 4, "Bei Inferenz und abschließender Bewertung",
    "Coefficient of Variation, CoV", Quelle [9].

    Args:
        values: 1D-Array der Wiederholungsmessungen eines einzelnen
            Parameters (z. B. VDP-Werte über mehrere Wiederholungsscans
            desselben Patienten).

    Returns:
        CoV in Prozent. NaN, falls weniger als 2 gültige Werte vorliegen
        oder der Mittelwert 0 ist.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size < 2:
        return float("nan")

    mean_val = np.mean(values)
    if mean_val == 0:
        return float("nan")

    return float(np.std(values, ddof=1) / abs(mean_val) * 100.0)


def intraclass_correlation(data: np.ndarray) -> float:
    """Intraclass Correlation Coefficient, ICC(3,1) - Two-Way Mixed-Effects,
    Consistency, Single Measurement (Shrout & Fleiss, 1979).

    Misst die Test-Retest-Reproduzierbarkeit globaler Kartenparameter
    (siehe PDF, Ebene 4, "Bei Inferenz und abschließender Bewertung",
    "Test-Retest-ICC", Quelle [9]).

    Wichtiger Hinweis: Für eine belastbare ICC-Schätzung werden mehrere
    Subjekte (Patienten) mit jeweils mehreren Wiederholungsmessungen
    benötigt, da die Zwischen-Subjekt-Varianz Teil der Formel ist. Liegen
    (wie aktuell im Projekt) Wiederholungsmessungen nur für einen
    einzelnen Patienten vor, ist die Zwischen-Subjekt-Varianz nicht
    schätzbar; die Funktion gibt in diesem Fall NaN zurück (n < 2), und es
    sollte primär der CoV (siehe ``coefficient_of_variation_repeated``)
    berichtet werden. Sobald Wiederholungsmessungen für mehrere Patienten
    vorliegen, kann dieselbe Funktion direkt verwendet werden.

    Args:
        data: (n_subjects, n_repeats)-Matrix der Messwerte.

    Returns:
        ICC(3,1) als float im Bereich (typischerweise) [0, 1]. NaN, falls
        weniger als 2 Subjekte oder weniger als 2 Wiederholungen vorliegen,
        oder falls die Varianz-Zerlegung degeneriert ist.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"data must be a 2D (n_subjects, n_repeats) array, got shape {data.shape}"
        )

    n, k = data.shape

    if n < 2 or k < 2:
        return float("nan")

    mean_subjects = np.mean(data, axis=1)
    mean_raters = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_between_subjects = k * np.sum((mean_subjects - grand_mean) ** 2)
    ss_between_raters = n * np.sum((mean_raters - grand_mean) ** 2)
    ss_error = ss_total - ss_between_subjects - ss_between_raters

    ms_between_subjects = ss_between_subjects / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_between_subjects + (k - 1) * ms_error
    if denominator == 0:
        return float("nan")

    icc = (ms_between_subjects - ms_error) / denominator
    return float(icc)


def evaluate_registration(
    registered_image: np.ndarray,
    reference_image: np.ndarray,
    mask: np.ndarray,
    image_series: np.ndarray,
    disp_field: np.ndarray,
    registered_mask: np.ndarray,
) -> dict:
    """Gesamtevaluierung aller Bild-, Registrierungs- und PREFUL-Metriken.

    Returns:
        dict: Wörterbuch mit allen berechneten Metrik-Scores.
    """
    results = {}

    # 1. Bildeigene Schärfe & Varianz
    results["ISM"] = image_sharpness_metric(registered_image, mask)
    results["CPV"] = coefficient_of_variation(registered_image, mask)

    # 2. Bildähnlichkeitsmetriken gegen Referenz
    results["LNCC"] = local_normalized_cross_correlation(
        registered_image, reference_image, mask
    )
    results["SSIM"] = structural_similarity_index(
        registered_image, reference_image, mask
    )

    # 3. Maskenüberlappung (falls registrierte Maske übergeben)
    if registered_mask is not None:
        results["Dice"] = dice_coefficient(registered_mask, mask)
        results["HD95"] = hausdorff_distance_95(registered_mask, mask)

    # 4. Physiologisches Belüftungssignal (falls Zeitserie übergeben)
    if image_series is not None:
        results["FV"] = fractional_ventilation(image_series, mask)

    # 5. Topologieprüfungen des Verformungsfeldes (falls Verschiebungsfeld übergeben)
    if disp_field is not None:
        results["Negative_Jacobian_Rate"] = negative_jacobian_rate(disp_field, mask)
        results["SDlogJ"] = sd_log_jacobian(disp_field, mask)
        results["Mean_Displacement"] = mean_displacement_magnitude(disp_field, mask)

    return results
