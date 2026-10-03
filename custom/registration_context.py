from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class RegistrationResult:
    """Results produced by the registration stage.

    All registration methods should return the same semantic structure.

    Attributes:
        registered_series:
            Registered 3D SimpleITK image series.
        reference_image:
            2D fixed/reference image used for registration.
        reference_index:
            Frame index of the fixed/reference image in the input series.
        fixed_lung_mask:
            Lung mask in reference-image geometry. This is the common
            registration-evaluation reference mask.
        registered_lung_mask:
            Lung mask after registration, in reference-image geometry.
        disp_fields:
            Per-frame displacement fields. Expected shape is
            (T, Y, X, 2) when available. ``None`` if the registration
            method does not provide displacement fields.
    """

    registered_series: sitk.Image

    reference_image: np.ndarray
    reference_index: int

    fixed_lung_mask: np.ndarray | None = None
    registered_lung_mask: np.ndarray | None = None
    disp_fields: np.ndarray | None = None
    transform_parameter_maps: object | None = None
