import itk
import numpy as np
import pydicom

def read_images_from_folder(folder_path):
    series_file_names = itk.GDCMSeriesFileNames.New()
    series_file_names.SetDirectory(folder_path)
    dicom_names = series_file_names.GetInputFileNames()
    print(f'{len(dicom_names)} files were read.')
    return itk.imread(dicom_names)

def center_crop_last_dim(arr, target_size=256):
    ### helperfunction for non-square images ###
    current_size = arr.shape[-1]
    if current_size > target_size:
        start = (current_size - target_size) // 2
        end = start + target_size
        return arr[..., start:end]
    else:
        return arr  # No cropping needed
    

def save_image_as_dicom(image, output_path):
    itk.imwrite(image, output_path)


def get_dicom_acquisition_times(main_directory):
    """
    Reads DICOM series from the specified directory and extracts acquisition times.

    Args:
    main_directory (str): Directory containing the DICOM files.

    Returns:
    np.ndarray: Array of acquisition times adjusted relative to the first acquisition time.
    """
    series_file_names = itk.GDCMSeriesFileNames.New()
    series_file_names.SetDirectory(main_directory)
    dicom_names = series_file_names.GetInputFileNames()
    
    acquisition_times = []
    for file in dicom_names:
        dataset = pydicom.dcmread(file, stop_before_pixels=True)
        info = dataset.get("AcquisitionDateTime")
        if info is None:
            raise ValueError(f"Missing DICOM AcquisitionDateTime (0008,002A): {file}")

        info = str(info).strip()
        if len(info) < 14:
            raise ValueError(f"Invalid DICOM AcquisitionDateTime (0008,002A): {info}")

        acquisition_times.append(
            int(info[8:10]) * 3600
            + int(info[10:12]) * 60
            + float(info[12:])
        )

    if not acquisition_times or any(value is None for value in acquisition_times):
        raise ValueError("No usable DICOM acquisition timing metadata was found.")
    
    adjusted_times = np.asarray(acquisition_times, dtype=float)
    adjusted_times -= adjusted_times[0]
    if len(adjusted_times) > 1 and not np.any(np.diff(adjusted_times) > 0):
        raise ValueError("DICOM acquisition timestamps do not vary between frames.")
    return adjusted_times


#### could be used in the future for groupwise registration. Currently not in use.
def read_grouped_intensities(file_path):
    """
    Reads the grouped intensities from a text file and returns them as a nested list.
    
    Args:
        file_path (str): The path to the text file containing the grouped intensities.
        
    Returns:
        grouped_intensities (list): A nested list containing the grouped intensities.
    """
    grouped_intensities = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_group = []
    for line in lines:
        if line.startswith('Group'):
            if current_group:
                grouped_intensities.append(current_group)
                current_group = []
        elif line.strip().startswith('Phase'):
            intensities_str = line.split(':')[1].strip().strip('[]')
            intensities = list(map(float, intensities_str.split(',')))
            current_group.append(intensities)
    
    # Append the last group if any
    if current_group:
        grouped_intensities.append(current_group)

    return grouped_intensities

def array_to_itk(image_array):
    print(image_array.shape)
    image_array = np.squeeze(np.transpose(image_array, (3, 4, 0, 1, 2)))
    print(image_array.shape)
    return itk.image_from_array(image_array)


def get_ismrmrd_acquisition_times(head):
    time_array = np.empty((len(head)))
    for img in range(len(head)):
        info = head[img].acquisition_time_stamp
        time_array[img] = 1e-3* info *2.5 # ismrmrd acquisition tim in units of 1/2.5 ms
    return time_array - time_array[0]