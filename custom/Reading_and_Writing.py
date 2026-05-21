import SimpleITK as sitk
import os
import numpy as np

def read_images_from_folder(folder_path):
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    print(f'{len(dicom_names)} files were read.')
    image = reader.Execute()
    return image

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
    # Cast to 16-bit
    cast_image = sitk.Cast(image, sitk.sitkUInt16)
    
    # Copy the spatial axes info (origin, spacing, direction) from the original
    cast_image.CopyInformation(image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cast_image)


def get_dicom_acquisition_times(main_directory):
    """
    Reads DICOM series from the specified directory and extracts acquisition times.

    Args:
    main_directory (str): Directory containing the DICOM files.

    Returns:
    np.ndarray: Array of acquisition times adjusted relative to the first acquisition time.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(main_directory)
    
    acquisition_times = []
    for file in dicom_names:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(file)
        file_reader.ReadImageInformation()
        info = file_reader.GetMetaData('0008|0013')
        hours, minutes, seconds = int(info[:2]), int(info[2:4]), float(info[4:].strip())
        converted_time = (hours * 60 + minutes) * 60 + seconds
        acquisition_times.append(converted_time)
    
    adjusted_times = np.array(acquisition_times) - acquisition_times[0]
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

def array_to_sitk(image_array):
    print(image_array.shape)
    image_array = np.squeeze(np.transpose(image_array, (3, 4, 0, 1, 2)))
    print(image_array.shape)
    image_sitk = sitk.GetImageFromArray(image_array)
    return image_sitk


def get_ismrmrd_acquisition_times(head):
    time_array = np.empty((len(head)))
    for img in range(len(head)):
        info = head[img].acquisition_time_stamp
        time_array[img] = 1e-3* info *2.5 # ismrmrd acquisition tim in units of 1/2.5 ms
    return time_array - time_array[0]