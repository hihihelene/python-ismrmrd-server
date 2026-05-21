import ismrmrd
import numpy as np

def matrix_to_ismrmrd(data, title_ismrmrd): 

    # Create an ISMRMRD dataset
    print(type(title_ismrmrd))
    dataset = ismrmrd.Dataset(title_ismrmrd, data, create_if_needed=True)

    # Create an ISMRMRD Acquisition object
    # acquisition = ismrmrd.Acquisition()

    # Set acquisition header properties (example: 128 samples, 1 channel)
    # acquisition.resize(num_samples=len(data), active_channels=1)

    # Assign the NumPy array data to the acquisition data
    # acquisition.data[:] = data.flatten()

    # Append the acquisition to the dataset
    # dataset.append_acquisition(acquisition)

    # Close the dataset
    dataset.close()

    print("ISMRMRD dataset created successfully!")

    return
