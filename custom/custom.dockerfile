# ----- 1. First stage to create a devcontainer -----
# Start from standard python-ismrmrd-server devcontainer
FROM kspacekelvin/fire-python-devcon AS fire-python-custom-devcon

# Clone the latest version of python-ismrmrd-server
RUN cd /opt/code && \
    git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git


# Install the remaining Python package dependencies
RUN pip install --no-cache-dir \
    SimpleITK\
    itk-elastix\
    scipy \
    matplotlib\
    opencv-python-headless\
    torch torchvision \
    nnunetv2\
    voxelmorph

# ----- 2. Second stage to create a runtime container for deployment -----
FROM fire-python-custom-devcon AS fire-python-custom-runtime

# Copy in modules and other files as needed
# COPY filter.py    /opt/code/python-ismrmrd-server
# COPY filter.json  /opt/code/python-ismrmrd-server
COPY Dynamic_Mode_Decomposition.py    /opt/code/python-ismrmrd-server/custom
COPY Fourier_Decomposition.py  /opt/code/python-ismrmrd-server/custom
COPY MAIN_VQMap.json /opt/code/python-ismrmrd-server/custom
COPY MAIN_VQMap.py /opt/code/python-ismrmrd-server/custom
COPY Plotting.py /opt/code/python-ismrmrd-server/custom
COPY Reading_and_Writing.py /opt/code/python-ismrmrd-server/custom
COPY Registration.py /opt/code/python-ismrmrd-server/custom
COPY HyperMorph_Registration.py /opt/code/python-ismrmrd-server/custom
COPY VoxelMorph_Registration.py /opt/code/python-ismrmrd-server/custom
COPY registration_context.py /opt/code/python-ismrmrd-server/custom
COPY Registration_Ablation.py /opt/code/python-ismrmrd-server/custom
COPY Segmentation.py /opt/code/python-ismrmrd-server/custom
COPY VQMapping.py /opt/code/python-ismrmrd-server/custom
COPY nnUnet_Segmentation.py /opt/code/python-ismrmrd-server/custom
COPY registration_parameter_file.txt /opt/code/python-ismrmrd-server/custom
COPY GORE_Registration_Parameter_File.txt /opt/code/python-ismrmrd-server/custom
COPY models /opt/code/python-ismrmrd-server/models
COPY Scripts /opt/code/python-ismrmrd-server/Scripts


# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

# Use the -d argument at the end to indicate the default (intended) module to be run by this Docker image
CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "-d=MAIN_VQMap"]

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
# ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "-d=MAIN_VQMap"]