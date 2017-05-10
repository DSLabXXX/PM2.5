# Create .theanorc file and write down the following setting.
"""
[global]
floatX = float32
device = gpu

# By default the compiled files were being written to my local network drive.
# Since I have limited space on this drive (on a school's network),
# we can change the path to compile the files on the local machine.
# You will have to create the directories and modify according to where you
# want to install the files.
# Uncomment if you want to change the default path to your own.
# base_compiledir = /local-scratch/jer/theano/

[nvcc]
fastmath = True

[gcc]
cxxflags = -ID:\MinGW\include

[cuda]
# Set to where the cuda drivers are installed.
# You might have to change this depending where your cuda driver/what version is installed.
root=/usr/local/cuda-7.5/
"""