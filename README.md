# ANN_gpu
Neural Network from scratch Using CUDA


Megh Shukla : 173310008 
Ushasi Chaudhari : 174310003

The code is in 2 files:
--> ANN_gpu.py
--> ANNGui.py

Other files/folders:
Stocks_Input.csv --> NSE stocks data
ANNGui.spec --> spec file for pyinstaller to make the executable
Saved Model : To save the ANN model in
Images : Sample images of our project
ML_ANN_Project_PPT.pdf : PPT of our demonstration

The third file (ErrorsInCuda.py)is demonstrating difficulties in using CUDA Python, which can be
ignored as it was meant to be presented in the Project demonstration only

Execute ANNGui.py for project demonstration, it makes calls to ANN_gpu.py internally.

IMPORTANT: 
Additionally we have also provided an executable file in "dist folder"
This executable will work on Windows 10, x64 architecture CPUs
However, this exe might fail if Numba cannot detect the installation of Cudatoolkit 
on the host system.
I have demonstrated that the executable works on my system to Deepak sir in the project demonstration,
if needed I can do so again. 
The fact that Numba cannot detect cudatoolkit on a new system is out of my scope, it is 
something that Numba CUDA needs to resolve.
On my part, we have everything in place for the correct execution of the file, as shown in the
project demonstration.

If I had made errors in making the .exe, the file would not have executed on my system 
throwing exceptions regarding missing imports. The exe running on my system is proof that 
the executable is working properly given Numba can detect cudatoolkit.

Please do bear this in mind, as this issue is something that can't be fixed by us.
