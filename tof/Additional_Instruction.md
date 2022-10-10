Requirements:
>Python (3.7-3.9)



Additional instructions in building ADTF3175D-NXZ SDK



*Turn on the DWITH_NETWORK
> Open/edit the setup_project.bat (\Tof\scripts\windows\)
> Input -DWITH_NETWORK=on-
cmake -G %generator% -DWITH_PYTHON=on -DWITH_NETWORK=on-



*Run the setup_project.bat (If success, skip the optional error-section)



*Optional Errors:
-Cannot convert ssize_t
> Open/Edit the aditopython (\Tof\bindings\pythong\aditofpython.c)
> Input the following command (#include -section)



#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif



*Checking
>Run the ADIToFGUI 3.0.0 (\Analog Devices\TOF_Evaluation_ADTF3175D-Rel3.0.0\bin\tools)
*Make ADIToFGUI 3.2.0
> Copy ADIToFGUI.exe (\ToF\scripts\windows\build\examples\tof-viewer\release)



*Building the SDK
> Create your folder (dev)
> Copy all files from (\ToF\scripts\windows\build\bindings\python\Release) to the created folder (dev)
>Copy all files from (\ToF\scripts\windows\build\examples\first-frame-network\release) -EXCEPT (ADIToFGUI.exe, logs)



*Run Examples
> Copy/Paste any .py from (C:\dev\ToF\bindings\python\examples) to the created folder(dev)
> Create environment

*Updating code for .py files
>Add path for "tof-viewer_config.json"
> Add "status = cameras[0].setControl("initialization_config", tof-viewer_config.json)"