# stainlib
![Currently implemented methods in stainlib](https://drive.google.com/uc?id=1By4Nw3X0sgwxamF0qN3TqiL-B1q2qZqQ)
The objective with this repository is to provide an easy to use python 3 library that includes 
the most commonly used methods for color augmentation and normalisation of histopathology images, having as input local image regions stained with H&amp;E.

## Pre-requisites and Installation
The library was developed and tested in a python 3.8 conda environment.  The following packages are required:
- scikit-image
- scipy
- pillow
- opencv-python
- spams

You can find a complete list of the packages installed when developed the library under utils/enviroment.yml
 
For installing the library you can do it with pip:
pip install -e stainlib/ 
 
## Examples
You can find examples for using stainlib in the jupyter notebooks stainlib_augmentation.ipynb and stainlib_normalization.ipynb
# Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">


