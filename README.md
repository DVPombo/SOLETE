# SOLETE
Author: **Daniel Vázquez Pombo** - Contact: daniel.vazquez.pombo@gmail.com<br/>
LinkedIn: https://www.linkedin.com/in/dvp/<br/>
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo   
ORCID: https://orcid.org/0000-0001-5664-9421

This repository used to be complementary material to its twin "Data in Brief" article [1], and a series of papers covering Solar PV power forecasting [2, 3, 4]. The objective is to increase the transparency of my work, which is one of the main limitations of Machine Learning in general.
However, as it sometimes happens, the project has grown life by itself and has now become a platform to experiment on time-series forecasting based on Machine Learning.
I included a number of functions that can be used by beginners to kickstart their projects with solar power, machine learning, forecasting, or simply python.

Long Live Open Science!

The papers were developed under the PhD thesis Operation and Planning of Isolated Hybrid Power Systems at the Technical University of Denmark (DTU).
Version v1.0 was released during the PhD thus, Copyright 2021 Technical University of Denmark.
Version v2.0 was released months after finalising my employment at DTU, therefore, Copyright belongs to me (yeah baby!).

# Dependencies
1. Python 3.9.12 
2. Pandas 1.5.0 
3. Numpy 1.23.1
4. Matplotlib 3.6.0
5. Scikit-Learn 1.1.2 
6. Keras 2.10.0
7. TensorFlow 2.10.0
8. CoolProp 6.4.3   
9. The SOLETE dataset [1] -> https://doi.org/10.11583/DTU.17040767 

# How to use
1. Store the SOLETE dataset in the same folder as the scripts from this repository 
2. Open the **RunMe.py** file. This allows you to load SOLETE and sneak a peek at its contents.
3. Open the  **MLForecasting.py** file. This allows you to configure Random Forest (RF), Support Vector Machine (SVM), and three kinds of Artificial Neuronal Networks: Convolutional Neuronal Network (CNN), Long-Short Term Memory (LSTM), and a Hybrid (CNN-LSTM).
   - The file itself contains notes explaining how to use it.
   - The main objective is to introduce the SOLETE dataset and help people learning basics of time series forecasting based on Machine Learning
   - You can basically replicate most of the methodology from [2, 3, 4] and build on top.
   - I included some error messages to debug what I expect are the most common errors when running stuff.
   - Let me know if you like it or what needs to be fixed.
4. Have Fun!

You can of course use your own dataset, you will have to adapt things here and there, but you will be able to reuse most of the code.

Note that the latest version of the SOLETE dataset includes a 1sec resolution version. The file is quite large, which might meant that you PC is not able to open it. Please consider only reading part of it if you really want to play with that resolution. Alternatively, drop it in an HPC and enjoy yourself. :D 

### Notes for _MATLAB_ users ###
I have been reached out by several people complaining that hdf5 can't be imported in MATLAB. That is not true, they weren't doing properly. Nevertheless, worry not dear user. Your peers have asked and I answer:
1. Open the file **RunMe_matlab.m** in MATLAB and hit F5. That will import SOLETE as a _table_.
2. Alternatively, you can still run the Python scripts from MATLAB, which I find a bit weird... but hey! You do you baby!

*I coded this using 2021b, so anything newer should work, but I haven't actually checked with older versions.

# How to cite this:
Technically, you should cite the repository itself, however I don't get those citations captured where it matters, so please cite [1] like this:

@article{pombo2022solete,
  title={SOLETE, a 15-month long holistic dataset including: Meteorology, co-located wind and solar PV power from Denmark with various resolutions},
  author={Pombo, Daniel Vazquez and Gehrke, Oliver and Bindner, Henrik W},
  journal={Data in Brief},
  volume={42},
  pages={108046},
  year={2022},
  publisher={Elsevier}
}


Nontheless, here is the citation for the git itself:
D. V. Pombo, The SOLETE platform (March, 2023).doi:10.11583/DTU.17040626.URL https://data.dtu.dk/articles/software/TheSOLETEplatform/17040626

@article{SOLETE2021Code,
author = "Daniel Vazquez Pombo",
title = "{The SOLETE platform}",
year = "2023",
month = "Mar,",
url = "https://data.dtu.dk/articles/software/The_SOLETE_platform/17040626",
doi = "10.11583/DTU.17040626"
} 




# References
    [1] Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month long holistic dataset including: Meteorology, co-located wind and solar PV power from Denmark with various resolutions. Data in Brief, 42, 108046.
        
    [2] Pombo, D. V., Bindner, H. W., Spataru, S. V., Sørensen, P. E., & Bacher, P. (2022). Increasing the accuracy of hourly multi-output solar power forecast with physics-informed machine learning. Sensors, 22(3), 749.
    
    [3] Pombo, D. V., Bacher, P., Ziras, C., Bindner, H. W., Spataru, S. V., & Sørensen, P. E. (2022). Benchmarking physics-informed machine learning-based short term PV-power forecasting tools. Energy Reports, 8, 6512-6520.
    
    [4] Pombo, D. V., Rincón, M. J., Bacher, P., Bindner, H. W., Spataru, S. V., & Sørensen, P. E. (2022). Assessing stacked physics-informed machine learning models for co-located wind–solar power forecasting. Sustainable Energy, Grids and Networks, 32, 100943.


