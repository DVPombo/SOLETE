# SOLETE
Author: **Daniel Vázquez Pombo** - Contact: daniel.vazquez.pombo@gmail.com<br/>
LinkedIn: https://www.linkedin.com/in/dvp/<br/>
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo   
ORCID: https://orcid.org/0000-0001-5664-9421

This repository is complementary material to a series of papers covering Solar PV power forecasting [1, 2] and its twin "Data in Brief" article [3]. The objective is to increase the transparency of [1, 2], which is one of the main limitations of Machine Learning in general.
In addition, I included a number of functions that can be use by begginers to kickstart their projects in time series forecasting with different machine learning methods.

This work was developed under the PhD thesis Energy Management Systems for Isolated Hybrid Power Systems at the Technical University of Denmark (DTU).

# Dependencies
1. Python 3.9.12 
2. Pandas 1.5.0 
3. Numpy 1.23.1
4. Matplotlib 3.6.0
5. Scikit-Learn 1.1.2 
6. Keras 2.10.0
7. TensorFlow 2.10.0
8. The SOLETE dataset [1] -> https://doi.org/10.11583/DTU.17040767 

# How to use
1. Store the SOLETE dataset in the same folder as the scripts from this repository 
2. Open the **RunMe.py** file. This allows you to load SOLETE and sneak  a peek at its contents.
3. Open the  **MLForecasting.py** file. This allows you to configure Random Forest (RF), Support Vector Machine (SVM), and three kinds of Artificial Neuronal Networks: Convolutional Neuronal Network (CNN), Long-Short Term Memory (LSTM), and a Hybrid (CNN-LSTM).
   - The file itself contains notes explaining how to use it.
   - You can basically replicate the studies from [2, 3, 4] and build on top.
4. Have Fun!

You can of course use your own dataset, you will have to adapt things here and there, but you will be able to reuse most of the code.

# How to cite this:
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


