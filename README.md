# SOLETE
Author: **Daniel Vázquez Pombo** - Contact: dvapo@elektro.dtu.dk<br/>
LinkedIn: https://www.linkedin.com/in/dvp/<br/>
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo   
ORCID: https://orcid.org/0000-0001-5664-9421

This repository is complementary material to a series of papers covering Solar PV power forecasting [1, 2] and its twin "Data in Brief" article [3]. The objective is to increase the transparency of [1, 2], which is one of the main limitations of Machine Learning in general.
In addition, I included a number of functions that can be use by begginers to kickstart their projects in time series forecasting with different machine learning methods.

This work was developed under the PhD thesis Energy Management Systems for Isolated Hybrid Power Systems at the Technical University of Denmark (DTU).

# Dependencies
1. Python 3.8.10 
2. Pandas 1.2.4 
3. Numpy 1.19.5
4. Matplotlib 3.4.2
5. Scikit-Learn 0.24.2 
6. Keras 2.5.0
7. TensorFlow 2.5.0
8. The SOLETE dataset -> https://data.dtu.dk/articles/dataset/The_SOLETE_dataset/17040767
9. Functions.py 

# How to use
1. Store the SOLETE dataset in the same folder as the scripts from this repository 
2. Open the **RunMe.py** file. This allows you to load SOLETE and take a peak at its contents.
3. Opend the  **MLForecasting.py** file. This allows you to configure Random Forest (RF), Support Vector Machine (SVM), and three kinds of Artificial Neuronal Networks: Convolutional Neuronal Network (CNN), Long-Short Term Memory (LSTM), and a Hybrid (CNN-LSTM).
   - The file itself contains notes explaining how to use it.
   - You can basically replicate the studies from [1, 2] and build on top.
4. Have Fun!

You can of course use your own dataset, you will have to adapt things here and there, but you will be able to reuse most of the code.

# How to cite this:
D. V. Pombo, The SOLETE platform (Jan, 2022).doi:10.11583/DTU.17040626.URL https://data.dtu.dk/articles/software/TheSOLETEplatform/17040626

@article{SOLETE2021Code,
author = "Daniel Vazquez Pombo",
title = "{The SOLETE platform}",
year = "2022",
month = "Jan,",
url = "https://data.dtu.dk/articles/software/The_SOLETE_platform/17040626",
doi = "10.11583/DTU.17040626"
} 




# References
[1] D. V. Pombo, H. W. Bindner, S. V. Spataru, P. E. Sørensen, P. Bacher, Increasing the Accuracy of Hourly Multi-Output Solar Power Forecast with Physics-Informed Machine Learning, Sensors 22 (3) (2022) 749.
 
[2]  D. V. Pombo, P. Bacher, C. Ziras, H. W. Bindner, S. V. Spataru, P. E. Sørensen, Benchmarking Physics-Informed Machine Learning-based Short Term PV-Power Forecasting Tools, Under Review.

[3] D.V. Pombo, O.G. Gehrke, H.W. Bindner, SOLETE, a 15-month long holistic dataset including:  meteorology, co-located wind and solar PV power from Denmark with various resolutions, Data in Brief. In Press.


Copyright 2021 Technical University of Denmark.