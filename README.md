# SOLETE
Author: **Daniel Vázquez Pombo** - Contact: dvapo@elektro.dtu.dk<br/>
LinkedIn: https://www.linkedin.com/in/dvp/<br/>
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo   
ORCID: https://orcid.org/0000-0001-5664-9421

This repository is complementary material to a paper covering Solar PV power forecasting [1] and its twin "Data in Brief" article [2]. The objective is to increase the transparency of [1], which is one of the main limitations of Machine Learning in general.

This work was developed under the PhD thesis Energy Management Systems for Isolated Hybrid Power Systems at the Technical University of Denmark (DTU).

# Note for reviewers
Whether you are reviewing either one of those papers you should be aware of the following:

1. What you can find here is a "trailer" of what will be provided upon acceptance.
2. The dataset only contains a minimum number of samples
3. There is a file missing called Functions.py which contains the most of the code.
4. Files RunMe.py and MlForecasting.py depend on it to be run, but you can see at the moment what they cover.

# Dependencies
1. Python 3.8.10 
2. Pandas 1.2.4 
3. Numpy 1.19.5
4. Matplotlib 3.4.2
5. Scikit-Learn 0.24.2 
6. Keras 2.5.0
7. TensorFlow 2.5.0
8. The SOLETE dataset -> https://figshare.com/s/19c47623082dfb2d2083
9. Functions.py -> A script that will be added to this git uppon publication of the related papers.

# How to use
1. Store the SOLETE dataset in the same folder as the scripts from this repository 
2. Open the **RunMe.py** file. This allows you to load SOLETE and take a peak at its contents.
3. Opend the  **MLForecasting.py** file. This allows you to configure Random Forest (RF), Support Vector Machine (SVM), and three kinds of Artificial Neuronal Networks: Convolutional Neuronal Network (CNN), Long-Short Term Memory (LSTM), and a Hybrid (CNN-LSTM).
   - The file itself contains notes explaining how to use it.
   - You can basically replicate the study from [1] and build on top.
4. Have Fun!

You can of course use your own dataset, you will have to adapt things here and there, but you will be able to reuse most of the code.

# How to cite this:
- under construction -

# References
[1] D.V. Pombo, H.W. Bindner, S.V. Spataru, P. Sørensen, P. Bacher, Increasing the Accuracy of Hourly Multi-Output Solar Power Forecast with Physics-Informed Machine Learning, Solar Energy. In Press.

[2] D.V. Pombo, O.G. Gehrke, H.W. Bindner, Solete, a 15-month long holistic dataset including:  meteorology, co-located wind and solar PV power from Denmark with hourly resolution, Data in Brief. In Press.
