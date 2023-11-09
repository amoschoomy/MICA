# MICA
T cell receptor are indicative of the immune status in the human body. Utilizing T cell receptor (TCRs) for early cancer detection has shown significant effectiveness. While traditional multiple instance learning methods allow for the simultaneous analysis of relationships between multiple TCRs and direct result predictions, but they cannot efficiently process complex TCRs data. In light of this, we introduce a multiple instance learning method based on convolutional neural network and self-attention (MICA). Firstly, MICA employs word vectors techniques to preprocess TCRs, which enables the model to gain a deeper understanding of the sequence feature of individual TCRs while preserving vital information to the maximum extent. Secondly, at the instance level, MICA utilizes a parallel convolutional  neural network structure, providing comprehensive consideration of the distinct features of each instance across various dimensions. Thirdly, at the bag level, MICA ranks instance scores and utilizes an enhanced self-attention mechanism to extract relationship features between individual instances, thereby enabling more accurate result predictions. Following ten rounds of five-fold cross-validation, MICA exhibited the best performance on both lung cancer and thyroid cancer datasets, achieving AUC values of 0.911 and 0.946, respectively. These results outperformed other state-of-the-art methods by 7.1% and 2.1%.
# Data 
All the data used by MICA can be found in the "data" folder
# Dependency:
python 3.8.12 <br>
torch  1.10.0 <br>
numpy 1.21.2 <br>
sklearn 1.0 <br>
# Usage:
python train_and_test.py
