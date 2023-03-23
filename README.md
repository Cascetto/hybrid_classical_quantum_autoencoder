# Hybrid Classical Quantum AutoEncoder
Implementation of the HAE used for anomaly detection. 
The model can be though as a pipeline consisting of:
+ a classical Endoer
+ a Variational Quantum Circuit
+ a classical Decoder

After the model training, we can perform anomaly detection using the output of the VQC as the input of an Isolation Forest
