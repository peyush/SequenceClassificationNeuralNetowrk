# SequenceClassificationNeuralNetowrk
Classifying sequences into classes using variety of Neural Network Models

#Data
Not the usual sequence to label task.
We have a dataset comprising of 'n' sequences('d' dim vectors) with each sequence being labelled as a class.
Thus its a 'n * d' size matrix with each row depending on its predecessor rows. Each row has a class associated with it.
TAsk is to classify the sequences in the test dataset.

#Base Model 
Using Logistic Regression - Assuming no corelation between the rows.

#RNN-with-seq
(Taken from http://robromijnders.github.io/LSTM_tsc/)

Make 'k * seq_len * d' sized tensor and feed into the LSTM 'batch_size * seq_len * d' where each 'seq_len * d' is a 
sequence sample with class decided by the last row in it.
Thus labels are also 'k * 1'.
Finally feed into the dense layer for classification.
