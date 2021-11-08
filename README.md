# Music-Classification
## Music Classification project using keras and tensorflow
- [Project description](#description)
- [Audio preprocessing routine](#preprocessing)
- [Model Training](models)
# Description
This is a genre classification project, which classifies songs from the GTZAN dataset which consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.
At first a preprocessing routine extracting mfccs and labels from the data is performed using the preprocessing.py file on the dataset and then a couple of neural networks are trained to perform the task.
# Preprocessing
The preprocessing routine consists on taking the raw audio files and extracting the mfcc(Mel-Frequency Cepstral Coefficients) for use as features, the [Mel scale](https://en.wikipedia.org/wiki/Mel_scale) is a important scale on sound that allows for retrieval of perceptual information on speech, to understand mfccs we first need to look at a cepstrum is (no it is not **just** a funny wordplay on spectrum) basically a cepstrum is the result of a certain operation with discrete fourier transforms that can give a variety of important information for speech recognition about a audio sample, for example with a cepstrum you can separate more easily different aspects of speech such as the glottal pulse and the vocal tract frequency response(there is no need to understand in depth these concepts right now).
After this very comprehensive background, we can look into the algorithm for computing the mfcc(done by librosa in the code)
1. Take the Fourier transform of (a windowed excerpt of) a signal.
2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows or alternatively, cosine overlapping windows.
3. Take the logs of the powers at each of the mel frequencies.
4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
5. The MFCCs are the amplitudes of the resulting spectrum.
The resulting mfcc is very useful for describing "large" structures of the spectrum while ignoring small scale variations
# Models
The 2 models used were based on a small 3-layered CNN and a small RNN-LSTM respectively, which are both simple models to showcase what could be done, to see the implementation you can check both notebooks where they are trained but due to the simplicity and size of the architectures used they don't get quite a good accuracy measure, staying both in around 75% accuracy, a bigger, more sophisticated model could be used to try to get better results on this task, using perhaps a transfer learning approach with a pre-trained model.
