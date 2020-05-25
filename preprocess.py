import os
import librosa
import math
import json

DATASET_PATH = r'C:\Users\Julio\Desktop\genres'
JSON_PATH = 'data.json'

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    #data dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples = int(SAMPLES_PER_TRACK / num_segments)
    expected_mfcc_vectors = math.ceil(num_samples / hop_length)

    #loop through genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
            
            #getting labels
            semantic_label = dirpath.split('/')[-1]
            data["mapping"].append(semantic_label)
            print("\n Processing {}".format(semantic_label))

            for f in filenames:
                
                #getting audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                #extracting mfcc
                for s in range(num_segments):
                    start_sample = num_samples * s
                    finish_sample = start_sample + num_samples

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                    sr=sr, n_fft = n_fft, n_mfcc = n_mfcc, hop_length = hop_length)

                    mfcc = mfcc.T

                    if len(mfcc) == expected_mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))
                    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)



if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)