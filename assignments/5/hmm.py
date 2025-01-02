import os 
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


import os
import numpy as np
import librosa
from hmmlearn import hmm


class hmm_file:
    def main():
        data_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..','data', 'external' ,'hmm_data'))
        print(data_path)

        file_path = os.path.join(data_path, os.listdir(data_path)[0]) 
        print(file_path)
        audio, sr = librosa.load(file_path, sr=None)  

        from IPython.display import Audio
        Audio(audio, rate=sr)
        print(audio.shape)

        # print(data_path)

        mfcc_x = []
        mfcc_y = []

        for audio_file in os.listdir(data_path):
            file_path = os.path.join(data_path, audio_file)
            y, sr = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # print(mfcc.shape)
            # print(mfcc.T.shape)
            mfcc_x.append(mfcc.T)   
            mfcc_y.append(os.path.splitext(audio_file)[0][0])  
            
        # np.array(mfcc_x)

        print(mfcc_x[0].shape)
        X = np.vstack(mfcc_x)
        y = np.array(mfcc_y)

        print(X.shape)
        print(y.shape)          

        plt.figure(figsize=(16, 6))

        for digit in range(10):
            plt.subplot(2, 5, digit + 1) 
            digit_indices = [i for i, label in enumerate(mfcc_y) if label == str(digit)]    
            # print(digit_indices)
            avg_mfcc = mfcc_x[0]
            plt.imshow(avg_mfcc, cmap='viridis', origin='lower', aspect='auto')
            plt.colorbar(label='MFCC Coefficients')
            plt.title(f'Digit {digit}', fontsize=16)
            plt.xlabel('Time Frames', fontsize=14)
            plt.ylabel('MFCC Coefficients', fontsize=14)

        plt.tight_layout()
        plt.show()


        digit_mfccs = {str(digit): [] for digit in range(10)}
        for filename in os.listdir(data_path):
            if filename.endswith(".wav"):
                digit = filename[0]  
                file_path = os.path.join(data_path, filename)
                y, sr = librosa.load(file_path)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc = mfccs.T
                digit_mfccs[digit].append(mfcc)
                
        def manual_train_test_split(data_dict, test_size=0.2):
            train_data = {}
            test_data = {}
            for digit, mfcc_list in data_dict.items():

                n_samples = len(mfcc_list)
                n_test = int(n_samples * test_size)
                
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                
                test_indices = indices[:n_test]
                train_indices = indices[n_test:]
                
                train_data[digit] = [mfcc_list[i] for i in train_indices]
                test_data[digit] = [mfcc_list[i] for i in test_indices]
            
            return train_data, test_data

        train_data, test_data = manual_train_test_split(digit_mfccs, test_size=0.2)

        import seaborn as sns

        models = {}
        for digit, mfcc_list in train_data.items():

            X = np.concatenate(mfcc_list)
            lengths = [len(mfcc) for mfcc in mfcc_list]
            
            model = hmm.GaussianHMM(n_components=7, covariance_type='diag', n_iter=100, random_state=42)
            model.fit(X, lengths)
            models[digit] = model

        def visualize_mfccs(mfccs, title="MFCC Heatmap"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(mfccs.T, cmap="viridis", xticklabels=False, yticklabels=False)
            plt.title(title)
            plt.xlabel("Time Steps")
            plt.ylabel("MFCC Coefficients")
            plt.show()

        example_digit = "0"
        if digit_mfccs[example_digit]: 
            visualize_mfccs(digit_mfccs[example_digit][0], title=f"MFCC Heatmap for Digit {example_digit}")

        def evaluate_model(models, test_data):
            correct = 0
            total = 0
            for digit, mfcc_list in test_data.items():
                for mfcc in mfcc_list:
                    X = np.concatenate([mfcc])
                    lengths = [len(mfcc)]
                    
                    scores = {model_digit: model.score(X, lengths) for model_digit, model in models.items()}
                    
                    predicted_digit = max(scores, key=scores.get)
                    if predicted_digit == digit:
                        correct += 1
                    total += 1
            return correct / total

        accuracy = evaluate_model(models, test_data)
        print(f"Recognition Accuracy on Test Set: {accuracy * 100:.2f}%")

        def train_and_evaluate_hmm(train_data, test_data, n_components):

            models = {}
            for digit, mfcc_list in train_data.items():
                X = np.concatenate(mfcc_list)
                lengths = [len(mfcc) for mfcc in mfcc_list]
                
                model = hmm.GaussianHMM(
                    n_components=n_components, 
                    covariance_type='diag', 
                    n_iter=100, 
                )
                model.fit(X, lengths)
                models[digit] = model
            
            correct = 0
            total = 0
            for digit, mfcc_list in test_data.items():
                for mfcc in mfcc_list:
                    X = np.concatenate([mfcc])
                    lengths = [len(mfcc)]
                    
                    scores = {model_digit: model.score(X, lengths) 
                            for model_digit, model in models.items()}
                    
                    predicted_digit = max(scores, key=scores.get)
                    if predicted_digit == digit:
                        correct += 1
                    total += 1
            
            accuracy = correct / total
            return accuracy, models

        results = {}
        best_accuracy = 0
        best_models = None
        best_n_components = None

        for n_components in tqdm(range(3, 11)):
            accuracy, models = train_and_evaluate_hmm(train_data, test_data, n_components)
            print("check")
            results[n_components] = accuracy
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_models = models
                best_n_components = n_components
            
            print(f"Components: {n_components}, Accuracy: {accuracy * 100:.2f}%")

        plt.figure(figsize=(10, 6))
        components = list(results.keys())
        accuracies = [results[n] * 100 for n in components]

        plt.plot(components, accuracies, marker='o')
        plt.grid(True)
        plt.xlabel('Number of Components')
        plt.ylabel('Accuracy (%)')
        plt.title('HMM Performance vs Number of Components')

        for i, acc in enumerate(accuracies):
            plt.annotate(f'{acc:.1f}%', 
                        (components[i], acc),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

        plt.tight_layout()
        plt.show()

        print(f"\nBest performance achieved with {best_n_components} components: {best_accuracy * 100:.2f}%")

        def create_confusion_matrix(models, test_data):
            digits = sorted(test_data.keys())
            n_digits = len(digits)
            confusion_matrix = np.zeros((n_digits, n_digits))
            
            for true_idx, true_digit in enumerate(digits):
                for mfcc in test_data[true_digit]:
                    X = np.concatenate([mfcc])
                    lengths = [len(mfcc)]
                    
                    scores = {model_digit: model.score(X, lengths) 
                            for model_digit, model in models.items()}
                    pred_digit = max(scores, key=scores.get)
                    pred_idx = digits.index(pred_digit)
                    
                    confusion_matrix[true_idx][pred_idx] += 1
            
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, 
                        annot=True, 
                        fmt='.2f',
                        xticklabels=digits,
                        yticklabels=digits,
                        cmap='Blues')
            plt.title('Confusion Matrix for Best Model')
            plt.xlabel('Predicted Digit')
            plt.ylabel('True Digit')
            plt.tight_layout()
            plt.show()
            
            return confusion_matrix

        conf_matrix = create_confusion_matrix(best_models, test_data)

        my_voice = {str(digit): [] for digit in range(10)}
        data_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..','data', 'external' ,'my_speech'))
        for filename in os.listdir(data_path):
            if filename.endswith(".wav"):
                digit = filename[0]  
                file_path = os.path.join(data_path, filename)
                y, sr = librosa.load(file_path)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc = mfccs.T
                my_voice[digit].append(mfcc)

        models = {}
        for digit, mfcc_list in train_data.items():
            X = np.concatenate(mfcc_list)
            lengths = [len(mfcc) for mfcc in mfcc_list]
            model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=100)
            model.fit(X, lengths)
            models[digit] = model

        def evaluate_model(models, test_data):
            correct = 0
            total = 0
            for digit, mfcc_list in test_data.items():
                for mfcc in mfcc_list:
                    X = np.concatenate([mfcc])
                    lengths = [len(mfcc)]
                    
                    scores = {model_digit: model.score(X, lengths) for model_digit, model in models.items()}
                    
                    predicted_digit = max(scores, key=scores.get)
                    if predicted_digit == digit:
                        correct += 1
                    total += 1
            return correct / total

        accuracy = evaluate_model(models, my_voice)
        print(f"Recognition Accuracy on Test Set: {accuracy * 100:.2f}%")

