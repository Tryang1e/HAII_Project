import os
import glob
import csv
import json
import numpy as np
import cv2

# ================= Configuration =================
DATA_DIR = '../gesture_dataset'  # Directory containing CSV files (run from the parent folder where CSVs are saved)
TARGET_FRAMES = 20  # Resample each gesture sequence to this many frames
MODEL_SAVE_PATH = 'virtual-touch-click/gesture_svm_model.xml'
LABEL_MAP_PATH = 'virtual-touch-click/gesture_labels.json'
# =================================================

def extract_features(data_rows, target_frames=TARGET_FRAMES):
    """
    Interpolate the time-series angle data to a fixed number of frames.
    """
    angles = np.array(data_rows, dtype=np.float32)
    
    num_frames = angles.shape[0]
    num_features = angles.shape[1]
    
    if num_frames == target_frames:
        resampled = angles
    else:
        resampled = np.zeros((target_frames, num_features), dtype=np.float32)
        original_indices = np.linspace(0, 1, num_frames)
        target_indices = np.linspace(0, 1, target_frames)
        
        for i in range(num_features):
            resampled[:, i] = np.interp(target_indices, original_indices, angles[:, i])
            
    return resampled.flatten()

def main():
    print("Searching for gesture data CSV files...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_data_*.csv"))
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        print("Make sure you are running this from the directory where the CSV files are saved.")
        return
        
    X = []
    y_str = []
    
    print(f"Found {len(csv_files)} gesture files. Processing...")
    for file in csv_files:
        basename = os.path.basename(file)
        gesture_name = basename.split('_data_')[0]
        
        try:
            data = []
            with open(file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    # Ignore the first column (time_sec), keep the rest
                    data.append([float(val) for val in row[1:]])
                    
            if len(data) < 5:
                print(f"Skipping {basename} (Sequence too short)")
                continue
            if len(data[0]) == 15:
                data_left = [row + [0.0] * 15 for row in data]
                data_right = [[0.0] * 15 + row for row in data]
                
                features_left = extract_features(data_left)
                features_right = extract_features(data_right)
                
                X.append(features_left)
                y_str.append(gesture_name)
                
                X.append(features_right)
                y_str.append(gesture_name)
                continue
                    
            # Skip if it's still not 30 features
            if len(data[0]) != 30:
                print(f"Skipping {basename} (Invalid feature dimension: {len(data[0])}, expected 30)")
                continue
                
            features = extract_features(data)
            X.append(features)
            y_str.append(gesture_name)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    # OpenCV SVM requires integer labels
    unique_labels = sorted(list(set(y_str)))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    int_to_label = {i: lbl for i, lbl in enumerate(unique_labels)}
    
    y = [label_to_int[lbl] for lbl in y_str]
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")
    print("\nClass distribution:")
    for lbl in unique_labels:
        count = y_str.count(lbl)
        print(f" - {lbl}: {count} samples")
    
    if len(X) < 10:
        print("Dataset is too small to split meaningfully! Collect more data.")
        return
        
    # Split data into training and test sets
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print("\nTraining OpenCV SVM model...")
    # scikit-learn의 기본 gamma='scale' 방식을 모방 (분산 역순)
    var = np.var(X_train) if np.var(X_train) > 0 else 1.0
    gamma = 1.0 / (X_train.shape[1] * var)
    
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(50.0) # 과적합을 허용하므로 마진 페널티를 크게 설정
    svm.setGamma(gamma)
    
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    
    # Evaluate model on the train and test split
    _, y_pred_train = svm.predict(X_train)
    train_acc = np.mean(y_pred_train.flatten().astype(np.int32) == y_train)
    
    _, y_pred_test = svm.predict(X_test)
    test_acc = np.mean(y_pred_test.flatten().astype(np.int32) == y_test)
    
    print(f"\nModel Evaluation:")
    print(f" - Train Accuracy (과적합 지표): {train_acc*100:.2f}%")
    print(f" - Test Accuracy (검증 지표): {test_acc*100:.2f}%")
    
    # 성능 극대화를 위해 *전체 데이터*로 최종 재학습 (과적합 허용)
    print("Retraining on all data for maximum deployment performance...")
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    svm.save(MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")
    
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(int_to_label, f)
    print(f"Label assignments saved to {LABEL_MAP_PATH}")

if __name__ == '__main__':
    main()
