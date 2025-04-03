import os
import ctypes
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =============================================
# WORKAROUND FOR WINDOWS TENSORFLOW ISSUE
# =============================================
if os.name == 'nt':
    try:
        # Backup original c_int
        original_c_int = ctypes.c_int
        
        # Create a new c_int that can handle larger values
        class c_int_new(ctypes.c_int):
            _type_ = ctypes.c_longlong
            
        # Replace c_int with our new version
        ctypes.c_int = c_int_new
        
        # Also patch c_uint if needed
        original_c_uint = ctypes.c_uint
        class c_uint_new(ctypes.c_uint):
            _type_ = ctypes.c_ulonglong
        ctypes.c_uint = c_uint_new
        
        print("Applied Windows integer overflow workaround")
    except Exception as e:
        print(f"Couldn't apply ctypes workaround: {e}")

# Additional TensorFlow configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.keras.backend.set_floatx('float32')

# =============================================
# MODEL PARAMETERS
# =============================================
MAX_AUDIO_DURATION = 3   # Further reduced from 5 to 3 seconds
SAMPLE_RATE = 16000      # Reduced from 22050 to 16000
FIXED_MAX_LENGTH = 500   # Further reduced from 1000 to 500
N_MFCC = 13              # Standard MFCC count (reduced from 20)

def load_and_preprocess_data(audio_path, label):
    """Load audio and extract MFCC features with fixed duration"""
    try:
        # Load audio file with fixed duration
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_AUDIO_DURATION)
        
        # Add random silence at the start or end occasionally (data augmentation)
        if np.random.random() > 0.7:
            silence_length = int(sr * np.random.uniform(0.1, 0.3))
            position = 'start' if np.random.random() > 0.5 else 'end'
            silence = np.zeros(silence_length)
            if position == 'start':
                audio = np.concatenate([silence, audio])
            else:
                audio = np.concatenate([audio, silence])
            # Trim to original length
            if len(audio) > sr * MAX_AUDIO_DURATION:
                audio = audio[:int(sr * MAX_AUDIO_DURATION)]
        
        # Extract MFCC features 
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Add delta features only (skip delta-delta to reduce dimensions)
        delta_mfcc = librosa.feature.delta(mfcc)
        
        # Stack features (reduced from 3 to 2 feature sets)
        combined_features = np.concatenate([mfcc, delta_mfcc], axis=0)
        
        # Transpose for model input
        return combined_features.T, label
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, label

def prepare_dataset(real_paths, fake_paths):
    """Prepare dataset from real and fake audio file paths"""
    X, y = [], []
    
    # Load real audio files and label them as 0 (real)
    for path in real_paths:
        features, label = load_and_preprocess_data(path, 0)
        if features is not None:
            X.append(features)
            y.append(label)
    
    # Load fake audio files and label them as 1 (fake)
    for path in fake_paths:
        features, label = load_and_preprocess_data(path, 1)
        if features is not None:
            X.append(features)
            y.append(label)
    
    return X, np.array(y)

def augment_data(X, y):
    """Augment data with various transformations, with more augmentation for minority class"""
    augmented_X, augmented_y = [], []
    
    # Count instances of each class
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)
    
    # Augmentation multiplier for minority class
    minority_multiplier = min(3, max(1, class_counts[majority_class] // class_counts[minority_class]))
    
    # Loop through each feature set and label
    for features, label in zip(X, y):
        # Original features
        augmented_X.append(features)
        augmented_y.append(label)
        
        # More augmentation for minority class
        augmentation_rounds = minority_multiplier if label == minority_class else 1
        
        for _ in range(augmentation_rounds):
            # Add noise
            if np.random.random() > 0.3:
                noise_factor = np.random.uniform(0.01, 0.05)  # Increased noise
                noise = np.random.normal(0, noise_factor, features.shape)
                augmented_features = features + noise
                augmented_X.append(augmented_features)
                augmented_y.append(label)
            
            # Time masking
            if np.random.random() > 0.5:
                masked_features = features.copy()
                mask_len = np.random.randint(5, min(15, features.shape[0] // 3))  # Increased mask length
                mask_start = np.random.randint(0, features.shape[0] - mask_len)
                masked_features[mask_start:mask_start+mask_len, :] = 0
                augmented_X.append(masked_features)
                augmented_y.append(label)
            
            # Feature scaling
            if np.random.random() > 0.6:
                scale_factor = np.random.uniform(0.8, 1.2)  # Wider scale range
                scaled_features = features * scale_factor
                augmented_X.append(scaled_features)
                augmented_y.append(label)
            
            # Frequency masking (new augmentation)
            if np.random.random() > 0.7:
                freq_masked_features = features.copy()
                freq_mask_size = np.random.randint(2, min(5, features.shape[1] // 2))
                freq_start = np.random.randint(0, features.shape[1] - freq_mask_size)
                freq_masked_features[:, freq_start:freq_start+freq_mask_size] = 0
                augmented_X.append(freq_masked_features)
                augmented_y.append(label)
    
    print(f"Original dataset size: {len(X)}")
    print(f"Augmented dataset size: {len(augmented_X)}")
    print(f"Class distribution in augmented dataset: {np.bincount(np.array(augmented_y))}")
    
    return augmented_X, np.array(augmented_y)

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def create_model(input_shape):
    """Create a model with regularization to prevent overfitting"""
    model = Sequential([
        Input(shape=input_shape),
        
        # Add masking for variable length sequences
        Masking(mask_value=0.0),
        
        # Bidirectional GRU instead of LSTM (less prone to overfitting)
        Bidirectional(GRU(32, return_sequences=False, 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        BatchNormalization(),
        Dropout(0.5),  # Increased dropout
        
        # Dense layers with regularization
        Dense(16, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.4),  # Increased dropout
        
        Dense(2, activation='softmax')
    ])
    
    # Use Adam optimizer with reduced learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=10):
    """Train and evaluate the model with early stopping and learning rate reduction"""
    # Create checkpoint directory
    os.makedirs('model', exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('model/deepvoice_detector.h5', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Calculate class weights to handle imbalance
    class_weights = None
    if len(np.unique(np.argmax(y_train, axis=1))) > 1:
        n_samples = np.sum(np.argmax(y_train, axis=1).shape[0])
        n_classes = np.bincount(np.argmax(y_train, axis=1))
        class_weights = {i: n_samples / (len(n_classes) * n_classes[i]) for i in range(len(n_classes))}
        print(f"Using class weights: {class_weights}")
    
    # Train the model with slightly larger batch size and monitoring val_loss
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.25,  # Increased validation set size
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return history

def main():
    try:
        # Force TensorFlow to use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Using CPU for training (to avoid memory issues)")
        
        # Create directories
        os.makedirs('model', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Set path to dataset
        base_path = "Audio"
        
        # Print directory contents
        print("Checking directory structure:")
        print(f"Looking in: {base_path}")
        if os.path.exists(base_path):
            print(f"Contents of {base_path}:")
            for item in os.listdir(base_path):
                print(f"  - {item}")
        else:
            print(f"Directory {base_path} does not exist!")
            return
        
        # Determine the correct subfolder names
        real_folder = None
        fake_folder = None
        
        for item in os.listdir(base_path):
            if "real" in item.lower():
                real_folder = item
            elif "fake" in item.lower():
                fake_folder = item
        
        if not real_folder or not fake_folder:
            print("Could not find 'real' and 'fake' folders in the Audio directory.")
            return
        
        print(f"Found folders: real='{real_folder}', fake='{fake_folder}'")
        
        # Get real and fake audio paths
        real_paths = [os.path.join(base_path, real_folder, f) for f in os.listdir(os.path.join(base_path, real_folder))]
        fake_paths = [os.path.join(base_path, fake_folder, f) for f in os.listdir(os.path.join(base_path, fake_folder))]
        
        # Print dataset statistics
        print(f"Number of real audio samples: {len(real_paths)}")
        print(f"Number of fake audio samples: {len(fake_paths)}")
        print(f"Total number of samples: {len(real_paths) + len(fake_paths)}")
        
        if len(real_paths) == 0 or len(fake_paths) == 0:
            print("Error: No audio files found. Check your directory structure.")
            return
        
        # Prepare dataset
        X, y = prepare_dataset(real_paths, fake_paths)
        
        if not X:
            print("Error: Failed to process any audio files successfully.")
            return
            
        # Data augmentation with more aggressive settings
        X_augmented, y_augmented = augment_data(X, y)
        
        # Determine max length
        lengths = [len(seq) for seq in X_augmented]
        print(f"Feature sequence length stats - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}")
        
        # Use minimum of actual max or fixed max
        max_length = min(FIXED_MAX_LENGTH, max(lengths))
        print(f"Using sequence length: {max_length}")
        
        # Save parameters for inference
        joblib.dump(max_length, 'model/max_length.joblib')
        joblib.dump(N_MFCC, 'model/n_mfcc.joblib')
        
        # Pad sequences
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X_padded = pad_sequences(X_augmented, maxlen=max_length, dtype='float32', padding='post', truncating='post')
        
        # Print shape info
        print(f"Shape of padded dataset: {X_padded.shape}")
        
        # Split data with stratification and more test data
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_augmented, test_size=0.3, random_state=42, stratify=y_augmented
        )
        
        # Convert labels to categorical
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of X_test: {X_test.shape}")
        
        # Create model with regularization
        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # Train model with slightly larger batch size (10) and more epochs (30)
        history = train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=10)
        
        # Plot training history
        plot_training_history(history)
        
        # Make predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
        
        # Save results to file
        with open('results/classification_report.txt', 'w') as f:
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
        
        # Save probability distributions to help with threshold analysis
        plt.figure(figsize=(10, 6))
        real_probs = y_pred_probs[y_true == 0][:, 0]
        fake_probs = y_pred_probs[y_true == 1][:, 1]
        
        plt.hist(real_probs, alpha=0.7, label='Real confidence', bins=10)
        plt.hist(fake_probs, alpha=0.7, label='Fake confidence', bins=10)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Distribution of Confidence Scores')
        plt.savefig('results/confidence_distribution.png')
        
        print("Training complete! Results saved to 'results' directory.")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()