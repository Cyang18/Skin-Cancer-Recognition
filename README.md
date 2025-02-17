# Skin Cancer Classification using HAM10000 Dataset

## Overview
This project is focused on classifying different types of skin cancer using the **HAM10000** dataset from Kaggle. The dataset consists of 10,015 dermatoscopic images belonging to **7 different classes**. The goal of this project is to develop a machine learning model that can accurately classify skin cancer types using convolutional neural networks (CNNs).

## Dataset
The **HAM10000** dataset contains images of skin lesions categorized into the following **7 classes**:
1. Actinic keratoses (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevi (nv)
7. Vascular lesions (vasc)

I utilized the **hmnist_28_28_RGB.csv** file, which contains 28x28 pixel representations of the images, to preprocess and train our model.

## Model Architecture
The model consists of a **fully connected neural network** with dropout regularization. The architecture is as follows:
- **Input Layer** (28x28 images)
- **Flatten Layer** (Converts 2D images into a 1D vector)
- **Dense Layer** (128 units, ReLU activation)
- **Dropout Layer** (0.5 dropout rate to prevent overfitting)
- **Output Layer** (7 units, softmax activation)

### Model Compilation
```python
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28)), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])
```

### Training Strategy
To improve model performance, **early stopping** was implemented to halt training if validation loss does not improve for three consecutive epochs.
```python
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=3,         
    restore_best_weights=True,  
    verbose=1            
)

history = model.fit(X_train_resampled, Y_train_resampled, 
                    validation_split=0.2, 
                    batch_size=128, 
                    epochs=20, 
                    callbacks=[early_stopping])
```

## Training Details
- **Learning Rate**: 0.0001
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 128
- **Epochs**: 20 (early stopping enabled)

## Evaluation Metrics
To assess the model's performance, we used:
- **Accuracy**: Measures overall correctness
- **Confusion Matrix**: Visualizes misclassification patterns


## Results & Improvements
The model showed promising results, but there is still room for improvement. Some of the key areas for future work include:
- **Data Augmentation**: Enhancing the dataset with transformations to improve generalization.
- **Hyperparameter Tuning**: Experimenting with different learning rates, batch sizes, and optimizers.
- **Transfer Learning**: Utilizing pre-trained models like ResNet or VGG for better feature extraction.
  

## References
- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)


## License
This project is open-source under the MIT License.


