# Melanoma Detection using CNN and Transfer Learning 🔬🧠

A deep learning solution for automated melanoma (skin cancer) detection using a custom Convolutional Neural Network (CNN) and transfer learning with MobileNetV2. This project implements binary classification to distinguish between benign and malignant skin lesions using the HAM10000 dataset.

![Melanoma Detection](images/thumbnail.png)

## 📋 Description

Melanoma is the most dangerous type of skin cancer, and early detection is critical for successful treatment. This project leverages deep learning to automate the detection process, providing a reliable tool for preliminary skin lesion analysis.

The implementation compares two approaches: a custom CNN built from scratch and a transfer learning model using MobileNetV2 pre-trained on ImageNet, both trained on the HAM10000 dataset containing 10,000+ dermatoscopic images.

<br>
<div align="center">
  <a href="https://codeload.github.com/TendoPain18/melanoma-detection-cnn/legacy.zip/main">
    <img src="https://img.shields.io/badge/Download-Files-brightgreen?style=for-the-badge&logo=download&logoColor=white" alt="Download Files" style="height: 50px;"/>
  </a>
</div>

## 🎯 Project Objectives

1. **Build Custom CNN**: Design convolutional neural network for skin lesion classification
2. **Implement Transfer Learning**: Leverage MobileNetV2 for improved performance
3. **Binary Classification**: Distinguish between benign and malignant melanoma
4. **Data Preprocessing**: Handle class imbalance and data augmentation
5. **Model Comparison**: Evaluate custom CNN vs. transfer learning approach

## ✨ Features

### Two Model Approaches
- **Custom CNN**: 3-layer convolutional network built from scratch
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet with frozen weights
- **Data Augmentation**: Rotation, zooming, and flipping for robust training
- **Class Balancing**: Weighted loss to handle imbalanced dataset

### Data Processing
- **HAM10000 Dataset**: 10,000+ dermatoscopic images from Kaggle
- **Binary Labels**: 
  - Benign: nv, bkl, df, vasc
  - Malignant: mel, akiec, bcc
- **Train-Validation-Test Split**: 70%-15%-15% stratified split
- **Image Preprocessing**: 128×128 pixel images, normalized to [0,1]

### Model Evaluation
- **Accuracy Tracking**: Training and validation metrics
- **Loss Curves**: Model convergence visualization
- **Test Performance**: Final evaluation on held-out test set
- **Model Comparison**: Side-by-side performance analysis

## 🔬 Technical Approach

### Custom CNN Architecture

```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Key Components:**
- Three convolutional blocks with increasing filters (32→64→128)
- MaxPooling for spatial dimension reduction
- Dropout (0.3) for regularization
- Sigmoid activation for binary output

### Transfer Learning: MobileNetV2

```python
base = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False  # Freeze pre-trained weights

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Advantages:**
- Pre-trained feature extraction from ImageNet
- Lightweight architecture optimized for mobile devices
- Faster convergence with frozen weights
- Better performance with limited training epochs

### Data Augmentation

**Training Set Augmentation:**
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Random rotation ±20°
    zoom_range=0.2,         # Random zoom up to 20%
    horizontal_flip=True,   # Random horizontal flip
    vertical_flip=False     # No vertical flip
)
```

**Validation/Test Sets:**
- Only rescaling (normalization) applied
- No augmentation to maintain evaluation consistency

## 🚀 Getting Started

### Prerequisites

**Python Requirements:**
```
Python 3.7+
TensorFlow 2.x
Keras
NumPy
Pandas
Matplotlib
scikit-learn
kagglehub
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/melanoma-detection-cnn.git
cd melanoma-detection-cnn
```

2. **Install dependencies**
```bash
pip install kagglehub tensorflow matplotlib scikit-learn
```

3. **Run the notebook**
```bash
jupyter notebook Skin_Cancer__Melanoma__Detection_using_CNN_and_Transfer.ipynb
```

Or use Google Colab (recommended for GPU acceleration):
- Upload notebook to Colab
- Dataset downloads automatically via kagglehub

## 📖 Usage Guide

### Dataset Preparation

The notebook automatically downloads the HAM10000 dataset from Kaggle:

```python
import kagglehub
dataset_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```

**Dataset Structure:**
- HAM10000_metadata.csv: Lesion metadata
- HAM10000_images_part_1/: First half of images
- HAM10000_images_part_2/: Second half of images

### Binary Label Mapping

```python
benign = ["nv", "bkl", "df", "vasc"]      # Label 0
malignant = ["mel", "akiec", "bcc"]       # Label 1

df["label"] = df["dx"].apply(lambda x: 1 if x in malignant else 0)
```

### Training Process

**1. Train Custom CNN:**
```python
history_cnn = cnn.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    class_weight=class_weights  # Handle class imbalance
)
```

**2. Train Transfer Learning Model:**
```python
history_transfer = transfer_model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    class_weight=class_weights
)
```

**3. Evaluate Models:**
```python
cnn_test_loss, cnn_test_acc = cnn.evaluate(test_data)
tl_test_loss, tl_test_acc = transfer_model.evaluate(test_data)
```

### Visualization

The notebook generates accuracy and loss plots for both models:

```python
plt.plot(history_cnn.history['accuracy'], label='Train Acc')
plt.plot(history_cnn.history['val_accuracy'], label='Val Acc')
plt.title("CNN Accuracy")
plt.legend()
```

## 📊 Model Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 128×128 pixels |
| Batch Size | 32 |
| Epochs | 3 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Dropout Rate | 0.3 |

### Data Split

| Split | Percentage | Purpose |
|-------|------------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

All splits are stratified to maintain class balance.

## 🎓 Key Concepts

### Custom CNN vs. Transfer Learning

**Custom CNN Advantages:**
- Full control over architecture
- Learns task-specific features from scratch
- No dependency on external models

**Transfer Learning Advantages:**
- Faster training with frozen weights
- Better generalization from ImageNet features
- Higher accuracy with fewer epochs
- Efficient for limited computational resources

### Handling Class Imbalance

The notebook uses class weights to address the imbalanced dataset:

```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df["label"]),
    y=df["label"]
)
```

This ensures the model doesn't bias towards the majority class.

## 🔧 Implementation Details

### Image Preprocessing Pipeline

```python
train_gen = ImageDataGenerator(
    rescale=1./255,           # Normalize to [0,1]
    rotation_range=20,         # Data augmentation
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col="path",              # Image file paths
    y_col="label",             # Binary labels
    target_size=(128, 128),    # Resize all images
    batch_size=32,
    class_mode="binary"        # Binary classification
)
```

### Model Compilation

Both models use identical compilation settings:

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HAM10000 dataset from Kaggle (kmader/skin-cancer-mnist-ham10000)
- TensorFlow and Keras for deep learning frameworks
- MobileNetV2 architecture from TensorFlow Hub

<br>
<div align="center">
  <a href="https://codeload.github.com/TendoPain18/melanoma-detection-cnn/legacy.zip/main">
    <img src="https://img.shields.io/badge/Download-Files-brightgreen?style=for-the-badge&logo=download&logoColor=white" alt="Download Files" style="height: 50px;"/>
  </a>
</div>

## <!-- CONTACT -->
<div id="toc" align="center">
  <ul style="list-style: none">
    <summary>
      <h2 align="center">
        🚀
        CONTACT ME
        🚀
      </h2>
    </summary>
  </ul>
</div>
<table align="center" style="width: 100%; max-width: 600px;">
<tr>
  <td style="width: 20%; text-align: center;">
    <a href="https://www.linkedin.com/in/amr-ashraf-86457134a/" target="_blank">
      <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="https://github.com/TendoPain18" target="_blank">
      <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="mailto:amrgadalla01@gmail.com">
      <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="https://www.facebook.com/amr.ashraf.7311/" target="_blank">
      <img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="https://wa.me/201019702121" target="_blank">
      <img src="https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
</tr>
</table>
<!-- END CONTACT -->

## **Advancing early melanoma detection through deep learning! 🔬✨**
