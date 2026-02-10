# ASL Alphabet Classification

## 1. Problem Description
This project aims to classify images of American Sign Language hand gestures into their corresponding English alphabet letters. The dataset used is the **Sign Language MNIST**.

## 2. Dataset Description
- **Source**: Sign Language MNIST (Kaggle/NIST)
- **Format**: 28x28 pixel grayscale images (CSV format)
- **Training Samples**: 27,455
- **Test Samples**: 7,172
- **Classes**: 24 (A-Y, excluding J and Z which require motion)
    - **Labels**: 0-24 (where 9=J and 25=Z are skipped in the original dataset, but mapped here for simplicity to 0-24 range or kept as is with exclusion).
    - **Mapping**: A=0, B=1, ..., Y=24.

## 3. Architecture Diagrams
```
Input (28x28) -> Flatten -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(25, Softmax)
```
- **Parameters**: ~237,000

### Convolutional Neural Network (CNN)
A custom CNN designed to exploit spatial structure.
```
Input (28x28x1)
   ↓
[Conv2D (32, 3x3) + ReLU] -> [MaxPooling (2x2)]
   ↓
[Conv2D (64, 3x3) + ReLU] -> [MaxPooling (2x2)]
   ↓
Flatten -> Dropout(0.5) -> Dense(128, ReLU) -> Output(25, Softmax)
```
- **Parameters**: ~420,000
- **Key Features**: 
    - **Conv Layers**: Extract local features (edges, shapes).
    - **Pooling**: Reduces dimensionality and adds translation invariance.
    - **Dropout**: Prevents overfitting.

## 4. Experimental Results

| Model | Accuracy (Test) | Observations |
|-------|-----------------|--------------|
| **Baseline (Dense)** | ~80-85% | Struggles with spatial variations; high parameter count for simple task. |
| **CNN (3x3 Kernel)** | **~90-95%+** | Significantly better generalization due to spatial feature learning. |
| **CNN (5x5 Kernel)** | ~90-94% | Comparable, but 3x3 is generally more efficient for 28x28 images. |

## 5. Interpretation and Architectural Reasoning

### Why CNNs Outperform Baseline?
1.  **Spatial Hierarchy**: CNNs preserve the 2D grid structure, allowing them to learn relationships between adjacent pixels (e.g., finger shapes). Flattening destroys this.
2.  **Locality**: Filters learn local features (edges, curves) that are composed into larger shapes.
3.  **Translation Invariance**: A hand gesture shifted 2 pixels to the left is still the same gesture. CNNs with pooling handle this naturally; Dense networks must relearn the pattern for every position.

### Inductive Bias of Convolution
-   **Translation Equivariance**: $f(g(x)) = g(f(x))$. Shifting the input shifts the feature map.
-   **Locality**: Inputs close to each other are related.

### When is Convolution NOT Appropriate?
1.  **Tabular Data**: Where feature order doesn't matter (e.g., Age, Salary, ZipCode).
2.  **Permutation Invariant Data**: Sets of objects without a grid structure.
3.  **No Local Correlations**: If the relationship between pixels (0,0) and (0,1) is no stronger than (0,0) and (27,27).