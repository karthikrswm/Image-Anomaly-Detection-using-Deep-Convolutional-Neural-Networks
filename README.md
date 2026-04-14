# Deep Learning Framework for Concrete Crack Detection

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Google Colab](https://img.shields.io/badge/Google_Colab-Environment-yellow.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data_Manipulation-blueviolet.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-success.svg)
![Matplotlib/Seaborn](https://img.shields.io/badge/Matplotlib_&_Seaborn-Data_Visualization-lightgrey.svg)

This project presents a robust and scalable Deep Learning framework to detect and classify cracks in concrete structures. Identifying structural damage early is critical for civil engineering and maintenance. The pipeline extracts critical visual features from images and trains several robust Convolutional Neural Networks (CNNs) entirely within a Google Colab environment, leveraging Google Drive for data storage and output persistence and NVMe storage for fast data processing.

## 📝 Datasets
The project evaluates the models across two distinct, publicly available datasets to ensure robust generalization:
- **Dataset 1:** [Concrete Crack Images for Classification (Mendeley Data)](https://data.mendeley.com/datasets/5y9wdsg2zt/2)
  - Contains precisely cropped regions of positive (cracked) and negative (non-cracked) concrete surfaces. Extremely high scale (40,000 images).
- **Dataset 2:** [Crack Detection Dataset (GitHub - tjdxxhy)](https://github.com/tjdxxhy/crack-detection)
  - Contains labeled pictures of concrete structures spanning diverse resolutions and cracking severity (split securely via train/val files).

## 🛠️ Technologies & Libraries Used
- **Environment:** ![Google Colab](https://img.shields.io/badge/Google_Colab-Environment-yellow.svg) (GPU mode - T4) optimized with Mixed-Precision training.
- **Storage:** ![Google Drive](https://img.shields.io/badge/Google_Drive-Storage-blue.svg) (Mounted for input data & output export)
- **Language:** ![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
- **Deep Learning:** ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red.svg)
- **Machine Learning:** ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
- **Data Manipulation:** ![NumPy](https://img.shields.io/badge/NumPy-Data_Manipulation-blueviolet.svg) ![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-success.svg)
- **Visualization:** ![Matplotlib/Seaborn](https://img.shields.io/badge/Matplotlib_&_Seaborn-Data_Visualization-lightgrey.svg)

---

### 1. Detailed Preprocessing & Pipeline Mechanisms

A major focus of this framework is the elimination of I/O bottlenecks and the maximization of GPU utilization via advanced data engineering.

- **Advanced Environment & Hardware Setup:** 
  - Activated the `mixed_float16` globally across the Keras runtime. This precision policy computes neural activations in fast 16-bit floats while securely retaining crucial weight updates in 32-bit (`variable_dtype=float32`). This optimization slashes GPU Memory (VRAM) consumption heavily, allowing for a 3x throughput speedup on NVIDIA T4/A100 GPUs and expanding the theoretical maximum batch size limits.
  - To completely circumvent Google Drive latency, raw monolithic archives (`Concrete Crack Images for Classification.rar` containing 40,000 files and `picture.zip` arrays) are explicitly extracted dynamically onto the Colab instance's direct-attached NVMe SSD storage during the first execution cell.

- **Dataset-Specific Parsing Strategies:**
  - **Dataset 1 Handling (Automated Directory Trees):** Utilizes `image_dataset_from_directory` to automatically infer class structures ("Positive" vs "Negative") straight from the extracted SSD subdirectories.
  - **Dataset 2 Handling (Manual Custom Parsing):** Images are decoded fundamentally differently by parsing arbitrary text logs (`train.txt` & `val.txt`). Custom paths are generated dynamically alongside sparse labels, subsequently mapped directly into memory via native `tf.data.Dataset.from_tensor_slices`.

- **High-Performance `tf.data` Pipeline Construction:**
  - All image buffers are actively structured into vectors of `batch_size=32`. 
  - Every image tensor is strictly scaled to `224x224x3` to standardize dimensions across architectures. Normalization is offloaded to the GPU gracefully via an inline `layers.Rescaling(1./255)` applied as the raw input to each network.
  - **Overfitting Prevention:** Training dataset partitions incorporate dynamic randomized sampling via `shuffle(buffer_size=2000)` to destroy any inherent sequential biases from the image storage prior to epoch injection.
  - **Asynchronous Execution:** Pipelines terminate exclusively with `prefetch(AUTOTUNE)`. This decouples the CPU/GPU cycle seamlessly, allowing the CPU to load, decode, and queue the next batch of JPEGs in RAM *while* the GPU processes the current batch.

- **Data Partitions:** Every experiment leverages a rigidly separated `80% Training / 20% Validation` cross-section. For Dataset 1, it's defined explicitly via automated sub-setting rules utilizing `seed=42` to guarantee absolutely reproducible benchmarks.

### 2. Advanced Model Architectures
Throughout the project, all CNN outputs terminate in a continuous standard Dense top compiled utilizing the `Adam` optimizer tracking `binary_crossentropy` loss alongside evaluation specific metrics (Accuracy, Precision, Recall, F1).

#### A. Basic CNN (Baseline Setup)
Designed for lightweight iteration speed while preventing network decay using active dimensionality reduction techniques.
- **Input Dimensions:** `224x224x3`
- **Feature Extraction Stages:** Composed of two sequential visual blocks: `Conv2D (64 filters, 3x3)` paired symmetrically with `BatchNormalization()`. The Batch Normalization zeroes the numerical mean, effectively limiting vanishing gradient problems. Non-linear representations flow out via a standard `ReLU` activation scaling deeply into a `MaxPooling2D (2x2)`. 
- **Topology:** The terminal `Flatten()` maps spatial nodes sequentially into a fully connected layer: `Dense(128) -> Dropout(0.3) -> Dense(1)`.

#### B. Zhang CNN (Theoretical Architectural Sub-sampling)
Modeled specifically upon theoretical architectures designed strictly to localize extreme high-frequency dimensional anomalies like sharp structural fracturizations.
- **Input Dimensions:** `224x224x3`
- **Progressive Aggressive Mapping:** Forces aggressive spatial map reductions early using a broad `Conv2D (48 filters, 11x11 kernel, stride=4)`. 
- **Sub-sampling Flow:** Progressively filters raw spatial density utilizing layered blocks: `MaxPooling2D(stride=2)` into `BatchNorm` moving into `Conv2D (128, 5x5)`. Expanding depth continues through `Conv2D (192, 3x3)`.
- **Topology:** The continuous data trails into `Flatten() -> Dense(256) -> Dropout(0.5) -> Dense(1)` utilizing massive dropouts to deter arbitrary network memorization.

#### C. VGG16 Fine-Tuned (Advanced Transfer Learning)
Capitalizes on generalized hierarchical characteristics parsed over thousands of categories inside the standard VGG16 structure.
- **Input Integration:** Initial raw tensors map mathematically correctly to ImageNet's original color distributions utilizing an initial inline `Lambda(applications.vgg16.preprocess_input)` layer pipeline.
- **Domain Adaptation Mechanism:** All generic feature extraction `blocks` within VGG16 are structurally frozen acting as mathematical standards, cleanly preventing back-propagation distortion—with the distinct exception of `block5` (`layer.trainable = True`). Adapting only terminal edges enables nuanced focus specifically for distinct concrete fractions. 
- **Topology:** Utilizing modern strategies, the network inherently dodges over-parametrization of linear nodes by integrating `GlobalAveragePooling2D()` flowing entirely into `Dense(256) -> Dropout(0.5) -> Dense(1)`.
- **Training Constraints:** Compiled dynamically avoiding standard optimizers, manually enforcing `optimizers.Adam(learning_rate=1e-4)` allowing the parameters to delicately tune pre-trained convolutions avoiding aggressive mathematical collapses.

### 3. Automated Training Guardrails
Inside the master fitting pipeline, automated structural loops deploy sophisticated `Callbacks`: 
- `ModelCheckpoint` isolates solely the absolute strongest epochs judging by strict internal validations on `val_accuracy` values.
- `EarlyStopping(patience=5)` halts wasted loop iterations while `ReduceLROnPlateau(factor=0.5)` functionally throttles gradients dynamically attempting convergence discovery when validations plateau mathematically on local loss minimums.

---

## 📊 Results & Comparative Analysis

The models were benchmarked fiercely, tracking Accuracy, Precision, Recall, and F1-Score to ensure class-balance stability. 

### Dataset 1 Evaluation
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **VGG16 Fine-Tuned** | **99.95%** | 99.92% | **99.97%** | **99.94%** |
| **Zhang CNN** | 99.92% | **99.97%** | 99.87% | 99.92% |
| **Basic CNN** | 99.22% | 99.07% | 99.37% | 99.22% |

<p align="left">
  <img src="./Output/Comparisons/Dataset1_BarChart.png" alt="Dataset 1 Bar Chart" width="100%" />
  <br />
  <img src="./Output/Comparisons/Dataset1_Combined_ROC.png" alt="Dataset 1 ROC Curve" width="100%" />
</p>

#### Dataset 1 Confusion Matrices & Training History
<details>
<summary>Click to expand charts for Dataset 1</summary>

**VGG16 Fine-Tuned**
- <img src="./Output/Dataset1/VGG16_FineTuned/VGG16_FineTuned_cm.png" alt="CM" width="300" /> <img src="./Output/Dataset1/VGG16_FineTuned/VGG16_FineTuned_history.png" alt="History" width="400" />

**Zhang CNN**
- <img src="./Output/Dataset1/Zhang_CNN/Zhang_CNN_cm.png" alt="CM" width="300" /> <img src="./Output/Dataset1/Zhang_CNN/Zhang_CNN_history.png" alt="History" width="400" />

**Basic CNN**
- <img src="./Output/Dataset1/Basic_CNN/Basic_CNN_cm.png" alt="CM" width="300" /> <img src="./Output/Dataset1/Basic_CNN/Basic_CNN_history.png" alt="History" width="400" />

</details>

---

### Dataset 2 Evaluation
Dataset 2 represents a more challenging baseline with slightly more nuanced background environments.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **VGG16 Fine-Tuned** | **99.50%** | **99.90%** | **99.52%** | **99.71%** |
| **Zhang CNN** | 99.25% | **99.90%** | 99.24% | 99.57% |
| **Basic CNN** | 89.85% | 96.69% | 91.47% | 94.01% |

<p align="left">
  <img src="./Output/Comparisons/Dataset2_BarChart.png" alt="Dataset 2 Bar Chart" width="100%" />
  <br />
  <img src="./Output/Comparisons/Dataset2_Combined_ROC.png" alt="Dataset 2 ROC Curve" width="100%" />
</p>

#### Dataset 2 Confusion Matrices & Training History
<details>
<summary>Click to expand charts for Dataset 2</summary>

**VGG16 Fine-Tuned**
- <img src="./Output/Dataset2/VGG16_FineTuned/VGG16_FineTuned_cm.png" alt="CM" width="300" /> <img src="./Output/Dataset2/VGG16_FineTuned/VGG16_FineTuned_history.png" alt="History" width="400" />

**Zhang CNN**
- <img src="./Output/Dataset2/Zhang_CNN/Zhang_CNN_cm.png" alt="CM" width="300" /> <img src="./Output/Dataset2/Zhang_CNN/Zhang_CNN_history.png" alt="History" width="400" />

**Basic CNN**
- <img src="./Output/Dataset2/Basic_CNN/Basic_CNN_cm.png" alt="CM" width="300" /> <img src="./Output/Dataset2/Basic_CNN/Basic_CNN_history.png" alt="History" width="400" />

</details>

---

## 🚀 How to Run (Google Colab Setup)

1. **Upload Notebook & Structure:** 
   - Upload the `MLProject.ipynb` file directly to a new Google Colab workspace.
   - Upload the `/Data/` directory containing the zipped datasets (Dataset1 and Dataset2) to your own Google Drive. Place it precisely under a folder named `MLProject` (so the path is `/MyDrive/MLProject/Data/`).

2. **Configure GPU Runtime:**
   - In Colab, go to **Runtime** > **Change runtime type**.
   - Select **GPU** (T4 or higher) as the hardware accelerator. This is absolutely essential to benefit from the built-in Mixed-Precision optimizations.

3. **Mount Google Drive:**
   - Run the very first cell of the notebook. It will securely prompt you to authorize Colab to connect to your Google Drive (`/content/drive`). This allows persistent access to your stored datasets.

4. **Automated Pipeline Execution:**
   - **Local NVMe Extraction:** Instead of training directly from Drive (which is slow), the notebook will instantly extract the `.rar` and `.zip` data into Colab's temporary, lightning-fast SSD storage (`/content/DatasetX`).
   - **Training & Evaluation:** It automatically processes `tf.data` batches through the constructed CNNs. 
   - **Output Persistence:** All resulting artifacts—`.keras` models, `.csv` training histories, Confusion Matrices, and ROC charts—are seamlessly exported backward into your `/MyDrive/MLProject/Output/` directory.
   - **Smart Checkpointing:** If you disconnect or return later, the notebook detects previously trained models in the `/Output/` folder and strictly skips them to save time.
