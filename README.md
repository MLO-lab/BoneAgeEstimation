# BoneAgeEstimation
This project estimates bone age based on spatial features extracted from multiplex images. The pipeline includes data loading, spatial transformation, probability density estimation, machine learning-based age prediction, and visualization of results.


-  [Requirements](#requirements)
-  [Workflow](#workflow)
   - [0. Libraries and Parameters](#0-libraries-and-parameters)
   - [1. Data Loading](#1-data-loading)
   - [2. Data Inspection and Preprocessing](#2-data-inspection-and-preprocessing)
   - [3. Bone Alignment and Transformation](#3-bone-alignment-and-transformation)
   - [4. Components Calculation and Estimation](#4-components-calculation-and-estimation)
   - [5. Bone Age Estimation](#5-bone-age-estimation)
   - [6. Visualization](#6-visualization)

- [Usage](#usage)

---

## Requirements

This pipeline relies on several Python packages, which can be installed automatically via the provided Conda environment file. 

### Installation

1. Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
2. Clone the repository and navigate to the directory
3. Create the Conda environment using the environment.yml file:
   ```bash
   conda env create -f environment.yml
4. Activate the environment:
   ```bash
   conda activate BoneAgeEstimation
   ```

These commands will set up the necessary environment with all required dependencies.
### Data Arrangement
To ensure proper functionality, organize your data in the following folder structure within the project's root directory:

**data_bone_age**: Organize other data files following this structure:
  - **`<condition>/`**: For the condition of interest (e.g., "3mo", "5fu30d"). One subfolder for one condition. The .ims files should be placed under the corresponding condition folder.
  - **`affinity_matrices/`**: For cluster affinity matrices.
  - The positions CSV files should be placed under the folder directly.

## Workflow

### 0. Libraries and Parameters
This section initializes the required libraries and sets user-defined parameters.

### 1. Data Loading
The first step is to import the necessary data files, which include:
- **CSV Files**: Contains quantitative data (positions).
- **IMS Image Files**: Images for bone samples for the bone space alignment and transformation.
- **Affinity Matrices**: Pre-computed matrices that represent the spatial relationships between different clusters within the bone samples.


### 2. Data Inspection and Preprocessing
After loading the data, this step visually inspects the image data and performs adjustments, such as flipping the image on the x or y axis to ensure proper orientation.


### 3. Bone Alignment and Transformation
In this step, each bone sample is aligned to a reference bone. All samples are transformed to the reference space, ensuring consistency across data.


### 4. Components Calculation and Estimation
The pipeline calculates the PDFs of the spatial distributions of HSCs, RDs, and cKits. It also computes the spatial cluster compositions for each sample. The results are saved in a structured format for further analysis.


### 5. Bone Age Estimation
This section includes the following:
- **Utility Functions**: Functions to assist in the bone age estimation process.
- **Training Data Generation**: Generates training data for linear and Gaussian models separately.
- **Weight Optimization**: Optimizes the weights for the linear model.
- **Age Prediction**: Uses the optimized models to predict the bone age based on the input data.
### 6. Visualization
This section generates various visualizations to interpret the results of the bone age estimation process. The visualizations include:
- Element-wise bone age maps
- Spatial density maps of HSCs, RDs, and cKits
- Heatmap of cKit cluster affinities
- Bone age visualizations
- SHAP visualizations for model interpretability

---

## Usage

1. **Run each section of the pipeline sequentially**, ensuring all paths and parameters are correctly set for your specific dataset.
2. **Inspect intermediate visualizations** to verify data quality and make adjustments as needed.
3. **Save output figures** and clustering data as needed for further analysis or reporting.
