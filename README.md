# Zillow’s Home Value Prediction (Capstone Project One)

## Structure

This project provides a potential solution to [Zillow's Zestimate competition](https://www.kaggle.com/c/zillow-prize-1) on Kaggle. It's one of two Capstone project requirements for the Springboard data science program. The goal of this capstone project is not to win the competition but to provide an in depth experience in learning various machine learning algorithms. In particular, Neural Networks, Decision trees and XGBoost will be explored in this project. 

The project is divided into four sections, each section is described in a corresponding Jupyter notebook. Your feedback on the notebooks is welcome!

* **[Part 1: Data Exploration and Prepocessing](Section1_Data_PreprocessingExplore.ipynb)** — we get accustomed with Zillow's dataset and do basic data exploration using tools such as histograms and boxplots.

* **[Part 2: Feature Selection](Section2_FeatureSelection.ipynb)** — we get accustomed with Zillow's dataset and do basic data analysis: we collect basic statistics, plot a correlation matrix, compare train and test distributions.

* **[Part 3: Regression and Feature Extraction](Section3_Regression.ipynb)** — In this section we run various types of regressions including Support Vector Regressions, Decision Tree Regressions and Random Forest Regressions. We will run model with and without dimensionality reduction to compare results. We will use Root Mean Square Error as a metric to compare model preformances. 

* **[Part 4: Artifical Neural Network](Section4_NeuralNetwork.py)** — we span the space with feed-forward neural networks. This section will be done with [TensorFlow](https://www.tensorflow.org/) (acting as a backend) and [Keras](https://keras.io/) (acting as a frontend). We introduce (and show by example) the concept of overfitting, use K-Fold cross-validation to compare the performance of our models, tune hyper-parameters (number of units, dropout rates, optimizers) via Hyperopt and select the best model.

* **[Part 5: XgBoost](Section5_XGBoost.py)** — we will run the latest the most popular Machine learning algorithm and compare results.



* **[Appendix A - Missing Values](Section6_AppendixA_MissingData.py)** — We analyze each feature that have missing values and designate a strategy specific to each. 

* **[Appendix B - Ordinary Least Sqaures Exploration of All Features](Section6_AppendixB_UniCont.py)** — We will take a univarte analysis of all categorical features in the dataset. 

* **[Appendix C - Univariate Exploration of all Continous Features](Section6_AppendixC_UniCat.py)** — We will take a univarte analysis of all continous features in the dataset. 

* **[Appendix D - Univariate Exploration of all Categorical Features](Section6_AppendixD_BiVarCont.py)** — We will take a univarte analysis of all categorical features in the dataset. 

* **[Appendix E - Bivariate Exploration of all Continous Features](Section6_AppendixE_BiVarCat.py)** — We will take a univarte analysis of all continous features in the dataset. 

* **[Appendix F - Bivariate Exploration of all Categorical Features](Section6_AppendixF_BiVarCat.py)** — We will take a univarte analysis of all categorical features in the dataset. 





You can also read a [Capstone Report](report.md) which summarizes the implementation as well as the methodology of the whole project without going deep into details.

## Requirements

### Dataset

The dataset consist of two files(properties_2016.csv.zip,train_2016_v2.csv.zip) and needs to be downloaded separately (~160 MB). Just unzip it in the same directory with notebooks. The dataset is available for free on [Kaggle's competition page](https://www.kaggle.com/c/zillow-prize-1/data).

### Software

This project uses the following software (if version number is omitted, latest version is recommended):


* **Python stack**: python 3.5.3, numpy, scipy, sklearn, pandas, matplotlib, h5py.
* **Neural Network**: multi-threaded xgboost should be compiled, xgboost python package is also required.
* **Deep Learning stack**: CUDA 8.0.44, cuDNN 5.1, TensorFlow 1.1.0 Keras 2.0.6


## Guide to running this project

### Option 1 - Setting up Desktop to run GPU (This project)

**Step 1. Install necessary drivers to use GPU**
The desktop is running Windows 7 with the following installs:
If you need the C++ complier, you can download it **[Appendix B](AppendiB - Histograms.ipynb)** 

* **cuda toolkit -** https://developer.nvidia.com/cuda-toolkit -  The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler and a runtime library to deploy your application.

* **Nvidia Drivers -** http://www.nvidia.com/Download/index.aspx

* **cuDNN 7 -** https://developer.nvidia.com/cudnn - cuDNN is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

**Step 2. Install DeepLearning Packages using Conda**
* **Theano -** '$ conda install -c conda-forge theano'

* **Tensorflow GPU- ** 

 "$ conda create -n tensorflow python=3.5"
 
 "$ activate tensorflow"
 
 "$ pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-win_amd64.whl"


* **Keras -** $ conda install -c conda-forge keras 

### Option 2 - Using AWS instances

**Step 1. Launch EC2 instance**
The cheaper option to run the project is to use EC2 AWS instances:

* `c4.8xlarge` CPU optimized instance for Feature Selection calculations (best for Part 2).
* `EC2 g2.2xlarge` GPU optimized instance for MLP and ensemble calculations (best for Part 3). If you run an Ireland-based spot instance, the price will be about $0.65 per hour or you can use "Spot Instances" to help reduce cost.

* http://markus.com/install-theano-on-aws/

Please make sure you run Ubuntu 14.04. For Ireland region you can use this AMI: **ami-ed82e39e**. Also, add 30 GB of EBS volume 

**Step 2. Clone this project**

`sudo apt-get install git`

`cd ~; git clone https://github.com/volak/zillow_capstone.git`

`cd zillow_capstone`
