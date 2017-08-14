# Zillow’s Home Value Prediction (Capstone Project One)

## Structure

This project provides a potential solution to [Zillow's Zestimate competition](https://www.kaggle.com/c/zillow-prize-1) on Kaggle. It's one of two Capstone project requirements for the Springboard data science program. The goal of this capstone project is not to win the competition but to provide an in depth experience in learning various machine learning algorithms. In particular, Neural Networks, Decision trees and XGBoost will be explored in this project. 

The project is divided into five sections, each section is described in a corresponding Jupyter notebook. We will use Root Mean Square Error to evaulate the different model performances. Your feedback on the notebooks is welcome!

* **[Section 1: Data Exploration and Prepocessing](Section1_Data_PreprocessingExplore.ipynb)** — we get accustomed with Zillow's dataset by quickly viewing the data structure and make necessary steps to prepare the dataset to used for the different models that will be run. A important step in the prepocessing step will be dealing with the missing data from the 52 potential features. We have designated an appendix to work through the details of how a missing data is handle. The output file from the Missing Data file will be used for section 2 through 5. 

* **[Section 2: Feature Selection](Section2_FeatureSelection.ipynb)** — For this section will try to create a model whose result could be intrepretated. Of the three main feature selection categories, Filter, Wrapper and Embedded, we will focus on the Filter Method, i.e. Correlation and Ordinary Least Squares. 

* **[Section 3: Regression and Feature Extraction](Section3_Regression.ipynb)** — In this section we run various types of regressions including Support Vector Regressions, Decision Tree Regressions and Random Forest Regressions. Due to computational constraints we will run the Regression models after doing a Dimenionality reduction using a feature extraction technique. Though we originally planned on running PCA, Linear Discrimant Analysis and Kernel PCA our computing limitations only allowed us to run a PCA analysis. 

* **[Section 4: Artifical Neural Network](Section4_NeuralNetwork.ipynb)** — We will use a standard neural network with backward propagation. In order to run this section of the notebook, we had to install [TensorFlow](https://www.tensorflow.org/) (acting as a backend) and [Keras](https://keras.io/) (acting as a frontend). We added a 10% dropout after every hiddne layer to minize the effects of overfitting. We also wanted to use a K-Fold cross validation to compare the performance of our model but due to processing power limitation we were only able to do a train test split on our dataset. It goes without saying that we were also unable to tune our model with Grid Search as well.


* **[Section 5: XgBoost](Section5_XGBoost.ipynb)** — XgBoost is one of the most popular model in machine learning. It is also the most powerful implementation of gradient boosting. One of the major advantages of Xgboost besides having high performance and fast execution speed, you can keep the interpretation of the original problem. We were also unable to do a K-fold cross validation on our boosted model.


* **[Appendix A - Missing Values](Section6_AppendixA_MissingData.py)** — We have 59 features and most have missing values that need to be taken care of. Since each feature could be vital to providing an accurate model, we made decisions on imputing the missing value on a case by case analysis. Some features we given the mean, median or mode while others had specific needs such as imputing a random value holder. There were a few feature that had no values at all and were dropped from the dataset.  

* **[Appendix B - Univariate Exploration of all Continous Features](Section6_AppendixB_UniCat.py)** — We will take a univarte analysis of all continous features in the dataset including the targetfeature. We will running histograms and check for outliers. 

* **[Appendix C - Univariate Exploration of all Categorical Features](Section6_AppendixC_BiVarCont.py)** — We will take a univarte analysis of all categorical features in the dataset. Most of the graphcs will take the form of bar plots.  

* **[Appendix D - Bivariate Exploration of all Continous Features](Section6_AppendixD_BiVarCat.py)** — We will take a univarte analysis of all continous features in the dataset. 

* **[Appendix E - Bivariate Exploration of all Categorical Features](Section6_AppendixE_BiVarCat.py)** — We will take a univarte analysis of all categorical features in the dataset. 





You can also read a [Capstone Report](report.doc) which summarizes the implementation as well as the methodology of the whole project.

## Requirements

### Dataset

The dataset consist of two files(properties_2016.csv.zip,train_2016_v2.csv.zip) and needs to be downloaded separately (~160 MB). Just unzip it in the same directory with notebooks. The dataset is available for free on [Kaggle's competition page](https://www.kaggle.com/c/zillow-prize-1/data).


### Software

This project uses the following software (if version number is omitted, latest version is recommended):


* **Python stack**: python 3.5.3, numpy, scipy, sklearn, pandas, matplotlib, h5py.
* **Neural Network**: multi-threaded xgboost should be compiled, xgboost python package is also required.
* **Deep Learning stack**: CUDA 8.0.44, cuDNN 5.1, TensorFlow 1.1.0 Keras 2.0.6


## Guide to running this project

### Option 1 - Setting up Desktop to run  Nvidia's GeForce 770 (This project)

**Step 1. Install necessary drivers to use GPU**
The desktop is running Windows 7 with the following installs:
If you need the C++ complier, you can download it **[C++ Compiler](http://landinghub.visualstudio.com/visual-cpp-build-tools)** 

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

`cd ~; git clone https://github.com/volak/Zillow.git`

`cd Zillow_capstone`
