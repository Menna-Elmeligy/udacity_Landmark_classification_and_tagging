# Landmark Classification & Tagging for Social Media

This project is part of the Udacity Nanodegree program [**AWS Machine Learning Fundamentals**](https://www.udacity.com/enrollment/nd189-aws-fundamentals)

## Project Overview

Photo sharing and photo storage services often rely on location data to enhance user experiences by providing features such as automatic tag suggestions and photo organization. However, many photos lack location metadata, which can occur if the camera does not have GPS or if the metadata has been removed for privacy reasons.

To address this challenge, this project aims to build a landmark classifier that can automatically predict the location of an image based on any discernible landmarks. By leveraging convolutional neural networks (CNNs), we can train models to recognize and classify these landmarks, allowing photo services to infer location data even when metadata is unavailable.

In this project, you'll find the following _key tasks_:

**Data Preprocessing:** Handling image data, including resizing, normalizing, and augmenting.
**Model Design:** Creating and training CNN models to classify images based on landmarks.
**Model Comparison:** Evaluating and comparing the accuracy of different CNN architectures.
**Deployment:** Building an app based on the best-performing CNN to predict landmark locations from new images.

## Notebooks

**1. `cnn_from_scratch.ipynb`**

This notebook involves building a Convolutional Neural Network (CNN) from scratch to classify landmarks in images. The process includes:

* Designing the architecture of the CNN.
* Training the model on a dataset of landmark images.
* Evaluating the model's performance.

**2. `transfer_learning.ipynb`**

This notebook focuses on applying transfer learning techniques to improve the accuracy and efficiency of the landmark classifier. It includes:

* Leveraging pre-trained models and fine-tuning them for the specific task.
* Comparing the results with the CNN built from scratch.
* Making use of transfer learning to achieve better performance with less training time.

## **Getting Started**

1. Download and install [**Miniconda**](https://docs.anaconda.com/miniconda/)
2. Create a new conda environment with Python 3.7.6:
```conda create --name landmark_class python=3.7.6 pytorch=1.11.0 torchvision torchaudio cudatoolkit -c pytorch```
3. Activate the environment:
   
`conda activate landmark_class`  

4. Install the required packages for the project:
   
`pip install -r requirements.txt`  

5. Test that the GPU is working (execute this only if you have a NVIDIA GPU on your machine, which Nvidia drivers properly installed)  

`python -c "import torch;print(torch.cuda.is_available())`  

This should return `True`. If it returns `False` your GPU cannot be recognized by pytorch. Test with `nvidia-smi` that your GPU is working. If it is not, check your NVIDIA drivers.  


6. Clone the repository:
   
```git clone https://github.com/Menna-Elmeligy/udacity_Landmark_classification_and_tagging.git```

7. Install and open jupyter lab:
```
pip install jupyterlab 
jupyter lab
```

