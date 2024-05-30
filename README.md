<img src="Cinnamon-Gui-Logo.png" alt="logo" style="display:block; margin:auto; width:200px; height:auto;">

# CINNAMON-GUI

## Version 0.4.0

### Convolutional Neural Network And Multimodal Learning with Graphic User Interface for Digital Pathology

CINNAMON-GUI is an advanced digital pathology tool that leverages the power of convolutional neural networks (CNN) and multimodal learning to provide an intuitive graphical user interface for research professionals. This tool is designed to enhance the analysis and interpretation of pathological samples, making it easier to study various diseases.

#### Developed by Lunan Foldomics LLC, Copyright (C) 2024

For more information, visit our website: [Lunan Foldomics LLC](http://www.lunanfoldomicsllc.com/)

### Features
- **Advanced CNN Models**: Utilize state-of-the-art convolutional neural networks optimized for digital pathology.
- **Multimodal Learning Capabilities**: Integrate multiple types of data to improve analysis accuracy (only for research purpose).
- **User-Friendly Interface**: Simplify the workflow with an intuitive GUI, designed for professionals without technical expertise.
- **Scalable Solutions**: Suitable for individual researchers and large organizations.

### Installation
1. Download the latest version of CINNAMON-GUI from our [official repository](https://github.com/lunanfoldomics/Cinnamon-GUI/).
2. Follow the installation guide provided in this `README.md` document to set up the software on your system.

## Description

Cinnamon-GUI is an application that leverages the convenience of interactive widgets in Jupyter to classify Digital Pathology images. It makes it effortless to load datasets, train Convolutional Neural Network (CNN) models, and test these models on images.

The general structure of the application is as follows:

1. Tab for dataset loading: Allows loading a dataset of images in pickle format.
2. Tab for model training: This feature-rich tab equips you with controls to fine-tune model training, such as the number of epochs, batch size, and a variety of CNN architecture options. This flexibility allows you to adapt the training process to your specific needs, giving you a sense of control over your work.
3. Tab for model loading: Allows loading a previously trained CNN model.
4. Tab for model testing: This tab contains a widget for selecting an image from the dataset and displaying the model's prediction. It also allows you to display feature maps from various layers of the CNN.
5. Tab for learning plot visualization: Displays the model's training progress plot over time.

Cinnamon-GUI has undergone rigorous testing in MacOS environments for M1 processors with Jupyter Notebook, Jupyter-Lab and in a Google Colab environment. 

## Virtual Environment

To run the code, we recommend using a conda virtual environment. You can create a virtual environment named `cinnamongui` and install the required dependencies by executing the following commands:

```bash
# Create a new virtual environment named medgui-convnet
conda create -n cinnamongui python=3.9

# Activate the virtual environment
conda activate cinnamongui  # On Windows
source activate cinnamongui  # On macOS and Linux

# Install the required packages
pip install numpy==1.23.2
pip install pandas==1.5.3
pip install keras==2.11.0
pip install tensorflow-estimator==2.11.0
pip install tensorflow-macos==2.11.0
pip install tensorflow-metal==0.7.1
pip install matplotlib==3.7.1
pip install scikit-learn==1.2.2
pip install ipywidgets==8.0.4
pip install ipyfilechooser==0.6.0
pip install ipython==8.11.0
pip install labelme

# Deactivate the virtual environment
conda deactivate  # On Windows
source deactivate  # On macOS and Linux

```

## Run Cinnamon-GUI

The python module cinnamongui.py contains all the Cinnamon-GUI functions and can be imported in a Jupyter Notebook cell typying:

```
import cinnamongui
```

You can use the Cinnamon-GUI.ipynb provided in this Repository.

## Cinnamon-GUI basic functions

Cinnamon-GUI functions are directly accessible (you do not need the GUI to use them). Import the module as an object:

```
import cinnamongui as cinnamongui
```
Upload a dataset with LoadImageArchive(path):
The dataset must be a pickle file of a NumPy array (more information will be available soon about how to construct a dataset)
```
X, y = cinnamongui.LoadImageArchive(classes=['D', 'K', 'M', 'P', 'SI'], path='path_to_a_dataset.pickle')
```

Splitting Training and Testing datasets
```
X_train, X_test, Y_train, Y_test = cinnamongui.splitDataset(X=X, Y=y, test_size=0.2, random_state=42)
```

Loading a saved model
```
model = cinnamongui.loadModel(model_path='path_to_a_model_directory')
```

Check the size of all the arrays
```
# Verify the arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

```
Defining a model:
You can define the CNN architecture by adjusting filters, the number of neurons, and activation functions.
```
model = defineModel(X=X, Y=Y,
              Conv2D_1_filters=44, Conv2D_1_kernelSize=3,  C2D_1_activation='relu', MP2D_1_filters=2, 
                Conv2D_2_filters=128, Conv2D_2_kernelSize=3,  C2D_2_activation='relu', MP2D_2_filters=2, 
                Conv2D_3_filters=256, Conv2D_3_kernelSize=3,  C2D_3_activation='relu', MP2D_3_filters=2, 
                Conv2D_4_filters=512, Conv2D_4_kernelSize=3,  C2D_4_activation='relu', MP2D_4_filters=2,
                Conv2D_5_filters=512, Conv2D_5_kernelSize=3,  C2D_5_activation='relu', MP2D_5_filters=2, 
                Dense_1_filters=128, Dense_1_activation='relu', l1=0.001, l2=0.001,
                Dense_2_activation='softmax')
```
Training:
You can manipulate training epochs, batch size, and two regularization parameters to fine-tune the training performances.
```
model = cinnamongui.trainCNN(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, epochs=30, batch_size=32)
```

## Datasets 
As example of implementation we uses the SIPAKMED Dataset
SIPAKMED Dataset Download Link: https://www.cs.uoi.gr/~marina/sipakmed.html


Disclaimer: the software use is intended ONLY for experimental purposes, not for clinical.

## Additional Files
The file "classes.tsv" contains pathological classes associated with the labels. It is pivotal for displaying the results.
To complete

### Documentation
For detailed documentation, including usage examples and configuration, please refer to the `docs` directory included with the software or visit our [documentation page](http://www.lunanfoldomicsllc.com/documentation).

### Contributing
We welcome contributions from the community. If you are interested in contributing to CINNAMON-GUI, please read our `CONTRIBUTING.md` file for guidelines on how to get started.

### License
CINNAMON-GUI is licensed under the GNU General Public License v3.0. For more details, see the `LICENSE` file included with the distribution or visit [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

### Contact
For support or inquiries, please contact us via email at [lucazammataro@lunanfoldomicsllc.com](mailto:lucazammataro@lunanfoldomicsllc.com) or visit the contact page on our website.

### Follow Us
Stay connected with updates and news:
- [Twitter](http://twitter.com/LunanFoldomics)
- [LinkedIn](http://linkedin.com/company/lunan-foldomics-llc)
