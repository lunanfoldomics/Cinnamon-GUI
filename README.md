<img src="Cinnamon-Gui-Logo.png" alt="logo" style="display:block; margin:auto; width:200px; height:auto;">

# CINNAMON-GUI
### Convolutional Neural Network And Multimodal Learning with Graphic User Interface for Digital Pathology
## Version 0.4.0
# SPDX-License-Identifier: AGPL-3.0-only

CINNAMON-GUI is an advanced digital pathology tool that leverages the power of convolutional neural networks (CNN) and multimodal learning to provide an intuitive graphical user interface for research professionals. This tool is designed to enhance the analysis and interpretation of pathological samples, making it easier to study various diseases.

#### Developed by Lunan Foldomics LLC, Copyright (C) 2024

Disclaimer: the software use is intended ONLY for experimental purposes, not for clinical.

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

The general CINNAMON-GUI architecture can be described as following:

<img src="/Images/CNN.png" alt="logo" style="display:block; margin:auto; width:800px; height:auto;">

But it can be modified by the user depending on the type of input and according to needs.

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

## Dataset
As example of implementation we uses the SIPAKMED Dataset

Download Link: https://www.cs.uoi.gr/~marina/sipakmed.html

Once the SipakMed dataset is downloaded, it needs to be unzipped into a directory, which we might call "sipakmed." The main directory structure of SipakMed is not particularly complex, but it is essential to understand where the images are located within the five cellular categories to correctly construct the pickle file. Therefore, a script must be generated to search for images within the sipakmed directory and generate the pickle file. Here is an example of how this can be done:
Upload all the necessary Python libraries
```
# routine for converting Bmp to a Pickle Dataset

import numpy as np
from numpy import load
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from matplotlib.pyplot import imshow
from matplotlib.pyplot import figure
import pickle
import math
import subprocess
import os
import cv2
from PIL import Image
%matplotlib inline
```
Define a Working Directory using os.getcwd() and concatenate the WD path to the sipakmed directory, where all the images are stored:

```
WD = os.getcwd()+'/sipakmed/'
```

Make a list of paths:

```
'''
The cell categories are assigned to labels

Dyskeratotic 0
Koilocytotic 1
Metaplastic 2
Parabasal 3
Superficial-Intermediate 4
'''

# Creating paths

path = []
path.append(WD+'/im_Dyskeratotic/im_Dyskeratotic/CROPPED/')
path.append(WD+'/im_Koilocytotic/im_Koilocytotic/CROPPED/')
path.append(WD+'/im_Metaplastic/im_Metaplastic/CROPPED/')
path.append(WD+'/im_Parabasal/im_Parabasal/CROPPED/')
path.append(WD+'/im_Superficial-Intermediate/im_Superficial-Intermediate/CROPPED/')
```

Finally, create the dataset:
```
# Warning: the OS command for producing the index file list.txt works only with iOS and Linux systems
images_array = []
label = 0
size = 256 # We want a 256 X 256 pixel resolution.

print('I\'m creating the pickle dataset...')

for p in range(len(path)):
    command = str("rm " +path[p]+ str("sunset_resized.bmp"))
    os.system(command)    
    
    print('label: ',label)    
    command = str("ls -lth ") + path[p]+ " | " + str("awk '{print $10}' | grep bmp | sed 's/.bmp//' > " + path[p]+ str("list.txt"))
    os.system(command)
    df_list = pd.read_csv(path[p]+"list.txt", sep='\t', header=0)
    df_list.rename(columns={'list.txt': 'filenames'}, inplace=True)


    for i in range(len(df_list)):
        image = Image.open(path[p]+df_list.values[i][0]+'.bmp')
        sunset_resized = image.resize((size, size))
        sunset_resized.save(path[p]+'sunset_resized.bmp')
        images_array.append([cv2.imread(path[p]+'sunset_resized.bmp').flatten(), label])
        
    label = label+1

df = pd.DataFrame(images_array)
df.rename(columns={0: 'X', 1: 'y'}, inplace=True)
df.to_pickle(WD+str(size)+'X'+str(size)+'.pickle')

print('Dataset created!')
print('bye.')
```

Congrats! Your  256X256.pickle dataset is ready, and now you can visulize one of the images per time in this way:

```
# Visualizing dataset

import numpy as np
import matplotlib.pyplot as plt

# Specify here what image you want to see
Image_ID = 1000

with open(WD+str(size)+'X'+str(size)+'.pickle', 'rb') as handle:
    CVX = pickle.load(handle)

X = np.stack(CVX['X'])
Y = np.int64(CVX['y'])    

# Reshape the image
image_size = int(np.sqrt(X.shape[1] / 3))  # Dimensione di ciascun lato dell'immagine quadrata
X_reshaped = X.reshape(-1, image_size, image_size, 3)

# One-image visualization
plt.imshow(X_reshaped[Image_ID])
plt.axis('off')
plt.show()
```


## The classes.tsv file
The file "classes.tsv" contains pathological classes associated with the labels. It is pivotal for displaying the results.

| Class | Label |
|-------|-------|
| D     | 0     |
| K     | 1     |
| M     | 2     |
| P     | 3     |
| SI    | 4     |

The classes.tsv file associates numerical labels with more intuitive literal characters. 
There are no predefined rules for generating class names, but using only a short class name is a good practice. Using letters with at most two characters to define a label is better. Remember to start with the label with a zero value when defining the first class, as we are in a Python system, and we know that Python always starts from zero and never from one!
The classes have been abbreviated to letters to make the various output more readable. They correspond respectively to the five classes 'Diskeratotic,' 'Koilocytes,' 'Metaplastic,' 'Parabasal,' and 'Superficial-Intermediate,' as reported in the SIPAKMED dataset.
Generating your dataset.pickle and classes. tsv files is a straightforward process. Always ensure that a classes.tsv file accompanies your dataset and is located in the same directory as the pickle file. Additionally, make sure that both files share the same name. For example, if you decide to name your dataset.pickle 256X256.pickle, ensure that its accompanying classes file is named 256X256.classes.tsv


### Documentation
For detailed documentation, including usage examples and configuration, please refer to the `docs` directory included with the software or visit our [documentation page](http://www.lunanfoldomicsllc.com/documentation).

### Contributing
We welcome contributions from the community. If you are interested in contributing to CINNAMON-GUI, please read our `CONTRIBUTING.md` file for guidelines on how to get started.

## License
This project is licensed under the terms of the GNU Affero General Public License version 3. See the LICENSE file in the root directory of the repository for details.

### Contact
For support or inquiries, please contact us via email at [lucazammataro@lunanfoldomicsllc.com](mailto:lucazammataro@lunanfoldomicsllc.com) or  at [info@lunanfoldomicsllc.com](mailto:info@lunanfoldomicsllc.com) or visit the contact page on our website.

### Follow Us
Stay connected with updates and news:
- [Twitter](http://twitter.com/LunanFoldomics)
- [LinkedIn](http://linkedin.com/company/lunan-foldomics-llc)
