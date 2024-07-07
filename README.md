<img src="Cinnamon-Gui-Logo.png" alt="logo" style="display:block; margin:auto; width:200px; height:auto;">

# CINNAMON-GUI
### Convolutional Neural Network And Multimodal Learning with Graphic User Interface for Digital Pathology
## Version 0.4.2

CINNAMON-GUI is an advanced digital pathology tool that leverages the power of convolutional neural networks (CNN) and multimodal learning to provide an intuitive graphical user interface for research professionals. This tool is designed to enhance the analysis and interpretation of pathological samples, making it easier to study various diseases.
Based on Shiny Python, CINNAMON-GUI ensures that all images and reports are generated on-demand, minimizing disk access and enhancing performance.

### General Features
- **Advanced CNN Models**: Utilize state-of-the-art convolutional neural networks optimized for digital pathology.
- **Multimodal Learning Capabilities**: Cinnamon-Gui is predispoded for integrating multiple types of data to improve analysis accuracy (only for research purpose).
- **User-Friendly Interface**: Simplify the workflow with an intuitive GUI, designed for professionals without technical expertise.
- **Scalable Solutions**: Suitable for individual researchers and large organizations.

### Key Features

### 1. Initialization and Logging

- **Log Initialization**: The application initializes a log file (`log.txt`) at startup to record all activities and messages throughout the session.
- **Notification System**: Provides real-time notifications to users about the status and progress of various operations.

### 2. Dataset Loading and Preprocessing

- **Dataset Upload**: Users can upload a dataset in the form of a pickle file. The dataset is then loaded into memory, normalized, and reshaped for processing.
- **Class Definitions**: Users can upload a classes file (TSV format) to define the classes used in the dataset. These classes are used for labeling and classification during model training and evaluation.

### 3. Model Loading and Summary

- **Model Upload**: Users can upload a pre-trained model file. The application extracts the model path from the log and loads the model.
- **Model Summary**: The loaded model's architecture is summarized and displayed in a tabular format, providing detailed information about each layer and its parameters.

### 4. Annotation Processing and Dataset Creation

- **Labelme Integration**: The application can open Labelme for defining regions of interest (ROIs) and labeling cells in specimen images.
- **Annotation Processing**: Users can upload an image file and its corresponding annotation file. The application processes these annotations to extract cells, classify them, and create a new dataset.
- **In-Memory Image Handling**: All images are handled in memory using `BytesIO` and base64 encoding, ensuring that no intermediate files are saved to disk.

### 5. Model Training

- **Training Configuration**: Users can configure various parameters for training the model, including the seed, epochs, batch size, learning rate, and data augmentation settings.
- **Training Process**: The application trains the model using the specified configuration. Progress is displayed in real-time, including a progress bar and status messages.
- **Early Stopping and Learning Rate Reduction**: The training process includes callbacks for early stopping and reducing the learning rate based on validation loss.
- **Model and Log Saving**: The trained model and log file are automatically saved to disk upon completion of training.

### 6. Evaluation and Analysis

- **Image Display and Feature Mapping**: Users can select an image index to display the image along with its predicted class. Optionally, feature mapping plots can be generated to visualize the activations of different layers in the model.
- **Report Generation**: The application generates a detailed report of the predicted and actual class counts, along with a summary table and plot. All reports are displayed in the application without saving to disk.

### 7. User Interface

- **Dynamic UI Components**: The user interface is built using Shiny components, providing a responsive and interactive experience.
- **Image and Plot Display**: The application dynamically generates and displays images and plots using base64 encoding, ensuring efficient memory usage and performance.

### 8. Credits and About

- **Credits Section**: The application includes a credits section with information about the product, the company (Lunan Foldomics LLC), and contact details for further inquiries and customization requests.

## Detailed Workflow

### Initialization and Logging

The application initializes with the creation of a log file that records all activities, ensuring traceability and debugging ease. Real-time notifications keep users informed about the status and progress of their tasks.

### Dataset Loading and Preprocessing

Users upload a dataset in pickle format. The dataset is loaded into memory, normalized, and reshaped for further processing. Additionally, a classes file can be uploaded to define the classes used in the dataset, aiding in accurate labeling and classification.

### Model Loading and Summary

Users can upload a pre-trained model, and the application extracts the model path from the log file. The loaded model's architecture is summarized and displayed in a tabular format, providing insights into the structure and parameters of each layer.

### Annotation Processing and Dataset Creation

The application integrates with Labelme, allowing users to define regions of interest and label cells in specimen images. Uploaded image and annotation files are processed to extract cells, classify them, and create a new dataset. This process is optimized to handle images in memory, minimizing disk usage.

### Model Training

Users can configure various parameters for model training, including data augmentation settings. The application trains the model while displaying real-time progress. Early stopping and learning rate reduction techniques are employed to optimize training. Upon completion, the trained model and log file are saved to disk, and training plots are generated on-demand.

### Evaluation and Analysis

Users can select images to display along with their predicted classes. Feature mapping plots can be generated to visualize model activations. The application generates detailed reports of predicted and actual class counts, which are displayed dynamically within the application.

### User Interface

The user interface, built with Shiny components, provides a responsive and interactive experience. Images and plots are generated dynamically using base64 encoding, ensuring efficient memory usage and performance.

### Installation
1. Download the latest version of CINNAMON-GUI from our [official repository](https://github.com/lunanfoldomics/Cinnamon-GUI/).
2. Follow the installation guide provided in this `README.md` document to set up the software on your system.


## Cinnamon-GUI: Now a Shiny App in Python (July 2024)
Cinnamon-GUI has evolved from its initial implementation with ipywidgets for Jupyter Lab and Notebook to a fully-fledged Shiny app in Python. This transition brings several advantages, notably an enhanced user interface and improved interactivity. The Shiny framework allows for dynamic and responsive web applications, making it more straightforward for users to interact with machine learning models and visualize results in real time. This update ensures that Cinnamon-GUI remains at the forefront of digital pathology tools, providing a more robust and user-friendly experience for researchers and clinicians.

The general CINNAMON-GUI architecture can be described as following:

<img src="/Images/Training_4.png" alt="logo" style="display:block; margin:auto; width:800px; height:auto;">

But it can be modified by the user depending on the type of input and according to needs.

The general structure of the application is as follows:

1. Tab for dataset loading: Allows loading a dataset of images in pickle format.
2. Tab for model training: This feature-rich tab equips you with controls to fine-tune model training, such as the number of epochs, batch size, and a variety of CNN architecture options. This flexibility allows you to adapt the training process to your specific needs, giving you a sense of control over your work.
3. Tab for model loading: Allows loading a previously trained CNN model.
4. Tab for model testing: This tab contains a widget for selecting an image from the dataset and displaying the model's prediction. It also allows you to display feature maps from various layers of the CNN.
5. Tab for learning plot visualization: Displays the model's training progress plot over time.

Cinnamon-GUI has undergone rigorous testing in MacOS environments for M1 processors.

## Virtual Environment

To run the code, we recommend using a conda virtual environment. You can create a virtual environment named `cinnamongui` and install the required dependencies by executing the following commands:

```
# Create a new virtual environment cinnamongui
conda create -n cinnamongui python=3.9
```
Activate the virtual environment

```
conda activate cinnamongui  # On Windows
source activate cinnamongui  # On macOS and Linux
```

To install the necessary packages for Cinnamon-GUI, you can create a requirements.txt file with the following content. This file lists all the dependencies that need to be installed:

```bash
shiny
os
io
re
shutil
pandas
numpy
json
tensorflow
Pillow
matplotlib
scikit-learn
datetime
pickle
base64
tempfile
asyncio
labelme
```

You can install these dependencies by running:
```bash
pip install -r requirements.txt
```

Deactivate the virtual environment
```bash
conda deactivate  # On Windows
source deactivate  # On macOS and Linux
```

## Running Cinnamon-Gui
```bash
cd dashboard-tips
python cinnamon-gui.py
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

| Class                     | Label |
|---------------------------|-------|
| Dyskeratocytes            | 0     |
| Koilocytes                | 1     |
| Metaplastic Cells         | 2     |
| Parabasal Cells           | 3     |
| Superficial-Intermediate Cells | 4     |

The classes.tsv file associates numerical labels with more intuitive literal characters. 
There are no predefined rules for generating class names, but using only a short class name is a good practice. Using letters with at most two characters to define a label is better. Remember to start with the label with a zero value when defining the first class, as we are in a Python system, and we know that Python always starts from zero and never from one!
The classes have been abbreviated to letters to make the various output more readable. They correspond respectively to the five classes 'Diskeratotic,' 'Koilocytes,' 'Metaplastic,' 'Parabasal,' and 'Superficial-Intermediate,' as reported in the SIPAKMED dataset.
Generating your dataset.pickle and classes. tsv files is a straightforward process. Always ensure that a classes.tsv file accompanies your dataset and is located in the same directory as the pickle file. Additionally, make sure that both files share the same name. For example, if you decide to name your dataset.pickle 256X256.pickle, ensure that its accompanying classes file is named 256X256.classes.tsv


### Documentation
For detailed documentation, including usage examples and configuration, please refer to the `docs` directory included with the software or visit our [documentation page](http://www.lunanfoldomicsllc.com/documentation).

### Contributing
We welcome contributions from the community. If you are interested in contributing to CINNAMON-GUI, please read our `CONTRIBUTING.md` file for guidelines on how to get started.

## License
# SPDX-License-Identifier: AGPL-3.0-only
This project is licensed under the terms of the GNU Affero General Public License version 3. See the LICENSE file in the root directory of the repository for details.

#### Developed by Lunan Foldomics LLC, Copyright (C) 2024

Disclaimer: the software use is intended ONLY for experimental purposes, not for clinical.

For more information, visit our website: [Lunan Foldomics LLC](http://www.lunanfoldomicsllc.com/)

### Contact
For support or inquiries, please contact us via email at [lucazammataro@lunanfoldomicsllc.com](mailto:lucazammataro@lunanfoldomicsllc.com) or  at [info@lunanfoldomicsllc.com](mailto:info@lunanfoldomicsllc.com) or visit the contact page on our website.

### Follow Us
Stay connected with updates and news:
- [Twitter](http://twitter.com/LunanFoldomics)
- [LinkedIn](http://linkedin.com/company/lunan-foldomics-llc)
  
