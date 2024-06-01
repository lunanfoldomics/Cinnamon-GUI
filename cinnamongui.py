#!/usr/bin/env python
# coding: utf-8

'''
CINNAMON-GUI version 0.4.0
ConvolutIonal Neural Network And multimOdal learNing with Graphic User Interface for Digital Pathology
Lunan Foldomics LLC, 2024
www.lunanfoldomicsllc.com/
GNU General Public License v3.0

'''


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import History
import ipywidgets as widgets
from ipywidgets import interact, Button, interactive, fixed, interact_manual, HBox, VBox, Label, Layout, Output, Tab, Dropdown
from ipyfilechooser import FileChooser
from IPython.display import display
import io
import os
import shutil
import re
from contextlib import redirect_stdout
from datetime import datetime
import pickle
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

import subprocess
import json
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Imposta il livello di log TensorFlow su ERROR (1) o FATAL (2)
tf.config.set_visible_devices([], 'GPU')


get_ipython().run_cell_magic('html', '', '<style>\n    .jp-Notebook {\n        display: flex;\n        justify-content: center;\n    }\n</style>\n')


get_ipython().run_cell_magic('html', '', '<style>\n    :root {\n        --jp-layout-color0: #fae9cf;\n    }\n    body, .jp-Notebook, .jp-Cell {\n        background-color: #fae9cf !important;\n    }\n</style>\n')



# Definisci una funzione per calcolare il massimo all'interno di ciascun elenco
def max_in_list(lst):
    return max(lst)


# In[107]:


def LoadImageArchive(path, classes):
    
    with open(path, 'rb') as handle:
        df_dataset = pickle.load(handle)

    # Applica la funzione alla colonna 'X' per ottenere il massimo in ogni lista
    max_values = df_dataset['X'].apply(max_in_list)

    # Trova il massimo tra tutti i valori ottenuti
    max_value = max_values.max()

    # Normalizza i valori della colonna 'X' dividendo ogni valore per il massimo trovato
    df_dataset['X_normalized'] = df_dataset['X'].apply(lambda x: [val / max_value for val in x])

    X = np.stack(df_dataset['X_normalized'])
    y = np.int64(df_dataset['y'])    

    #Y = to_categorical(y, len(set(df_dataset['y'].values)))
    Y = to_categorical(y, len(classes))    
    #X = X.reshape((X.shape[0], np.int64(np.sqrt(X.shape[1])), np.int64(np.sqrt(X.shape[1]))))
    
    # Reshape dell'immagine in una forma adatta alla visualizzazione
    image_size = int(np.sqrt(X.shape[1] / 3))  # Dimensione di ciascun lato dell'immagine quadrata
    X = X.reshape(-1, image_size, image_size, 3)

    return X, Y


# In[108]:


def splitDataset(X, Y, test_size, random_state):
    # Dividi gli array X e Y in set di training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, Y_train, Y_test


# In[109]:


def defineModel(X, Y,
                Conv2D_1_filters, Conv2D_1_kernelSize,  C2D_1_activation, MP2D_1_filters, 
                Conv2D_2_filters, Conv2D_2_kernelSize,  C2D_2_activation, MP2D_2_filters, 
                Conv2D_3_filters, Conv2D_3_kernelSize,  C2D_3_activation, MP2D_3_filters, 
                Conv2D_4_filters, Conv2D_4_kernelSize,  C2D_4_activation, MP2D_4_filters, 
                Dense_1_filters, Dense_1_activation, l1, l2,
                Dense_2_activation):
    

    # Acquire the input shape
    input_shape = X.shape[1:]
    # Acquire the numer of classes
    num_classes = Y.shape[1]

    model = models.Sequential()

    # Primo strato convoluzionale con 32 filtri 3x3 e attivazione ReLU
    model.add(layers.Conv2D(Conv2D_1_filters, (Conv2D_1_kernelSize, Conv2D_1_kernelSize), activation=C2D_1_activation, input_shape=input_shape))

    # Max pooling per ridurre le dimensioni delle feature map
    model.add(layers.MaxPooling2D((MP2D_1_filters, MP2D_1_filters)))

    # Secondo strato convoluzionale con 64 filtri 3x3 e attivazione ReLU
    model.add(layers.Conv2D(Conv2D_2_filters, (Conv2D_2_kernelSize, Conv2D_2_kernelSize), activation=C2D_2_activation))

    # Max pooling per ridurre ulteriormente le dimensioni delle feature map
    model.add(layers.MaxPooling2D((MP2D_2_filters, MP2D_2_filters)))

    # Terzo strato convoluzionale con 128 filtri 3x3 e attivazione ReLU
    model.add(layers.Conv2D(Conv2D_3_filters, (Conv2D_3_kernelSize, Conv2D_3_kernelSize), activation=C2D_3_activation))

    # Max pooling per ridurre ulteriormente le dimensioni delle feature map
    model.add(layers.MaxPooling2D((MP2D_3_filters, MP2D_3_filters)))

    # Quarto strato convoluzionale con 256 filtri 3x3 e attivazione ReLU
    model.add(layers.Conv2D(Conv2D_4_filters, (Conv2D_4_kernelSize, Conv2D_4_kernelSize), activation=C2D_4_activation))

    # Max pooling per ridurre ulteriormente le dimensioni delle feature map
    model.add(layers.MaxPooling2D((MP2D_4_filters, MP2D_4_filters)))

    # Appiattimento delle feature map
    model.add(layers.Flatten())

    # Strato denso con 256 unità e attivazione ReLU
    model.add(layers.Dense(Dense_1_filters, activation=Dense_1_activation))

    # Strato di output con attivazione softmax per la classificazione
    #model.add(layers.Dense(num_classes, activation='softmax'))
    model.add(layers.Dense(num_classes, activation=Dense_2_activation))    
    
    


    # Compilazione del modello
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    
    return model


# In[110]:


def trainCNN(X_train, X_test, Y_train, Y_test, epochs, batch_size):
    # Assicurati di avere i tuoi dati di addestramento (train_images) e le relative etichette (train_labels)
    # Assicurati di avere anche i tuoi dati di test (test_images) e le relative etichette (test_labels)

    # Addestramento del modello
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

    # Valutazione del modello
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f'Test accuracy: {test_acc}')

    # Visualizzazione delle curve di apprendimento

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    #plt.show()
    plt.close()    
    
    return model


# In[111]:


def loadModel(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model


# In[112]:


def loadClasses(classes_path):
    # Load the classes file
    df_classes = pd.read_csv(classes_path, sep='\t')
    classes = list(df_classes['class'].values) 
    return classes


# In[113]:


def saveModel(model_path):
    tf.keras.models.save_model(model, model_path)


# In[114]:


def testCNN(X, Y, model, classes, img_index, testCNN_output, dataset_filename):
    
    x = len(classes)
    def list_my_classes(x):
        return [i for i in range(x)]

    classes_list = list_my_classes(x)
    classes_keys = classes_list
    classes_values = classes
    tuple_lists = zip(classes_keys, classes_values)
    dict_classes = dict(tuple_lists)    

    img = X[img_index]

    # Reshape the image to a 1x28x28 array (as expected by the model)
    img = np.expand_dims(img, axis=0)

    # Make a prediction using the model
    logits = model.predict(img)

    # Get the predicted class (the index of the maximum logit)
    predicted_class = np.argmax(logits)

    # Print the predicted class and show the image
    with testCNN_output:    
        # Pattern regex per trovare la data nel nome del file
        pattern = r'external.cropped'

        # Trova la data nel nome del file
        match = re.search(pattern, dataset_filename)

        if match:
            plt.title("image:{} predicted:{}".format(img_index, dict_classes[predicted_class]), fontsize=12)         

        else:
            plt.title("image:{} real:{} predicted:{}".format(img_index, dict_classes[np.argmax(Y[img_index])], dict_classes[predicted_class]), fontsize=12)         

        plt.imshow(X[img_index])
        plt.axis('off')  # Rimuovi gli assi
        plt.show()
        
    return testCNN_output
        


# In[115]:


def featureMapsDisplay(feature_mapping_plot_Dropdown_index, feature_mapping_plot_output):
    import IPython.display as ip_display_2  # Rinomina il modulo per evitare conflitti di nomi
    
    i = feature_mapping_plot_Dropdown_index

    with feature_mapping_plot_output:
        
        try:
            # Tenta di aprire il file in modalità lettura
                                            
            feature_mapping_plot_filename = feature_mapping_Dir + '/Test-' + str(img_index) + '.' + str(i) + '.FeatureMapping.png'
            print_message(feature_mapping_plot_filename)
            
            with open(feature_mapping_plot_filename, "rb") as file:
                # Leggi il contenuto del file PNG
                image_data = file.read()

            # Pulisci il contenuto del widget di output
            feature_mapping_plot_output.clear_output(wait=True)

            # Visualizza l'immagine all'interno del widget di output
            with feature_mapping_plot_output:
                ip_display_2.display(ip_display_2.Image(data=image_data, format='png'))  # Cambia display in ip_display                

        except FileNotFoundError:
            print_message("Learning Plot file not found.")
        except Exception as lp:
            print_message(f"An error occurred: {lp}")

    return feature_mapping_plot_output
        


# In[151]:


'''
runGUI

'''

# THE LOG WINDOW

# Widgets
log_output_text = widgets.Textarea(description="Log:")  # Log widget
log_output_text.style = {'font-size': '10px', 'font-family': 'Arial, sans-serif'}
# Set window dimensions
log_output_text.layout.width = '1000px'
log_output_text.layout.height = '200px'


# Function for print a message using the Log winsow
def print_message(message):
    log_output_text.value += message + '\n'

    
# Save the TextArea content in an output log file
def save_to_file(change):
    text = log_output_text.value
    with open('log.txt', 'w') as file:
        file.write(text)

    
# Observe the log modifications
log_output_text.observe(save_to_file, names='value')


# INTRO
print_message('CINNAMON-GUI version 0.4.0')
print_message('')


# 1. LOADING DATASET'S TAB. Create the FileChooser widget for uploading datasets

# Widgets
dataset_file_chooser = FileChooser(filter_pattern='*.pickle', title='Select an image dataset')
dataset_file_chooser.use_dir_icons = True
dataset_file_chooser.title = 'Upload a pickle dataset'

# Create a button for start the uploading action
load_dataset_button = widgets.Button(description="Load dataset", layout={'width': '100px'}, 
                                     button_style='info', style={'button_color': '#2a98db'})
        
# Function for handling the event Load_Dataset
def on_button_click_Load_Dataset(b):
    global X, Y, classes, directory_path, dataset_filename
    
    progress_bar.layout.visibility = 'visible'


    # Define for the first time the Main Directory
    directory_path = dataset_file_chooser.selected_path   
    dataset_filename = dataset_file_chooser.selected_filename
    try:
        
        # Aggiorna la barra di progresso
        progress_bar.value += 50       

        # Load the classes file
        df_classes = pd.read_csv(directory_path+'/'+dataset_filename.replace('pickle','')+'classes.tsv', sep='\t')
        classes = list(df_classes['class'].values)        

        #display(progress_bar)        
        X, Y = LoadImageArchive(directory_path+'/'+dataset_filename, classes)

        print_message("\n")
        print_message(f"Dataset: {directory_path+'/'+dataset_filename.replace('pickle','')} successfully loaded!")
        print_message(f"Class file: {directory_path+'/'+dataset_filename.replace('pickle','')+'classes.tsv'} successfully loaded!")     
        print_message("\n")        
        print_message(f"Dataset size: {X.shape[0]} items")  
        print_message(f"Image size: {str(X.shape[1])+'X'+str(X.shape[2])+'px'}")  


        progress_bar.layout.visibility = 'hidden'

        # Finally, show model Filechooser and the button
        model_file_chooser.layout.visibility = 'visible'
        load_model_button.layout.visibility = 'visible'   
        
        

    except Exception as e:
        print_message(f"Error in loading dataset: {e}")
            

# Connect the function on_button_click to the button clic event Load_Dataset
load_dataset_button.on_click(on_button_click_Load_Dataset)


# 2a ROI (Region of Interest)

# Widgets per Labelme e caricamento delle annotazioni
labelme_button = widgets.Button(description="Run ROI",  layout={'width': '100px'}, 
                                   button_style='info', style={'button_color': '#2a98db'})
labelme_output = widgets.Output()
labelme_button.on_click(lambda b: run_labelme())


# Run Labelme
def run_labelme():
    #subprocess.run(["labelme"])
    subprocess.Popen(["labelme"])    
    print_message('Opening Labelme for definining ROIs and labelling cells of interest.')

    

# 2b. LOADING CEFA (CELL EXTRACTION FROM ANNOTATION) Create the FileChooser widget for uploading external images
    
# Widgets
cefa_file_chooser = FileChooser(filter_pattern='*.json', title='Cell Extraction From Annotation: Select an annotation file (json) created with ROI')
cefa_load_button = widgets.Button(description="Load Annotation", layout={'width': '150px'}, button_style='info', style={'button_color': '#2a98db'})
cefa_output = widgets.Output()
            
# Funzione per estrarre le cellule annotate
def extract_cells(annotations_path, image_path, output_dir, log_output):
    
    print_message(f"opening {cefa_file_chooser.selected_filename}")
    SmearFileName = (cefa_file_chooser.selected_filename.split('.json')[0])
                  
    try:
        # Crea la directory di output se non esiste
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Carica l'immagine
        image = Image.open(image_path)

        # Carica le annotazioni
        with open(annotations_path, 'r') as file:
            data = json.load(file)
        
        # Load the classes file
        df_classes = pd.read_csv(os.getcwd()+'/'+SmearFileName+'.external.cropped/external.cropped.query.256X256.classes.tsv', sep='\t')
        classes = list(df_classes['class'].values)        
        

        # Estrai le regioni annotate
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            description = shape['description']
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            min_x = int(min(x_coordinates))
            max_x = int(max(x_coordinates))
            min_y = int(min(y_coordinates))
            max_y = int(max(y_coordinates))
            
            if len(description)!=0:
                classified = list(df_classes.loc[df_classes['class']==description].label)[0]
            else:
                classified='0' # Is automatically classified as 0
            # Ritaglia l'immagine
            cropped_image = image.crop((min_x, min_y, max_x, max_y))
            output_path = os.path.join(output_dir, f'{SmearFileName}-{label}-{min_x}-{min_y}-{classified}.png')
            cropped_image.save(output_path)

        with log_output:
            #print_message(f'Saved: {output_path}')
            print_message('')
            print_message(f'Creating the Annotation dataset in "/external.cropped"')                
            print_message(f'The following cropped images have been included in the dataset:')                            
            print_message('')            
    
    except Exception as e:
        with log_output:
            print_message(f"Error during processing: {e}")


# Create the Annotation dataset
def create_dataset(image_folder, log_output, size=256):
    try:
        import os
        import glob

        filelist = os.listdir(image_folder)
        jpg_files = [f for f in filelist if f.lower().endswith('.png')]
        images_array = []

        for img_file in jpg_files:
            label = int(img_file.split('-')[4].replace('.png',''))
            image = Image.open(os.path.join(image_folder, img_file)).resize((size, size))
            images_array.append([np.array(image).flatten(), label])  # label extracted from the file
            print_message(img_file)
        
        df = pd.DataFrame(images_array, columns=['X', 'y'])
        #df.to_pickle(os.path.join(image_folder, f'external.cropped.{size}X{size}.pickle'))
        df.to_pickle(os.path.join(image_folder, f'external.cropped.query.{size}X{size}.pickle'))        
        with log_output:
            print_message('')                        
            print_message("External Cropped Dataset successfully created!")
            print_message('')            
            print_message("Now, you can upload it as a query dataset, through the Loading Dataset Tab, specifying 'Query' in the dataset name")
            print_message("Upload a trained CNN model for running predictions and pattern analyses.")
            print_message("Go to the Testing/Prediction Tab for visualizing the results.")                                                  
            print_message("You can also collect the cropped images for making a new training dataset.")
            print_message("See the CINNAMON-GUI Documentation for more info.")
            print_message('')                                    

    
    except Exception as e:
        with log_output:
            print_message(f"Error during dataset creation: {e}")

    # Trova tutti i file con estensione .jpg nella directory
    #jpg_files = glob.glob(os.path.join(os.getcwd()+'/'+'external.cropped/', '*.png'))
    # Remove all *.png file
    #for file_path in jpg_files:
    #    os.remove(file_path)
            

        
def on_button_click_CEFA(b):
    if cefa_file_chooser.selected_filename is not None:
        SmearFileName = (cefa_file_chooser.selected_filename.split('.json')[0])
        annotations_path = os.path.join(cefa_file_chooser.selected_path, cefa_file_chooser.selected_filename)
        image_path = annotations_path.replace('json', 'bmp')
        
        SmearFileName
        
        # Imposta il percorso della directory che vuoi creare
        directory_CEFA_path = os.getcwd()+'/'+SmearFileName+'.'+"external.cropped"

        # Crea la directory se non esiste già
        os.makedirs(directory_CEFA_path, exist_ok=True)

        #output_dir = os.path.join(os.getcwd(), "external.cropped")
        output_dir = directory_CEFA_path
        
        shutil.copyfile(os.getcwd()+'/'+'sys'+'/'+'external.cropped.query.256X256.classes.tsv', directory_CEFA_path+'/'+'external.cropped.query.256X256.classes.tsv')        
        
        extract_cells(annotations_path, image_path, output_dir, cefa_output)
        
        create_dataset(output_dir, cefa_output)
        

    else:
        with cefa_output:
            print_message("Please select an annotation file.")

        
# Cefa Button
cefa_load_button.on_click(on_button_click_CEFA)


# 3. TRAINING MODEL's TAB
# Widgets
# MODEL TRAINING CONTROLS
seed_widget = widgets.IntSlider(value=42, min=1, max=10000000, step=1, description='Seed:')    
seed_widget.style.handle_color = 'orange'                        
epochs_widget = widgets.IntSlider(value=30, min=1, max=100, step=1, description='Epochs:')
epochs_widget.style.handle_color = 'orange'      

test_size_widget = widgets.FloatSlider(value=0.2, min=0.1, max=1.0, step=0.1, description='Test Size:')
test_size_widget.style.handle_color = 'orange'                                    
batch_size_widget = widgets.IntSlider(value=32, min=1, max=128, step=1, description='Batch Size:')
batch_size_widget.style.handle_color = 'orange'                                

Conv2D_filter_options =['2','4','8','16','32','64','128','256','512']

C2D_1_f = widgets.Dropdown(options=Conv2D_filter_options, value='32', description='C2D1_filt:')
C2D_1_k = widgets.IntSlider(value='3', min=1, max=10, step=1, description='C2D1_Ker:')
C2D_1_k.style.handle_color = 'lightblue'        
C2D_1_a = widgets.Dropdown(options=['relu', 'softmax'], value='relu', description='C2D1_act:')
MP2D_1 = widgets.IntSlider(value='2', min=1, max=10, step=1, description='MP2D1:')  
MP2D_1.style.handle_color = 'lightblue'            


C2D_2_f = widgets.Dropdown(options=Conv2D_filter_options, value='64', description='C2D2_filt:')
C2D_2_k = widgets.IntSlider(value='3', min=1, max=10, step=1, description='C2D2_Ker:')    
C2D_2_k.style.handle_color = 'lightblue'            
C2D_2_a = widgets.Dropdown(options=['relu', 'softmax'], value='relu', description='C2D2_act:')    
MP2D_2 = widgets.IntSlider(value='2', min=1, max=10, step=1, description='MP2D2:')  
MP2D_2.style.handle_color = 'lightblue'                

C2D_3_f = widgets.Dropdown(options=Conv2D_filter_options, value='128', description='C2D3_filt:')
C2D_3_k = widgets.IntSlider(value='3', min=1, max=10, step=1, description='C2D3_Ker:')    
C2D_3_k.style.handle_color = 'lightblue'            
C2D_3_a = widgets.Dropdown(options=['relu', 'softmax'], value='relu', description='C2D3_act:')    
MP2D_3 = widgets.IntSlider(value='2', min=1, max=10, step=1, description='MP2D3:')  
MP2D_3.style.handle_color = 'lightblue'                

C2D_4_f = widgets.Dropdown(options=Conv2D_filter_options, value='256', description='C2D4_filt:')
C2D_4_k = widgets.IntSlider(value='3', min=1, max=10, step=1, description='C2D4_Ker:')    
C2D_4_k.style.handle_color = 'lightblue'            
C2D_4_a = widgets.Dropdown(options=['relu', 'softmax'], value='relu', description='C2D4_act:')    
MP2D_4 = widgets.IntSlider(value='2', min=1, max=10, step=1, description='MP2D4:')  
MP2D_4.style.handle_color = 'lightblue'                

D_1_f = widgets.Dropdown(options=Conv2D_filter_options, value='256', description='Dense1_filt:')
D_1_a = widgets.Dropdown(options=['relu', 'softmax'], value='relu', description='Dense1_act:') 
l1_widget = widgets.FloatSlider(value=0.001, min=0, max=0.1, step=0.001, description='L1:')
l1_widget.style.handle_color = 'lightblue'                    
l2_widget = widgets.FloatSlider(value=0.001, min=0, max=0.1, step=0.001, description='L2:')
l2_widget.style.handle_color = 'lightblue'                        

D_2_a = widgets.Dropdown(options=['relu', 'softmax'], value='softmax', description='Dense2_act:')

learning_plot_widget = widgets.Output()  # Display Learning Plot

# Create a button for start the Training action
train_model_button = widgets.Button(description="Train Model", layout={'width': '100px'}, 
                                    button_style='info', style={'button_color': '#2a98db'})

# Mettere insieme i widget e i pulsanti all'interno di una VBox
training_controls_box = VBox([
    HBox([seed_widget, epochs_widget, batch_size_widget, test_size_widget]),
    HBox([C2D_1_f, C2D_1_k, C2D_1_a, MP2D_1]),
    HBox([C2D_2_f, C2D_2_k, C2D_2_a, MP2D_2]),
    HBox([C2D_3_f, C2D_3_k, C2D_3_a, MP2D_3]),
    HBox([C2D_4_f, C2D_4_k, C2D_4_a, MP2D_4]),
    HBox([D_1_f, D_1_a, l1_widget, l2_widget]),
    HBox([D_2_a]),
    train_model_button])

# Function for handling the event Training
def on_button_click_Train_Model(b):
    import IPython.display as ip_display  # Rinomina il modulo per evitare conflitti di nomi
    global model, X_train, X_test, Y_train, Y_test, display_output


    try:
        # Suddividere il dataset in set di training e di test
        X_train, X_test, Y_train, Y_test = splitDataset(X=X, Y=Y, test_size=test_size_widget.value, random_state=seed_widget.value)

        #print_message(f"Seed: {seed_widget.value}")

        # Verifica le nuove forme degli array
        print_message("\n")
        print_message(f"Training and testing sets have been randomly split based on the selected seed: {seed_widget.value}")
        print_message("\n")
        print_message("Training set:")
        print_message(f"X_train shape: {X_train.shape}")
        print_message(f"Y_train shape: {Y_train.shape}")
        print_message("\n")        
        print_message("Testing set:")
        print_message(f"X_test shape: {X_test.shape}")
        print_message(f"Y_test shape: {Y_test.shape}")
        print_message("\n")            

        # Define a model
        model = defineModel(X=X, Y=Y, 
                            Conv2D_1_filters=int(C2D_1_f.value), Conv2D_1_kernelSize=int(C2D_1_k.value), C2D_1_activation=C2D_1_a.value, MP2D_1_filters=int(MP2D_1.value),
                            Conv2D_2_filters=int(C2D_2_f.value), Conv2D_2_kernelSize=int(C2D_2_k.value), C2D_2_activation=C2D_1_a.value, MP2D_2_filters=int(MP2D_2.value), 
                            Conv2D_3_filters=int(C2D_3_f.value), Conv2D_3_kernelSize=int(C2D_3_k.value), C2D_3_activation=C2D_1_a.value, MP2D_3_filters=int(MP2D_3.value),
                            Conv2D_4_filters=int(C2D_4_f.value), Conv2D_4_kernelSize=int(C2D_4_k.value), C2D_4_activation=C2D_1_a.value, MP2D_4_filters=int(MP2D_4.value), 
                            Dense_1_filters=int(D_1_f.value), Dense_1_activation=D_1_a.value, l1=l1_widget.value, l2=l2_widget.value,
                            Dense_2_activation=D_2_a.value)

        # Print Model Summary
        print_message("Model Summary:")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.summary()
        summary_text = buffer.getvalue()
        print_message(summary_text)

        print_message("Training...")    

        history = History()  # Crea un oggetto History per registrare la storia dell'addestramento
        
        progress_bar.layout.visibility = 'visible'
        progress_bar.description = 'Training...'
        progress_bar.style={'bar_color': 'orange'}
        progress_bar.value = 0
        progress_bar.max = epochs_widget.value
        
    
        for _ in range(epochs_widget.value):
            # Addestramento del modello per una singola epoca con il callback History
            model.fit(X_train, Y_train, epochs=1, batch_size=batch_size_widget.value, validation_data=(X_test, Y_test), verbose=0, callbacks=[history])

            #Update the progress bar
            progress_bar.value += 1    
            
        progress_bar.description = 'Complete.'    
        progress_bar.style={'bar_color': 'lightgreen'}
    

        # Valutazione del modello
        test_loss, test_acc = model.evaluate(X_test, Y_test)
        print_message(f'Test accuracy: {test_acc}')

        '''
        # Making Plots and Saving files
        # Files\ names embody date: get date and current hour.
        now = datetime.now()
        # Formattare la data e l'ora come stringa nel formato desiderato
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        '''

        num = seed_widget.value
        #print(num)

        # File names embody date: get date and current hour.
        now = datetime.now()

        # Formattare la data e l'ora come stringa nel formato desiderato
        timestamp = now.strftime("%Y.%m.%d-%H.%M.%S")

        # Files path
        #WD = os.getcwd()+'/'+timestamp+'-'+str(num)        
        

        # Plot History

        # Estract learning metrics and validation from the history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epochs = range(1, len(train_loss) + 1)

        # Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, 'orange', label='Training loss')
        plt.plot(epochs, val_loss, 'red', label='Validation loss')
        plt.plot(epochs, train_accuracy, 'blue', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'green', label='Validation accuracy')
        plt.title(f'{timestamp} - Seed: {seed_widget.value} - TL:{np.round(train_loss[-1],3)}/VL:{np.round(val_loss[-1],3)} - TA:{np.round(train_accuracy[-1],3)}/VA:{np.round(val_accuracy[-1],3)}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.savefig(directory_path+'/'+dataset_filename.replace('pickle','')+timestamp+'.learning_plot.png')  
        plt.close()
        
        print_message("\n")
        print_message("Model successfully trained!")

        # Automatically save the trained model to disk, with date.
        saveModel(directory_path+'/Models/'+dataset_filename.replace('pickle','')+timestamp+'.model')

        print_message("\n")
        print_message(f"Model saved in: {directory_path+'/Models/'+dataset_filename.replace('pickle','')+timestamp+'.model'}")

        train_df = pd.DataFrame([list(X_train.reshape(X_train.shape[0], -1)), list(np.argmax(Y_train, axis=1))]).T
        test_df = pd.DataFrame([list(X_test.reshape(X_test.shape[0], -1)), list(np.argmax(Y_test, axis=1))]).T
        train_df.columns = ['X', 'y']
        test_df.columns = ['X', 'y']

        # Save the training and testing DataFrame in separated pickle files
        #train_df.to_pickle(directory_path+'/'+filename.replace('pickle','')+timestamp+'.training.dataset.pickle')
        #shutil.copyfile(directory_path+'/'+filename.replace('pickle','')+'classes.tsv', directory_path+'/'+filename.replace('pickle','')+timestamp+'.training.dataset.classes.tsv')
        test_df.to_pickle(directory_path+'/'+dataset_filename.replace('pickle','')+timestamp+'.testing.dataset.pickle')
        shutil.copyfile(directory_path+'/'+dataset_filename.replace('pickle','')+'classes.tsv', directory_path+'/'+dataset_filename.replace('pickle','')+timestamp+'.testing.dataset.classes.tsv')        

        print_message(f"Training and testing datasets saved in: {directory_path}")


    except Exception as e:
        print_message(f"Error during model training: {e}")
        
        
    try:
        # Open the learning plot after training.
        learning_plot_filename = directory_path+'/'+dataset_filename.replace('pickle','')+timestamp+'.learning_plot.png'
        print_message(learning_plot_filename)
        with open(learning_plot_filename, "rb") as file:

            # Leggi il contenuto del file PNG
            image_data = file.read()

        # Pulisci il contenuto del widget di output
        learning_plot_widget.clear_output(wait=True)

        # Visualizza l'immagine all'interno del widget di output
        with learning_plot_widget:
            ip_display.display(ip_display.Image(data=image_data, format='png'))  # Cambia display in ip_display                

    except FileNotFoundError:
        print_message("Learning Plot file not found.")
    except Exception as lp:
        print_message(f"An error occurred: {lp}")
        
        

# Connect the function on_button_click to the button clic event Training
train_model_button.on_click(on_button_click_Train_Model)



# 4. LOADING MODEL's TAB

# Widgets
model_file_chooser = FileChooser(title='Select a saved CNN model')
model_file_chooser.use_dir_icons = True
model_file_chooser.title = 'Upload a model'
learning_plot_widget = widgets.Output()  # Display Learning Plot
PredictionTestPlot_widget = widgets.Output(layout=Layout(width='1000px', height='500px', overflow='hidden'))
#PredictionTestPlot_widget = widgets.Output()
# Create a button for start the uploading action
load_model_button = widgets.Button(description="Load the model", layout={'width': '200px'}, 
                                   button_style='info', style={'button_color': '#2a98db'})

# Function for handling the event Load_Model
def on_button_click_Load_Model(b):
    
    import IPython.display as ip_display  # Rinomina il modulo per evitare conflitti di nomi
    global model, directory_model_path   
    directory_model_path = model_file_chooser.selected_path


    try:
        model = loadModel(directory_model_path)  
        print_message("\n")        
        print_message(f"Model: {directory_model_path} successfully loaded!")
        
        # Print Model Summary
        print_message("Model Summary:")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.summary()
        summary_text = buffer.getvalue()
        print_message(summary_text)

        # Here, testCNN slider max assumes the value of the X length
        testCNN_slider.max=X.shape[0]-1
        img_options = list(range(X.shape[0]-1))
        # Finally, show the testCNN_slider_widget and the testCNN_button
        testCNN_slider.layout.visibility = 'visible'
        testCNN_button.layout.visibility = 'visible'   
        feature_mapping.layout.visibility = 'visible'
        feature_mapping_plot_Dropdown.layout.visibility = 'visible'
        feature_mapping_plot_button.layout.visibility = 'visible'

        # Pattern regex per trovare la data nel nome del file
        pattern = r'\d{4}.\d{2}.\d{2}-\d{2}.\d{2}.\d{2}'

        # Trova la data nel nome del file
        match = re.search(pattern, directory_model_path)

        if match:
            # Estrai la data dalla corrispondenza
            date_str = match.group(0)

    except Exception as e:
        print_message("\n")        
        print_message(f"Error in loading model: {e}")
        
    # Pattern regex per trovare la data nel nome del file
    pattern = r'external.cropped'

    # Trova la data nel nome del file
    match = re.search(pattern, dataset_filename)

    if not match:
    # Open the Learning Plot

        try:
            corrected_filename = dataset_filename.replace('pickle','').replace('testing.dataset','').replace('training.dataset','').replace(date_str,'').replace('...','')
            corrected_filename = directory_path+'/'+corrected_filename+'.'+date_str+'.learning_plot.png'

            if os.path.exists(corrected_filename):
                with open(corrected_filename, "rb") as file:

                    # Read the png
                    image_data = file.read()

                # Clean the output widget
                learning_plot_widget.clear_output(wait=True)

                # Show the plot inside the output widget
                with learning_plot_widget:
                    ip_display.display(ip_display.Image(data=image_data, format='png'))  # Change display in ip_display                
            else:
                print_message(f"File not found: {corrected_filename}")
                # Gestione dell'errore, ad esempio mostrando un messaggio all'utente

        except Exception as e:
            print_message(f"An error occurred: {e}")
        
        
    
    # Make the Prediction Test Plot (The Top 100)
    # A Prediction Test Plot will be generated automatically

    # F1 score calculation (it works ONLY after you did the training.)
    from sklearn.metrics import f1_score
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_classes =to_categorical(y_pred_classes, len(classes))
    
    #f1 = f1_score(Y_test, y_pred_classes, average='weighted')
    
    x = len(classes)
    def list_my_classes(x):
        return [i for i in range(x)]

    classes_list = list_my_classes(x)
    classes_keys = classes_list
    classes_values = classes
    tuple_lists = zip(classes_keys, classes_values)
    dict_classes = dict(tuple_lists)    
    
    
    # Check the size of the current dataset, for a better visibility of the plot
    if X.shape[0] < 100:
        rows=np.int64(np.round(np.sqrt(X.shape[0])))
        cols=np.int64(np.round(np.sqrt(X.shape[0])))        

    if X.shape[0] > 100:
        rows = 10
        cols = 10
        
    fig, axes = plt.subplots(rows, cols, figsize=(30, 30))

    # Calculatr font size based on the number of subplots
    #font_size = max(5, 20 - (rows * cols))  # Decrease font size 
    font_size = 24 + max(0, 8 - rows * cols)  # Increase font size
    
    for i in range(rows):
        for j in range(cols):
            axes[i, j].axis('off')  # Hide axes                    
            idx = i * cols + j  # Calculate the image's index in the list
            if idx < len(X):
                axes[i, j].imshow(X[idx])
                real_output = dict_classes[np.argmax(Y, axis=1)[idx]]
                prediction = dict_classes[np.argmax(y_pred, axis=1)[idx]]  # Get the prediction of the current image
                
                # Pattern regex for checking if the dataset is an external.cropped query
                # If it's a query, only the prediction will be displayed, (the real labe will is hide)
                # since is supposed that this dataset is not provided by any annotation
                pattern = r'query'

                # Find the pattern in the dataset filename
                match = re.search(pattern, dataset_filename)

                if match:
                    # if it is an external.cropped run the prediction and hide the zero label
                    axes[i, j].set_title("i:{} p:{}".format(idx, prediction), fontsize=font_size)  # Aggiungi il titolo desiderato

                else:
                    # If it is a testing dataset show real and predicted outputs
                    axes[i, j].set_title("i:{} r:{} p:{}".format(idx, real_output, prediction), fontsize=font_size)  # Aggiungi il titolo desiderato
                
            else:
                axes[i, j].axis('off')                  
            

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Aggiungi spaziatura tra le immagini
    print_message('Top predictions plot has been generated in:')   
    print_message(directory_model_path +'/'+'PredictionTestPlot.png')       
    plt.savefig(directory_model_path +'/'+'PredictionTestPlot.png', bbox_inches='tight')                                            
    #plt.savefig(directory_model_path +'/'+'PredictionTestPlot.png')                                                
    plt.close()
        
    

    # Open the Prediction Test  Plot (The Top 100)
    try:
        corrected_filename = directory_model_path +'/'+'PredictionTestPlot.png'
        with open(corrected_filename, "rb") as file:

            # Read the png
            image_data = file.read()

        # Clean the output widget
        PredictionTestPlot_widget.clear_output(wait=True)

        # Show the plot inside the widget
        with PredictionTestPlot_widget:
            ip_display.display(ip_display.Image(data=image_data, format='png'))  # Change display in ip_display                

    # Orginal dataset case: Open the Learning Plot using special tricks for correctly reading the file name
    except FileNotFoundError:
        print_message("Top 100 file not found.")
        
# Connect the function on_button_click to the button clic event Load_Model
load_model_button.on_click(on_button_click_Load_Model)





# 5. MODEL TESTING TAB. Create the testCNN widget

# 5.a  TestCNN

# Widgets
X = np.array([[0],[0],[0]])
testCNN_output = widgets.Output(layout=Layout(width='500px', height='500px', overflow='hidden'))
testCNN_slider = widgets.IntSlider(value=0, min=0, max=len(X)-1, description='Image ID:', continuous_update=False)
testCNN_button = widgets.Button(description='OK',layout={'width': '50px'}, 
                                   button_style='info', style={'button_color': '#2a98db'} )



# Function for updating the testCNN function output
def update_testCNN_output(_):
    
    global img_index, feature_mapping_Dir
    testCNN_output.clear_output()
    with testCNN_output:
        
        testCNN(X, Y, model, classes, testCNN_slider.value, testCNN_output, dataset_filename)

        if feature_mapping.value:
            
            # Pattern regex per trovare la data nel nome del file
            pattern = r'\d{4}.\d{2}.\d{2}-\d{2}.\d{2}.\d{2}'


            # Trova la data nel nome del file
            match = re.search(pattern, directory_model_path)

            if match:
                # Estrai la data dalla corrispondenza
                date_str = match.group(0)
                
            img_index = testCNN_slider.value                
            feature_mapping_Dir = directory_model_path + '/Test-' + str(img_index)       
            print_message(feature_mapping_Dir)
            if not os.path.exists(feature_mapping_Dir):
                # Se non esiste, creala
                os.makedirs(feature_mapping_Dir)

            # Prendi l'immagine dal vettore X_train
            image_array = X[img_index]

            # Espandi le dimensioni dell'immagine per adattarle al formato dell'input del modello
            image_array = np.expand_dims(image_array, axis=0)

            # Esegui l'inferenza sul modello per ottenere le attivazioni degli strati intermedi
            layer_outputs = [layer.output for layer in model.layers[1:]]  # Escludi il layer di input
            activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
            activations = activation_model.predict(image_array)

            # Visualization of activations of intermediate layers
            for i, activation in enumerate(activations):
                #print("Layer {}: {}".format(i, model.layers[i+1].name))  # Print the name of the layer
                LayerName = "Layer {}: {}".format(i+1, model.layers[i+1].name).replace('/','')
                print_message(f'Producing map: {LayerName}')

                # Handle different layer types
                if len(activation.shape) == 4:  # Conv2D layer
                    num_filters = activation.shape[3]
                    rows = int(np.ceil(np.sqrt(num_filters)))
                    cols = int(np.ceil(num_filters / rows))
                    plt.figure(figsize=(8, 8))
                    plt.title('Test-' + str(img_index) + ' ' + LayerName, fontsize=16)                    
                    for j in range(num_filters):
                        if j < num_filters:
                            plt.subplot(rows, cols, j+1)
                            plt.imshow(activation[0, :, :, j], cmap='viridis')
                            plt.xticks([])
                            plt.yticks([])
                   
                    plt.savefig(feature_mapping_Dir + '/Test-' + str(img_index) + '.' + str(i) + '.FeatureMapping.png', bbox_inches='tight')                    
                    plt.close()                    

                elif len(activation.shape) == 2:  # Dense layer
                    num_neurons = activation.shape[1]
                    plt.figure(figsize=(8, 8))
                    plt.title('Dense Layer Activations: '+'Test-' + str(img_index) + ' ' +  LayerName, fontsize=12)                          
                    plt.plot(activation.flatten())  # Plot activations of all neurons
                    plt.xlabel('Neuron Index')
                    plt.ylabel('Activation Value')
                    plt.savefig(feature_mapping_Dir + '/Test-' + str(img_index) + '.' + str(i) + '.FeatureMapping.png', bbox_inches='tight')                                        
                    plt.close()                    

                elif len(activation.shape) == 1:  # Flatten layer
                    num_units = activation.shape[0]
                    plt.figure(figsize=(8, 8))
                    plt.title('Flatten Layer Activations: '+'Test-' + str(img_index) + ' ' +  LayerName, fontsize=12)                          
                    plt.plot(activation)  # Plot activations of all units
                    plt.xlabel('Unit Index')
                    plt.ylabel('Activation Value')
                    plt.savefig(feature_mapping_Dir + '/Test-' + str(img_index) + '.' + str(i) + '.FeatureMapping.png', bbox_inches='tight')                                        
                    plt.close()
        


# Connect the button to the click event
testCNN_button.on_click(update_testCNN_output)

 
# 5.b FEATURE MAPS DISPLAY

# Widgets
map_options = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#map_options = ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7', 'Layer 8', 'Layer 9']
feature_mapping = widgets.Checkbox(description='Feature Mapping') 
feature_mapping_plot_output = widgets.Output(layout=Layout(width='500px', height='500px', overflow='hidden'))  # Display Feature Maps
feature_mapping_plot_Dropdown = widgets.Dropdown(options=map_options, value='0', description='map')
feature_mapping_plot_button = widgets.Button(description='Map',layout={'width': '100px'}, 
                                   button_style='info', style={'button_color': '#2a98db'} )

# Function for updating the Feature Mapping function output
def update_featureMapping_output(_):
    feature_mapping_plot_output.clear_output()
    
    with feature_mapping_plot_output:
        feature_mapping_plot_index = feature_mapping_plot_Dropdown.value
        #print_message(f"feature_mapping_plot_Dropdown.value {feature_mapping_plot_Dropdown.value}")
        #print_message(f"testCNN_slider.value {testCNN_slider.value}")        
        featureMapsDisplay(feature_mapping_plot_index, feature_mapping_plot_output)

# Connect the button to the click event
feature_mapping_plot_button.on_click(update_featureMapping_output)
    

# 6 ABOUT TAB

name = 'CINNAMON-GUI version 0.4.0' 
description = 'ConvolutIonal Neural Network And multimOdal learNing with Graphic User Interface'
author = 'Lunan Foldomics LLC, 2024'
website = 'www.lunanfoldomicsllc.com/'
license = 'GNU General Public License v3.0'


# Crea la tabella con i tuoi dati
table_html = f"""
<table>
  <tr>
    <th colspan="5" style="text-align:left;">{name}</th>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>

  <tr>
    <th colspan="5" style="text-align:left;">{description}</th>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>

  <tr>
    <th colspan="5" style="text-align:left;">{author}</th>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>

  <tr>
    <th colspan="5" style="text-align:left;">{website}</th>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>

  <tr>
    <th colspan="5" style="text-align:left;">{license}</th>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>


</table>
"""

# Make HTML widget with the table
# Widgets

# Create the widget FileChooser Selecting the Logo
logo_file_path = os.getcwd()+"/Images/Cinnamon-Gui-Logo-3.png"  # Imposta il percorso del file del logo
with open(logo_file_path, "rb") as file:
    logo_data = file.read()
logo_widget_cell = widgets.Image(value=logo_data, format='png', layout={'width': '200px', 'height': '200px'})

# Create the widget FileChooser Selecting the Logo
logo_file_path = os.getcwd()+"/Images/Cinnamon-Gui-Logo.png"  # Imposta il percorso del file del logo
with open(logo_file_path, "rb") as file:
    logo_data = file.read()
logo_widget = widgets.Image(value=logo_data, format='png', layout={'width': '250px', 'height': '200px'})

# about widget
about_table_widget = VBox([
    HBox([logo_widget, widgets.HTML(value=table_html)])
])    
    

    
# 7. CREATE THE TABS

# Make a separation line
separator = widgets.HTML(value='<hr style="border: 1px solid #f0f0f0;">')
#html_widget = widgets.HTML(value=html_content)


# The testCNN slider and button are initially disabled
testCNN_slider.layout.visibility = 'hidden'
testCNN_button.layout.visibility = 'hidden'
feature_mapping.layout.visibility = 'hidden'

feature_mapping_plot_Dropdown.layout.visibility = 'hidden'
feature_mapping_plot_button.layout.visibility = 'hidden'

# Creare una barra di progresso
progress_bar = widgets.IntProgress(value=0,min=0,max=100,description='Loading...',style={'bar_color': 'lightblue'})
progress_bar.layout.visibility = 'hidden'

# The model filechooser and button are initially disabled (does not work)
#model_file_chooser.layout.visibility = 'hidden'
#load_model_button.layout.visibility = 'hidden'   


tab_contents = ['Dataset Loading','ROI', 'CEFA', 'Model Training','Model Loading','Testing/Prediction','Learning Plot','About']


children = [widgets.VBox([widgets.HBox([dataset_file_chooser, load_dataset_button]), separator, log_output_text, separator, progress_bar]), 
            widgets.VBox([widgets.HBox([labelme_button, labelme_output]), separator, log_output_text,separator, progress_bar]),            
            widgets.VBox([widgets.HBox([cefa_file_chooser, cefa_load_button]), separator, log_output_text,separator, progress_bar]),                        
            widgets.VBox([training_controls_box, separator, log_output_text, separator, progress_bar]),
            widgets.VBox([widgets.HBox([model_file_chooser, load_model_button]), separator, log_output_text, separator, progress_bar]),
            
            widgets.VBox([widgets.HBox(
                [widgets.VBox([widgets.HBox([testCNN_slider, testCNN_button]), separator, testCNN_output,feature_mapping]), 
                 widgets.VBox([widgets.HBox([feature_mapping_plot_Dropdown, 
                                             feature_mapping_plot_button]),
                               separator, feature_mapping_plot_output])]), separator, PredictionTestPlot_widget, separator, log_output_text,separator, progress_bar]), 
            widgets.VBox([learning_plot_widget, separator, log_output_text, separator, progress_bar]),
            widgets.VBox([about_table_widget, separator, log_output_text, separator, progress_bar])            
           ]


tab = widgets.Tab()
tab.children = children
for i in range(len(tab_contents)):
    tab.set_title(i, tab_contents[i])
    

display(logo_widget_cell,tab)


