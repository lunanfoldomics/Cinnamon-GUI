'''
CINNAMON-GUI
Convolutional Neural Network And Multimodal Learning with Graphic User Interface for Digital Pathology
Lunan Foldomics, Copyright(c) 2024
'''

import subprocess
import os
import io
import re
import shutil
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import History
from datetime import datetime
import pickle
import base64
from io import BytesIO
from contextlib import redirect_stdout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import plotly.express as px
from shinywidgets import output_widget, render_widget
from shiny import App, reactive, render, req, ui

# Declaration of global variables as reactive
model_reactive = reactive.Value(None)
test_maxvalue = reactive.Value(0)
X = None
Y = None
classes = None
df_classes = reactive.Value(None)
dataset_filename = None
model_number = None
model_path = reactive.Value(None)
specimen_name = None
model = reactive.Value(None)
version = "0.4.3"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

# Utility function
def max_in_list(lst):
    return max(lst)

# Function to initialize the log
def initialize_log():
    with open("log.txt", "w") as log_file:
        log_file.write(f"{datetime.now()}: Log initialized\n")

# Function to show notifications 
def show_notification(message, duration=5):
    ui.notification_show(message, duration=duration)

# Function to write to the log
def print_log(message):
    with open("log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")


# Function for plotting the training history
def plot_history(history, save_path, model_number, seed):
    plt.figure(figsize=(12, 8))

    # Final values of accuracy and loss
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    # Accuracy subplot
    plt.subplot(2, 1, 1)
    plt.title(f'Model ID: {model_number} - Seed: {seed} - Training Accuracy:{np.round(final_train_accuracy, 3)} / Validation Accuracy:{np.round(final_val_accuracy, 3)}')
    plt.plot(history.history['accuracy'], 'blue', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'green', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss subplot
    plt.subplot(2, 1, 2)
    plt.title(f'Training Loss:{np.round(final_train_loss, 3)} / Validation Loss:{np.round(final_val_loss, 3)}')
    plt.plot(history.history['loss'], 'orange', label='Training Loss')
    plt.plot(history.history['val_loss'], 'red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Save the image to disk
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(img_str))

    plt.close()
    return img_str    

def define_model(input, X, Y):
    optimizer = input.optimizer()
    loss = input.loss()
    learning_rate = input.learning_rate()

    C2D_1_f = input.C2D_1_f()
    C2D_1_k = input.C2D_1_k()
    C2D_1_a = input.C2D_1_a()
    MP2D_1 = input.MP2D_1()
    C2D_2_f = input.C2D_2_f()
    C2D_2_k = input.C2D_2_k()
    C2D_2_a = input.C2D_2_a()
    MP2D_2 = input.MP2D_2()
    C2D_3_f = input.C2D_3_f()
    C2D_3_k = input.C2D_3_k()
    C2D_3_a = input.C2D_3_a()
    MP2D_3 = input.MP2D_3()
    C2D_4_f = input.C2D_4_f()
    C2D_4_k = input.C2D_4_k()
    C2D_4_a = input.C2D_4_a()
    MP2D_4 = input.MP2D_4()
    D_1_f = input.D_1_f()
    D_1_a = input.D_1_a()
    l1 = input.l1()
    l2 = input.l2()
    dropout_value = input.dropout_value()
    D_2_a = input.D_2_a()

    input_shape = X.shape[1:]
    num_classes = Y.shape[1]

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(int(C2D_1_f), (C2D_1_k, C2D_1_k), activation=C2D_1_a, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((MP2D_1, MP2D_1)))
    model.add(tf.keras.layers.Conv2D(int(C2D_2_f), (C2D_2_k, C2D_2_k), activation=C2D_2_a))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((MP2D_2, MP2D_2)))
    model.add(tf.keras.layers.Conv2D(int(C2D_3_f), (C2D_3_k, C2D_3_k), activation=C2D_3_a))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((MP2D_3, MP2D_3)))
    model.add(tf.keras.layers.Conv2D(int(C2D_4_f), (C2D_4_k, C2D_4_k), activation=C2D_4_a))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((MP2D_4, MP2D_4)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(int(D_1_f), activation=D_1_a, kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
    model.add(Dropout(dropout_value))
    model.add(tf.keras.layers.Dense(num_classes, activation=D_2_a))

    # Model compilation
    if optimizer == 'SGD':
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    if optimizer == 'Adam':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    if optimizer == 'RMSprop':
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    if optimizer == 'Adagrad':
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    if optimizer == 'Adadelta':
        model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    if optimizer == 'Adamax':
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    if optimizer == 'Nadam':
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])    
    return model

# Function to test the CNN model and generate feature maps
def testCNN(X, Y, model, classes, img_index, query, feature_mapping):
    x = len(classes)

    def list_my_classes(x):
        return [i for i in range(x)]

    classes_list = list_my_classes(x)
    classes_keys = classes_list
    classes_values = classes
    tuple_lists = zip(classes_keys, classes_values)
    dict_classes = dict(tuple_lists)

    img = X[img_index]

    img = np.expand_dims(img, axis=0)

    logits = model.predict(img)

    predicted_class = np.argmax(logits)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(X[img_index])

    if query == 'yes':
        ax.set_title(f"Dataset: {specimen_name} - Image: {img_index} - Predicted: {dict_classes[predicted_class]}", fontsize=12)
        ax.axis('off')

    if query == 'no':
        ax.set_title(f"Dataset: {specimen_name} - Image: {img_index} - Real: {dict_classes[np.argmax(Y[img_index])]} - Predicted: {dict_classes[predicted_class]}", fontsize=12)
        ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    combined_image = Image.open(io.BytesIO(base64.b64decode(img_str)))

    if feature_mapping:
        image_array = X[img_index]
        image_array = np.expand_dims(image_array, axis=0)

        layer_outputs = [layer.output for layer in model.layers[1:]]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(image_array)

        feature_map_images = []

        for i, activation in enumerate(activations):
            LayerName = "Layer {}: {}".format(i + 1, model.layers[i + 1].name).replace('/', '')
            print('Producing map:', LayerName)
            show_notification(f'Producing map: {LayerName}')

            if len(activation.shape) == 4:  # Conv2D layer
                num_filters = activation.shape[3]
                rows = int(np.ceil(np.sqrt(num_filters)))
                cols = int(np.ceil(num_filters / rows))
                fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
                fig.suptitle('Test-' + str(img_index) + ' ' + LayerName, fontsize=20)
                for j in range(num_filters):
                    ax = axs[j // cols, j % cols]
                    ax.imshow(activation[0, :, :, j], cmap='viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])

                for j in range(num_filters, rows * cols):
                    fig.delaxes(axs[j // cols, j % cols])

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                feature_map_images.append(Image.open(buf))
                plt.close()

            elif len(activation.shape) == 2:  # Dense layer
                num_neurons = activation.shape[1]
                plt.figure(figsize=(12, 12))
                plt.title('Dense Layer Activations: ' + 'Test-' + str(img_index) + ' ' + LayerName, fontsize=20)
                plt.plot(activation.flatten())
                plt.xlabel('Neuron Index')
                plt.ylabel('Activation Value')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                feature_map_images.append(Image.open(buf))
                plt.close()

            elif len(activation.shape) == 1:  # Flatten layer
                num_units = activation.shape[0]
                plt.figure(figsize=(12, 12))
                plt.title('Flatten Layer Activations: ' + 'Test-' + str(img_index) + ' ' + LayerName, fontsize=20)
                plt.plot(activation)
                plt.xlabel('Unit Index')
                plt.ylabel('Activation Value')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                feature_map_images.append(Image.open(buf))
                plt.close()

        cols = 2
        rows = (len(feature_map_images) + 1 + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(16, 8 * rows), constrained_layout=True)

        axs[0, 0].imshow(X[img_index])
        axs[0, 0].set_title(f"Image: {img_index} Real: {dict_classes[np.argmax(Y[img_index])]} Predicted: {dict_classes[predicted_class]}", fontsize=16)
        axs[0, 0].axis('off')

        for idx, feature_map_img in enumerate(feature_map_images):
            row = (idx + 1) // cols
            col = (idx + 1) % cols
            axs[row, col].imshow(feature_map_img)
            axs[row, col].axis('off')

        for idx in range(len(feature_map_images) + 1, rows * cols):
            row = idx // cols
            col = idx % cols
            axs[row, col].axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        plt.close()

    return img_str


# Extract Model Number
def extract_model_number(file_path):
    match = re.search(r'/data/models/(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}-\d+)\.model', file_path)
    if match:
        return match.group(1)
    else:
        return None
    

def extract_cells(annotations_path, image_path, df_classes):
    print_log(f"Opening {annotations_path}")
    print_log(f"Opening {image_path}")

    try:
        print(f"Loading image from {image_path}")
        image = Image.open(image_path)

        with open(annotations_path, 'r') as file:
            data = json.load(file)

        cells = []
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            description = shape.get('description', '')
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            min_x = int(min(x_coordinates))
            max_x = int(max(x_coordinates))
            min_y = int(min(y_coordinates))
            max_y = int(max(y_coordinates))

            if description:
                classified = list(df_classes.loc[df_classes['class'] == description].label)[0]
            else:
                classified = '0'
            if description == None:
                classified = '0'

            cropped_image = image.crop((min_x, min_y, max_x, max_y))
            buffered = BytesIO()
            cropped_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            cells.append([img_str, classified])

        print_log("Cell extraction completed.")
        return cells

    except Exception as e:
        print_log(f"Error during processing: {e}")
        return []


def create_dataset(cells, size=256):
    try:
        images_array = []

        for img_str, label in cells:
            img_data = base64.b64decode(img_str)
            image = Image.open(BytesIO(img_data)).resize((size, size))
            images_array.append([np.array(image).flatten(), label])

        df = pd.DataFrame(images_array, columns=['X', 'y'])
        return df

    except Exception as e:
        print_log(f"Error during dataset creation: {e}")
        return pd.DataFrame()


# Functions to convert an image to a base64 string
def img_to_base64_str(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Set the logo file path
logo_file_path = os.path.join(os.getcwd(), "Images", "Cinnamon-Gui-Logo.png")
logo_base64 = get_image_as_base64(logo_file_path)
# Set the CNN model representation file path
CNN_representation_file_path = os.path.join(os.getcwd(), "Images", "CNN_model_representation.png")
CNN_base64 = get_image_as_base64(CNN_representation_file_path)
# Set the small icon file path
icon_small_file_path = os.path.join(os.getcwd(), "Images", "Cinnamon-Gui-Logo-3.png")
icon_small_base64 = get_image_as_base64(icon_small_file_path)

# GUI
app_ui = ui.page_fluid(
    ui.tags.div(
        ui.tags.div(ui.img(src=f'data:image/png;base64,{icon_small_base64}', style="height: 60px; width: 60px; display: inline-block; vertical-align: middle;")),        
        ui.tags.div(ui.h2("CINNAMON-GUI"), style="display: inline-block; vertical-align: middle; margin-right: 10px;"),
        ui.tags.div(),
        ui.tags.div(ui.h6("Convolutional Neural Network And Multimodal Learning with Graphic User Interface for Digital Pathology"), style="display: inline-block; vertical-align: middle; margin-right: 5px;"),        
        ui.hr()  # Horizontal line        
      
    ),
    ui.tags.style("""
        .model-summary-table {
            width: 100%;
            border-collapse: collapse;
        }
        .model-summary-table th, .model-summary-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .model-summary-table th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #f2f2f2;
            color: black;
        }
        .model-summary-table td:nth-child(1) {
            width: 26%;
        }
        .model-summary-table td:nth-child(2) {
            width: 26%;
        }
        .model-summary-table td:nth-child(3) {
            width: 26%;
        }
        .model-summary-table td:nth-child(4) {
            width: 22%;
        }
    """),

    ui.tags.style("""
        .statistics-table {
            width: 100%;
            border-collapse: collapse;
        }
        .statistics-table th, .statistics-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .statistics-table th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: center;
            background-color: #f2f2f2;
            color: black;
        }
        .statistics-table td:nth-child(1),
        .statistics-table td:nth-child(2),
        .statistics-table td:nth-child(3){
            width: 33%;
        }
        """),
    ui.tags.style("""
        .image-index-display-table {
            width: 100%;
            border-collapse: collapse;
        }
        .image-index-display-table th, .image-index-display-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .image-index-display-table th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: center;
            background-color: #f2f2f2;
            color: black.
        }
        .image-index-display-table td:nth-child(1),
        .image-index-display-table td:nth-child(2),
        .image-index-display-table td:nth-child(3) {
            width: 33%;
        }
    """),
    ui.tags.style("""
        .plot-scrollable-container {
            overflow-y: auto;
            height: 450px;
            border: 1px solid #ddd;
        }
    """),    
    ui.tags.style("""
        .table-scrollable-container {
            overflow-y: auto;
            height: 300px;
            border: 1px solid #ddd;
        }
    """),
    ui.tags.style("""
        .plot-model-scrollable-container {
            overflow-y: auto;
            height: 400px;
            border: 1px solid #ddd;
        }
    """),
    ui.tags.style("""
        .image-scrollable-container {
            overflow-y: auto;
            height: 700px;
            border: 1px solid #ddd;
        }
    """),
    
    ui.navset_tab(
        ui.nav_panel(
            "Loading",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_file("dataset_file", "Choose a Pickle Dataset", multiple=False, accept=[".pickle"]),
                    ui.output_text_verbatim("dataset_info", placeholder=True),
                    ui.input_file("model_dir", "Choose a Model", multiple=False,  accept=[".mod"]),
                    ui.output_text_verbatim("model_info", placeholder=True),
                    ui.input_file("classes_file", "Choose a Classes file", multiple=False,  accept=[".tsv"]),                          
                    ui.output_text_verbatim("classes_info", placeholder=True),   
                    ui.hr(),  # Horizontal line
                    ui.div(style="margin-bottom: 5px;"),  # Extra space  
                    ui.p(f"Make a dataset from specimen annotation"),                                      
                    ui.input_action_button("run_labelme", "Open Labelme", class_="btn-primary"),
                    ui.input_file("specimen_file", "Choose an image file from specimen (BMP)", multiple=False, accept=[".bmp"]),
                    ui.output_text_verbatim("specimen_info", placeholder=True),
                    ui.input_file("annotation_file", "Choose an annotation file (JSON)", multiple=False, accept=[".json"]),
                    ui.output_text_verbatim("annotation_info", placeholder=True),
                    ui.input_action_button("process_annotation", "Process Annotation", class_="btn-primary"),
                ),                                 
                ui.panel_main(
                ui.p(f"Model Performances"),                                   
                ui.div(     
                    ui.output_ui("saved_model_plot_display"),
                    class_="plot-model-scrollable-container"
                ),
                ui.p(f"Model Summary"),
                ui.div(
                    ui.output_ui("model_summary_display"),  # Aggiunto per mostrare la tabella del sommario del modello
                    class_="table-scrollable-container"
                    ),                      
                ),
            )
        ),
        ui.nav_panel(
            "Training",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                        ui.column(6, ui.input_slider("seed", "Seed", 1, 10000000, value=42)),
                        ui.column(6, ui.input_text("seed_text", "Or enter seed value", value="42")),     
                    ),
                    ui.row(                   
                        ui.column(2, ui.input_slider("rotation_range", "Rotation", 1, 100, value=20)),
                        ui.column(2, ui.input_slider("width_shift_range", "W-shift", 0.1, 10, value=0.2)),
                        ui.column(2, ui.input_slider("height_shift_range", "H-shift", 0.1, 10, value=0.2)),
                        ui.column(2, ui.input_slider("shear_range", "Shear", 0.1, 10, value=0.2)),
                        ui.column(2, ui.input_slider("zoom_range", "Zoom", 0.1, 10, value=0.2)),
                        ui.column(2, ui.input_select("horizontal_flip", "Horiz-flip:", ['True', 'False'], selected='True')),
                        ui.column(2, ui.input_select("vertical_flip", "Vert-flip:", ['True', 'False'], selected='False')),
                        ui.column(2, ui.input_select("fill_mode", "Fill mode:", ['nearest', 'constant', 'reflect', 'wrap'], selected='nearest')),                        
                        ui.column(2, ui.input_slider("cval", "cval", 0, 255, value=255)),
                        ui.column(2, ui.input_select("optimizer", "Optimizer:", ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'], selected='Adam')),                        
                        ui.column(2, ui.input_select("loss", "Loss:", ['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error', 
                                                                       'huber', 'categorical_crossentropy', 'binary_crossentropy', 
                                                                       'sparse_categorical_crossentropy','kullback_leibler_divergence'], selected='categorical_crossentropy')),
                        ui.column(2, ui.input_slider("learning_rate", "Learn. rate", 0, 1.0, value=0.0001))                                                                   
                    ),
                    ui.row(                   
                        ui.column(4, ui.input_slider("epochs", "Epochs", 1, 100, value=40)),
                        ui.column(4, ui.input_slider("test_size", "Test Size", 0.1, 1.0, value=0.2)),
                        ui.column(4, ui.input_slider("batch_size", "Batch Size", 1, 128, value=32))
                    ),
                    ui.row(
                        ui.column(3, ui.input_select("C2D_1_f", "C2D1_filt:", ['2', '4', '8', '16', '32', '64', '128', '256', '512'], selected='32')),
                        ui.column(3, ui.input_slider("C2D_1_k", "C2D1_Ker:", 1, 10, value=3)),
                        ui.column(3, ui.input_select("C2D_1_a", "C2D1_act:", ['relu', 'softmax'], selected='relu')),
                        ui.column(3, ui.input_slider("MP2D_1", "MP2D1:", 1, 10, value=2))
                    ),
                    ui.row(
                        ui.column(3, ui.input_select("C2D_2_f", "C2D2_filt:", ['2', '4', '8', '16', '32', '64', '128', '256', '512'], selected='64')),
                        ui.column(3, ui.input_slider("C2D_2_k", "C2D2_Ker:", 1, 10, value=3)),
                        ui.column(3, ui.input_select("C2D_2_a", "C2D2_act:", ['relu', 'softmax'], selected='relu')),
                        ui.column(3, ui.input_slider("MP2D_2", "MP2D2:", 1, 10, value=2))
                    ),
                    ui.row(
                        ui.column(3, ui.input_select("C2D_3_f", "C2D3_filt:", ['2', '4', '8', '16', '32', '64', '128', '256', '512'], selected='128')),
                        ui.column(3, ui.input_slider("C2D_3_k", "C2D3_Ker:", 1, 10, value=3)),
                        ui.column(3, ui.input_select("C2D_3_a", "C2D3_act:", ['relu', 'softmax'], selected='relu')),
                        ui.column(3, ui.input_slider("MP2D_3", "MP2D3:", 1, 10, value=2))
                    ),
                    ui.row(
                        ui.column(3, ui.input_select("C2D_4_f", "C2D4_filt:", ['2', '4', '8', '16', '32', '64', '128', '256', '512'], selected='256')),
                        ui.column(3, ui.input_slider("C2D_4_k", "C2D4_Ker:", 1, 10, value=3)),
                        ui.column(3, ui.input_select("C2D_4_a", "C2D4_act:", ['relu', 'softmax'], selected='relu')),
                        ui.column(3, ui.input_slider("MP2D_4", "MP2D4:", 1, 10, value=2))
                    ),
                    ui.row(
                        ui.column(2, ui.input_select("D_1_f", "Dense1_filt:", ['2', '4', '8', '16', '32', '64', '128', '256', '512'], selected='256')),
                        ui.column(2, ui.input_select("D_1_a", "Dense1_act:", ['relu', 'softmax'], selected='relu')),
                        ui.column(2, ui.input_slider("l1", "L1:", 0, 0.1, value=0.001, step=0.001)),
                        ui.column(2, ui.input_slider("l2", "L2:", 0, 0.1, value=0.001, step=0.001)),
                        ui.column(2, ui.input_slider("dropout_value", "Dropout:", 0, 2.0, value=0.5, step=0.001))                        
                        
                    ),
                    ui.input_select("D_2_a", "Dense2_act:", ['relu', 'softmax'], selected='softmax'),
                    ui.input_action_button("train_model", "Train Model", class_="btn-primary")
                ),
                ui.panel_main(
                    ui.output_text_verbatim("training_log"),
                    ui.output_text("progress_text"),
                    ui.output_ui("progress_bar"),
                    ui.output_ui("image_output")
               
                )
            )
        ),    
        ui.nav_panel(
            "Report",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                        ui.input_action_button("analyze_btn", "Analyze Data", class_="btn-primary"),
                    ),
                ),
                ui.panel_main(
                    ui.p(f"Summary Chart and Table of Predicted Class Counts"),
                    ui.div(
                        ui.output_ui("stats_plot"),  # Output container for the plot
                        class_="plot-scrollable-container"
                    ),
                    ui.div(
                        ui.output_ui("report_display"),
                        class_="table-scrollable-container"
                    ),
                )
            ),
        ),

        ui.nav_panel(
            "Analysis",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                    ui.input_slider("img_index", "Select an image", 0, 1, 0, step=1),  # Imposta i valori min e max appropriati
                    ui.input_checkbox("feature_mapping", "Generate Feature Mapping Plots", value=False),                    
                    ),
                ),
                ui.panel_main(
                    ui.div(
                    ui.output_ui("image_display"),
                    class_="image-scrollable-container"
                    ),
                    ui.div(
                    ui.p(f"Image-by-Image Predictions"),
                    ui.div(
                        ui.output_ui("image_index_display"),  # Output container for the image index DataFrame
                        class_="table-scrollable-container"
                        )                        
                    ),
                    ui.layout_columns(
                        ui.card(output_widget("filtered_data"), height="400px"),
                        col_widths=[12],
                    ),                    
                ) 
            
            )   
        ),
                
        ui.nav_panel(
            "Credits",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.img(src=f'data:image/png;base64,{logo_base64}', style="height: 400px; width: 450px;"),
                ),
                ui.panel_main(
                    ui.h2("CINNAMON-GUI"),
                    ui.p(f"Version {version}"),
                    ui.p("Lunan Foldomics LLC, Copyright (c) 2024"),
                    ui.br(),
                    ui.h3("About the Product"),
                    ui.p("Cinnamon-GUI is a state-of-the-art software tool designed to redefine digital pathology. Licensed under the AGPLv3, this user-friendly interface harnesses the power of Convolutional Neural Networks (CNNs) and pre-trained models to classify biomedical images with unparalleled precision."),  # Informazioni sul prodotto
                    ui.br(),
                    ui.h3("About the Company"),
                    ui.p("At Lunan Foldomics LLC, our goal is to drive innovation in biomedicine through advanced AI and multimodal learning techniques. Explore the vast potential of Cinnamon-GUI and join us in propelling the field of biomedicine forward with innovative AI and multimodal learning solutions. Whether you are a researcher, clinician, or biomedical professional, Cinnamon-GUI equips you with the tools necessary to make groundbreaking discoveries."),  # Informazioni sulla compagnia
                    ui.p("For more information and to request software customization, please [contact us] (info@lunanfoldomicsllc.com), or visit our website www.lunanfoldomicsllc.com."),  # Informazioni sulla compagnia                    
                    ui.p("The CINNAMON-GUI code is available via: https://github.com/lunanfoldomics/Cinnamon-GUI, and it is released under GNU General Public License v3.0."),  # Informazioni sulla compagnia                                        
                )
            )   
        )
    )
)

# Definizione del server
def server(input, output, session):
    training_in_progress = reactive.Value(False)
    progress_value = reactive.Value(0)
    progress_text_value = reactive.Value("")
    plot_image = reactive.Value("")    
    global X, Y, classes, model

    model_loaded = reactive.Value(False)
    dataset_loaded = reactive.Value(False)

    # Sincronizza il valore della casella di testo con il cursore del seed
    @reactive.Effect
    def sync_text_to_slider():
        text_value = input.seed_text()
        try:
            slider_value = int(text_value)
            session.send_input_message("seed", {"value": slider_value})
        except ValueError:
            pass

    # Sincronizza il valore del cursore del seed con la casella di testo
    @reactive.Effect
    def sync_slider_to_text():
        slider_value = input.seed()
        session.send_input_message("seed_text", {"value": str(slider_value)})

    # Inizializza il file di log all'avvio
    initialize_log()

    # Effetto reattivo per il log
    @reactive.Effect
    def log_output():
        print("\n")
        print(f"Cinnamon-GUI v{version} started.")                
        show_notification(f"Cinnamon-GUI v{version} started.")
        print_log(f"Cinnamon-GUI v{version} started.")        

    # TAB DATI
    # Effetto reattivo per il caricamento del dataset
    @reactive.Effect
    def load_dataset():
        global X, Y, y, classes, dataset_filename, specimen_name
        dataset_file = input.dataset_file()
        if dataset_file is not None:
            dataset_file = dataset_file[0]
            specimen_file_name = dataset_file['name']
            print("Specimen File Name:", specimen_file_name)
            specimen_file = dataset_file
            specimen_name = specimen_file['name'].split(".pickle")[0]

            path = dataset_file["datapath"]
            with open(path, 'rb') as handle:
                df_dataset = pickle.load(handle)

            print("Loading dataset... wait until the progress signal is off.")
            show_notification("Loading dataset... wait until the progress signal is off.")
            print_log("Loading dataset...")        

            max_values = df_dataset['X'].apply(max_in_list)
            max_value = max_values.max()
            df_dataset['X_normalized'] = df_dataset['X'].apply(lambda x: [val / max_value for val in x])
            
            if df_dataset['X_normalized'].empty:
                print_log("No arrays to stack. Dataset might be empty or incorrectly loaded.")
                return

            X = np.stack(df_dataset['X_normalized'])
            y = np.int64(df_dataset['y'])
            classes = list(set(y))
            Y = to_categorical(y, len(classes))
  
            print("Dataset loaded:", X.shape, Y.shape)

            dataset_filename = dataset_file["name"]
            dataset_size = dataset_file["size"]
            dataset_type = dataset_file["type"]
            dataset_datapath = dataset_file["datapath"]

            image_size = int(np.sqrt(X.shape[1] / 3))
            X = X.reshape(-1, image_size, image_size, 3)
            show_notification("Dataset loaded.")
            print_log("Dataset loaded.")        

            print_log(f"Dataset name: {dataset_filename}")
            print_log(f"Dataset size: {dataset_size}")            
            print_log(f"Dataset type: {dataset_type}")                        
            print_log(f"Dataset datapath: {dataset_datapath}")                        
            print_log(f"Dataset info: {X.shape[0]} items; Image size: {X.shape[1]}x{X.shape[2]} px")

            test_maxvalue.set(X.shape[0] - 1)
            dataset_loaded.set(True)
        else:
            print("Dataset file is None")

    @reactive.Effect
    def load_classes():
        global classes, df_classes
        classes_file = input.classes_file()
        if classes_file is not None:
            classes_file = classes_file[0]
            path = classes_file["datapath"]
            with open(path, 'rb') as handle:
                df_classes = pd.read_csv(path, sep='\t')
                classes = list(df_classes['class'].values)
            print("Classes loaded:", classes)
            show_notification(f"Classes loaded: {classes}")
            print_log(f"Classes loaded: {classes}")            
        else:
            print("Classes file is None")
            show_notification(f"Classes file is None")
            print_log(f"Classes file is None")            

    @reactive.Effect
    def load_model():
        global model, model_path

        model_dir = input.model_dir()
        if model_dir is not None:
            model_dir = model_dir[0]
            model_file_path = model_dir["datapath"]
            with open(model_file_path, 'r') as log_file:
                for line in log_file:
                    match = re.search(r'Log saved in: (.+)', line)
                    if match:
                        model_path.set(match.group(1))
                        break

            print("Model file path:", model_file_path)
            if model_path.get():
                print("Model path found:", model_path.get())
                model.set(tf.keras.models.load_model(model_path.get()))
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    model.get().summary()
                    summary_text = buffer.getvalue()
                    print_log(summary_text)
                    
                summary_lines = summary_text.split('\n')
                model_summary = []
                for line in summary_lines:
                    parts = line.split()
                    if len(parts) > 0 and 'Layer' not in line and 'Model:' not in line and '===' not in line and len(parts) > 3:
                        clean_parts = [part.replace(',', '').replace('(', '').replace(')', '') for part in parts]
                        model_summary.append(clean_parts)
                
                max_len = max(len(row) for row in model_summary)
                model_summary = [row + [''] * (max_len - len(row)) for row in model_summary]
                columns = ['Layer', 'Type', 'Output Shape', 'Param #'] + [''] * (max_len - 4)
                model_summary_df = pd.DataFrame(model_summary, columns=columns)
                
                model_summary_html.set(model_summary_df.to_html(index=False, border=0, classes='model-summary-table'))
                model_loaded.set(True)
            else:
                print("Model path not found in the log file.")
        else:
            print("Model directory is None")

    # Invocare Labelme            
    @reactive.Effect
    def _():
        if input.run_labelme() > 0:  
            subprocess.Popen(["labelme"])  # Apri Labelme
            print("Opening Labelme for defining ROIs and labelling cells of interest.")

    @reactive.Effect
    @reactive.event(input.process_annotation)
    def process_annotation():
        global specimen_name
        annotation_file = input.annotation_file()
        specimen_file = input.specimen_file()
        specimen_file_name = specimen_file[0]
        specimen_file_name = specimen_file_name['name']
        
        print("Specimen File Name:", specimen_file_name)

        if annotation_file is not None and specimen_file is not None and df_classes is not None:
            annotation_file = annotation_file[0]
            annotations_path = annotation_file["datapath"]
            specimen_file = specimen_file[0]
            specimen_name = specimen_file['name'].split(".bmp")[0]
            specimen_file_path = specimen_file["datapath"]

            print("Specimen file name:", specimen_file)
            print("Specimen file name type:", type(specimen_file))            
            print("Specimen name:", specimen_name)            

            print(f"Annotation file path: {annotations_path}")
            print(f"Image file path: {specimen_file_path}")

            if not os.path.exists(specimen_file_path):
                print(f"Image file {specimen_file_path} does not exist.")
                print_log(f"Image file {specimen_file_path} does not exist.")
                return

            print("Model Path:", model_path.get())

            cells = extract_cells(annotations_path, specimen_file_path, df_classes)
            df = create_dataset(cells)
            dataset_path = os.path.join(os.getcwd()+'/data/datasets/', f'{specimen_name}.specimen.pickle')
            df.to_pickle(dataset_path)
            
            show_notification("Annotation processed.")
            print_log("Annotation processed.")  
            print_log(f"Specimen dataset in: {dataset_path}")          
            print("Annotation processed.")  
            print(f"Specimen dataset in: {dataset_path}")          
        else:
            print_log("Please select an annotation file and a specimen file.")

    @output
    @render.text
    def annotation_info():
        annotation_file = input.annotation_file()
        if annotation_file is not None:
            return f"Selected file: {annotation_file[0]['name']}"
        else:
            return "No annotation file selected."

    @output
    @render.text
    def specimen_info():
        specimen_file = input.specimen_file()
        if specimen_file is not None:
            return f"Selected file: {specimen_file[0]['name']}"
        else:
            return "No specimen file selected."

    model_summary_html = reactive.Value("")

    @output
    @render.ui
    def model_summary_display():
        return ui.HTML(model_summary_html.get())

    # Funzione di output per visualizzare il grafico del modello salvato
    @output
    @render.ui
    def saved_model_plot_display():
        print("saved_model_plot_display called")
        if model_loaded.get() and dataset_loaded.get():
            model_number = extract_model_number(model_path.get())         
            image_path = model_path.get() + '/' + model_number + '.learning_plot.png'                        

            if not os.path.exists(image_path):
                print("Image path does not exist:", image_path)
                return ui.HTML("Image not found. Please ensure the image has been saved.")

            try:
                saved_model_plot = Image.open(image_path)
                print("Model loaded.")                
                show_notification("Model loaded.")                
                print_log("Model loaded.")                
                saved_model_plot_str = img_to_base64_str(saved_model_plot)
                return ui.HTML(f'<img src="data:image/png;base64,{saved_model_plot_str}" alt="Image" style="max-width: 100%;">')

            except FileNotFoundError:
                print("Image not found")
                return ui.HTML("Image not found. Please ensure the image has been saved.")
        else:
            print("Model or dataset not loaded")
            return ui.HTML("You can load a dataset and run training or a saved model. Choose a Classes file for a better identification of the outcomes.")

    # Forza l'invalidazione e il ricalcolo dell'output quando il modello o il dataset sono caricati
    @reactive.Effect
    def invalidate_plot_on_load():
        if model_loaded.get() and dataset_loaded.get():
            session.send_input_message("saved_model_plot_display", {})

    @output
    @render.text
    def dataset_info():
        if input.dataset_file() is None:
            return "No dataset loaded."
        else:
            return f"Dataset loaded: {X.shape[0]} items; image resolution: {X.shape[1]}x{X.shape[2]} px"

    @reactive.Effect
    def update_slider_max():
        if test_maxvalue.get() > 0:
            session.send_input_message("img_index", {"max": test_maxvalue.get()})
    
    @output
    @render.text
    def classes_info():
        if input.classes_file() is None:
            return "No file classes loaded."
        else:
            return f"File classes loaded. Classes defined as: {classes}"
        
    @output
    @render.text
    def model_info():
        if model_loaded.get():
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                model.get().summary()
                summary_text = buffer.getvalue()
            return "Model loaded."
        else:
            return "No model loaded."

    @output
    @render.text
    def progress_text():
        return progress_text_value.get()

    @output
    @render.ui
    def image_output():
        img_str = plot_image.get()
        if img_str:
            return ui.HTML(f'<img src="data:image/png;base64,{img_str}" alt="Training plot">')
        else:
            return ui.HTML(f'<img src="data:image/png;base64,{CNN_base64}" alt="Default Image" style="max-width: 100%;">')        
            #return ui.HTML("No images available")            

    # TAB DI ADDESTRAMENTO
    @reactive.Effect
    @reactive.event(input.train_model)
    async def train_model():
        if training_in_progress.get():
            return
        training_in_progress.set(True)

        try:
            model_save_path = os.getcwd()+'/data/models/'
            seed = input.seed()
            total_epochs = input.epochs()
            test_size = input.test_size()
            batch_size = input.batch_size()
            rotation_range = input.rotation_range()
            width_shift_range = input.width_shift_range()
            height_shift_range = input.height_shift_range()
            shear_range = input.shear_range()
            zoom_range = input.zoom_range()
            horizontal_flip = input.horizontal_flip()
            vertical_flip = input.vertical_flip()
            fill_mode = input.fill_mode()
            cval = input.cval()

            optimizer = input.optimizer()
            loss = input.loss()
            learning_rate = input.learning_rate()
            l1 = input.l1()
            l2 = input.l2()
            dropout_value = input.dropout_value()

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            train_datagen = ImageDataGenerator(
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,                
                fill_mode=fill_mode,
                cval=(cval)                
            )

            test_datagen = ImageDataGenerator()

            train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
            validation_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size)

            model.set(define_model(input, X, Y))
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                model.get().summary()
            summary_text = buffer.getvalue()

            print_log(f"Training started.")
            print_log(f"seed: {seed}")
            print_log(f"total epochs: {total_epochs}")                                    
            print_log(f"test size: {test_size}")            
            print_log(f"batch size: {batch_size}")            
            print_log(f"training dataset size: {X_train.shape[0]} items; Image size: {X_train.shape[1]}x{X_train.shape[2]} px")                        
            print_log(f"testing dataset size: {X_test.shape[0]} items; Image size: {X_test.shape[1]}x{X_test.shape[2]} px")   
            print_log(f"rotation_range: {rotation_range}")
            print_log(f"width_shift_range: {width_shift_range}")
            print_log(f"height_shift_range: {height_shift_range}")
            print_log(f"shear_range: {shear_range}")
            print_log(f"zoom_range: {zoom_range}")
            print_log(f"horizontal_flip: {horizontal_flip}")
            print_log(f"vertical_flip: {vertical_flip}")
            print_log(f"fill_mode: {fill_mode}")
            print_log(f"cval: {cval}")
            print_log(f"regularization params: {l1, l2}")  
            print_log(f"dropout_value: {dropout_value}")         
            print_log(f"optimizer: {optimizer}")
            print_log(f"loss: {loss}")            
            print_log(f"learning_rate: {learning_rate}")      

            print_log(summary_text)
            print_log('Training in progress...')                

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            history = tf.keras.callbacks.History()            

            with ui.Progress(min=0, max=100, session=session) as progress:
                print('Training in progress...')                
                progress.set(message="Training in progress...", detail="(this may take a while)")
                try:
                    for epoch in range(total_epochs):
                        model.get().fit(train_generator, epochs=1, validation_data=validation_generator, callbacks=[history, early_stopping, reduce_lr])
                        current_progress = (epoch + 1) / total_epochs * 100
                        progress.set(current_progress)
                        progress_value.set(current_progress)
                        await session.send_custom_message('update-progress-bar', {'percent': current_progress})

                        if early_stopping.stopped_epoch > 0:
                            break
                except Exception as e:
                    print(f'Error during training: {str(e)}')
                    print_log(f'Error during training: {str(e)}')
                    show_notification(f'Error: {str(e)}')

            test_loss, test_acc = model.get().evaluate(validation_generator)

            print_log(f'Test accuracy: {test_acc}')
            print_log(f'Test loss: {test_loss}')            

            progress_text_value.set(f"Test accuracy: {test_acc:.2f}")    

            show_notification('Training done.')         
            show_notification('Training done.')            
            print_log('Training Done.')                       

            now = datetime.now()
            timestamp = now.strftime("%Y.%m.%d-%H.%M.%S")
            model_name = str(timestamp)+'-'+str(seed)+'.model'
            model_save_full_path = os.path.join(model_save_path, model_name)
            print(model_save_full_path)
            tf.keras.models.save_model(model.get(), model_save_full_path)
            print(f'Model {model_name} saved.')                 
            show_notification(f'Model {model_name} saved.')
            print_log(f'Model {model_name} saved.')     

            img_str = plot_history(history, os.path.join(model_save_full_path, f'{timestamp}-{seed}.learning_plot.png'), f'{timestamp}-{seed}', input.seed())
            print(f'Learning plot saved.')            
            show_notification(f'Learning plot saved.')
            print_log(f'Learning plot saved.')            
            plot_image.set(img_str)
            print_log(f'Log saved in: {model_save_full_path}')                
            shutil.copyfile(os.getcwd()+'/log.txt', os.path.join(model_save_full_path, f'{timestamp}-{seed}.mod'))                  

            test_df = pd.DataFrame([list(X_test.reshape(X_test.shape[0], -1)), list(np.argmax(Y_test, axis=1))]).T
            test_df.columns = ['X', 'y']
            test_df.to_pickle(os.path.join(model_save_full_path, f'{timestamp}-{seed}.testing.dataset.pickle'))
            print_log(f"Testing datasets saved in: {model_save_full_path}")

        finally:
            training_in_progress.set(False)

    @output
    @render.ui
    def image_display():
        img_index = input.img_index()
        feature_mapping = input.feature_mapping()

        if model.get() is not None and X is not None and Y is not None:
            pattern = r'specimen'
            print('dataset_filename:', dataset_filename)
            match = re.search(pattern, dataset_filename)
            print(match)
            if match:
                img_str = testCNN(X, Y, model.get(), classes, img_index, 'yes', feature_mapping)
                return ui.HTML(f'<img src="data:image/png;base64,{img_str}" alt="Image" style="max-width: 70%;">')
            else:
                img_str = testCNN(X, Y, model.get(), classes, img_index, 'no', feature_mapping)
                return ui.HTML(f'<img src="data:image/png;base64,{img_str}" alt="Image" style="max-width: 70%;">')

        else:
            return ui.HTML("Load the dataset and the model first.")

    # TAB OF REPORT and ANALISYS
    report_display_html = reactive.Value("")
    plot_data = reactive.Value(None)
    image_index_df_html = reactive.Value("")

    @reactive.Effect
    def _():
        if input.analyze_btn() > 0:
            if globals().get('model') is not None and globals().get('X') is not None and globals().get('Y') is not None:
                show_notification('Making report...')                

                model_here = model.get()

                X_data = globals()['X']
                Y_data = globals()['Y']
                
                predictions = model_here.predict(X_data)
                predicted_classes = predictions.argmax(axis=1)
                actual_classes = Y_data.argmax(axis=1)
                
                predicted_class_counts = pd.Series(predicted_classes).value_counts().sort_index()
                predicted_class_counts.index = [classes[i] for i in predicted_class_counts.index]
                
                actual_class_counts = pd.Series(actual_classes).value_counts().sort_index()
                actual_class_counts.index = [classes[i] for i in actual_class_counts.index]

                pattern = r'specimen'
                print('dataset_filename:', dataset_filename)
                match = re.search(pattern, dataset_filename)
                print(match)                
                if match:
                    results_df = pd.DataFrame({
                        'Class': classes,
                        'Predicted Count': predicted_class_counts.reindex(classes).fillna(0).astype(int),
                    })
                else:
                    results_df = pd.DataFrame({
                        'Class': classes,
                        'Predicted Count': predicted_class_counts.reindex(classes).fillna(0).astype(int),
                        'Actual Count': actual_class_counts.reindex(classes).fillna(0).astype(int)
                    })

                if match:
                    image_index_df = pd.DataFrame({
                        'Image Index': range(len(predicted_classes)),
                        'Predicted Class': [classes[i] for i in predicted_classes]
                    })
                else:
                    image_index_df = pd.DataFrame({
                        'Image Index': range(len(predicted_classes)),
                        'Predicted Class': [classes[i] for i in predicted_classes],
                        'Actual Class': [classes[i] for i in actual_classes]                        
                    })

                print(results_df)
                print(image_index_df)
                print_log(f"Report: {results_df}")

                report_display_html.set(results_df.to_html(classes='statistics-table', index=False))
                image_index_df_html.set(image_index_df.to_html(classes='image-index-display-table', index=False))
                session.send_input_message("report_display", {"html": report_display_html.get()})
                session.send_input_message("image_index_display", {"html": image_index_df_html.get()})
                plot_data.set(results_df)
                show_notification('Report done.')    

    @output
    @render.ui
    def report_display():
        return ui.HTML(report_display_html.get())



    # Aggiungere l'effetto reattivo per il filtraggio dei dati
    @reactive.calc
    def filtered_df():
        # Utilizza i dati del dataset caricati
        req(X is not None)

        pattern = r'specimen'
        match = re.search(pattern, dataset_filename)
        if match:
            df = pd.DataFrame({
                "Image Index": range(len(Y)),
                "Predicted Class": [classes[np.argmax(y)] for y in model().predict(X)]
            })
        else:
            df = pd.DataFrame({
                "Image Index": range(len(Y)),
                "Predicted Class": [classes[np.argmax(y)] for y in model().predict(X)],
                "Actual Class": [classes[np.argmax(y)] for y in Y]                
            })

        return df

    @render_widget
    def filtered_data():
        df = filtered_df()
        pattern = r'specimen'
        match = re.search(pattern, dataset_filename)
        if match:  
            return px.scatter(
                df,
                x="Image Index",
                y="Predicted Class",
                color="Predicted Class",
                title="Filtered Data"
            )
        else:
            return px.scatter(
                df,
                x="Image Index",
                y="Predicted Class",
                color="Actual Class",
                title="Filtered Data"
            )

    # Aggiungere l'output per il widget filtrato
    @output
    @render.ui
    def image_index_display():
        df = filtered_df()
        return ui.HTML(df.to_html(classes='image-index-display-table', index=False))
    
    '''
    @output
    @render.ui
    def image_index_display():
        return ui.HTML(image_index_df_html.get())
    '''
    
    @output
    @render.ui
    def stats_plot():
        df = plot_data.get()
        if df is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            pattern = r'specimen'
            print('dataset_filename:', dataset_filename)
            match = re.search(pattern, dataset_filename)
            print(match)    
            if match:
                df.plot(kind='bar', x='Class', y=['Predicted Count'], ax=ax)
                ax.set_ylabel("Count")
                ax.set_title(f"Dataset: {specimen_name} - Predicted")
                plt.xticks(rotation=45)   
                plt.tight_layout()             
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                buffer.close()                
            else:
                df.plot(kind='bar', x='Class', y=['Predicted Count', 'Actual Count'], ax=ax)
                ax.set_ylabel("Count")
                ax.set_title(f"Dataset: {specimen_name} - Predicted vs Actual Counts")
                plt.xticks(rotation=45)   
                plt.tight_layout()             
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                buffer.close()

            return ui.HTML(f'<img src="data:image/png;base64,{img_str}" alt="Statistics Plot" style="max-width: 70%;">')
        else:
            return ui.HTML("No data to display.")

app = App(app_ui, server)

if __name__ == '__main__':
    app.run()
