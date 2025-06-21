



import os # Importing os module for environment variable access
import numpy as np
import pandas as pd
import cv2
from PIL import Image # Importing PIL for image processing
import traceback 
import tensorflow as tf



from flask import Flask, request,  render_template, jsonify # Importing Flask modules for web app functionality
from werkzeug.utils import secure_filename # Importing secure_filename for safe file handling

from keras.models import  Model  # Importing Model class from Keras to create a model instance
from keras.layers import Input , Flatten , Dense , Dropout , Conv2D , MaxPooling2D

from keras.applications import VGG19# Importing VGG19 model from Keras applications
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3)) # Load the VGG19 model with pre-trained weights, excluding the top layer

for layer in base_model.layers:  # Freeze the layers of the base model to prevent them from being trained
    layer.trainable = False  # Set the trainable attribute of each layer to False

x = base_model.output  # Get the output of the base model
x = Flatten()(x) # Flatten the output to convert it into a 1D vector

class_1= Dense(4608 , activation='relu')(x)  # Add a fully connected layer with 4608 units and ReLU activation
dropout= Dropout(0.1)(class_1)  # Add a dropout layer with a dropout rate of 0.2 to prevent overfitting

class_2= Dense(2304, activation='relu')(dropout)  # Add another fully connected layer with 2304 units and ReLU activation


output = Dense(2, activation='softmax')(class_2)  # Add the final output layer with 2 units (for binary classification) and softmax activation function to produce probabilities for each class



model_03 =  Model(inputs=base_model.input, outputs=output)  # model of empty weights

model_03.load_weights(r"C:\downloads\pneumonia_detection\model_weights\vgg19_model_01.weights.h5")



#create flask app

app = Flask(__name__)  # Create a Flask application instance

print("Model loaded successfully")  # Print a message indicating that the model has been loaded successfully



# helper functions
def get_className(class_index):
    if class_index == 0:
        return "Normal"
    elif class_index == 1:
        return "Pneumonia"
    else:
        return "Unknown"


    
def get_Result(img):

        #read the image
        image = cv2.imread(img)
        # convert the image to RGB format
        image = Image.fromarray(image, 'RGB')
        # resize the image to the input shape of the model
        image = image.resize((128,128)) # Resize the image to 128x128 pixels as expected by the model

        # convert the image to a numpy array
        image = np.array(image)

        # Normalize the image data to 0 and 1 beacuase the model expects input in this range
        image = image / 255.0

        # Add an extra dimension to match the input shape of the model
        input_img = np.expand_dims(image, axis=0)  # Adding an extra dimension for the batch size as the model expects 4D input

        # Make a prediction using the model
        result = model_03.predict(input_img)

        # Get the class index with the highest probability
        result01 = np.argmax(result)  

 

        return result01


@app.route('/', methods=['GET'])
 # Define the route for the home page
def index():
    return render_template('index.html')




import traceback  # Add at the top


import traceback  # at the top of your file if not already present

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(filepath)

            print(f"[INFO] File saved at: {filepath}")

            value = get_Result(filepath)
            print(f"[INFO] Prediction index: {value}")

            result = get_className(value)
            print(f"[INFO] Prediction class: {result}")

            return jsonify({'prediction': result})
        except Exception as e:
            print("[ERROR] Exception occurred during prediction:")
            traceback.print_exc()  # This prints the full error stack trace
            return "Prediction error", 500
    return "Invalid request", 400




'''@app.route('/predict', methods=['POST'])  # Define the route for the prediction endpoint
def upload():
    if request.method == 'POST':
        try:   # Check if the request method is POST
            f = request.files['file']  # Get the uploaded file from the request
            basepath = os.path.dirname(__file__)  # Get the directory of the current file
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))  # Create a secure file path
            f.save(filepath)  # Save the uploaded file to the specified path
            value = get_Result(filepath)  # Call the getResult function to make a prediction
            result = get_className(value)  # Get the class name based on the prediction result
            return result  # Return the class name as the response
        except Exception as e:   # Handle any exceptions that occur during the file upload or prediction process
            print(f"Error: {e}")
            

    return None'''




if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application in debug mode
    print("App is running")  # Print a message indicating that the app is running


