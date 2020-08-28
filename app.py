from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)


# Load your trained model
model = load_model("C:/Users/Abhishek Choudhary/.spyder-py3/model")


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'abhi', secure_filename(f.filename))
        file_path = file_path.replace("\\\\","\\" )
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        if preds[0][0] == 0 :
            ans = "yes it has tumor" 
        else :
            ans  = "no it has not tumor"
        return ans 
    return None


if __name__ == '__main__':
    app.run(debug=True)