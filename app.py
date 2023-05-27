from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import cv2 as cv
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)
load_dotenv()
model_weights = os.getenv('MODEL')

model = Sequential()
model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.load_weights(model_weights)


@app.route('/predict', methods=['GET'])
def home():
    classes = {4: ('nv', ' Melanocytic Nevi'), 6: ('mel', 'Melanoma'), 2 :('bkl', 'Benign Keratosis-like Lesions'), 1:('bcc' , ' Basal Cell Carcinoma'), 5: ('vasc', ' Pyogenic Granulomas and Hemorrhage'), 0: ('akiec', 'Actinic Keratoses and Intraepithelial Carcinomae'),  3: ('df', 'Dermatofibroma')}
    file = request.files.get('image')
    image = Image.open(file)
    image.save(file.filename)
    img = cv.imread(file.filename,0)
    invImg = cv.bitwise_not(img)
    img_str = cv.imencode('.jpg', invImg )[1] 
    nparr = np.frombuffer(img_str, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    image = cv.resize(image, (28,28))
    predictions = model.predict(image.reshape(1,28,28,3))
    predictions = predictions[0]
    max_prob = max(predictions)
    os.remove(file.filename)
    conf_thr = 0.80
    if max_prob > conf_thr:
        class_ind = list(predictions).index(max_prob)
        class_name = classes[class_ind]
        disease = class_name[1]
        return jsonify({'disease': str(class_name), 'probability': str(max_prob)})
        
    disease = 'No Disease'
    return jsonify({'disease': str(disease)})

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")