from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from keras.models import load_model
import cv2
import numpy as np
from django.conf import settings
import os
from rest_framework.response import Response
import joblib
import pickle 
# Create your views here.
class Brain(APIView):
    def preprocess_image(self,image_path, target_size=(224, 224)):
        img = cv2.imread(image_path)
        image = cv2.resize(img, target_size)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        print('passed preprocessing')
        return image

    def get_prediction(self,image):
        folder_path = os.path.join(settings.BASE_DIR, 'api\models\Brain')
        file_name = 'brain_xception.h5'
        model_path=os.path.join(folder_path, file_name)
        model=load_model(model_path)

        pred=model.predict(image)
        return pred[0][0]
    def print_results(self,pred):
        if pred < .5:
            result="You do not have brain tumor"
        else:
            result="You  have brain tumor"
        return result

    def post(self,request,format=None):
        folder_path = os.path.join(settings.BASE_DIR, 'api\models')
        temp_image_path = os.path.join(folder_path, 'received_image.jpg')
        # print(temp_image_path)
        with open(temp_image_path, 'wb') as temp_image:
            temp_image.write(request.data['photo'].read())
        # print(picture)
        image=self.preprocess_image(temp_image_path)
        prediction=self.get_prediction(image)
        result=self.print_results(prediction)
        return Response({"message":f"{result}"})

class BreastCancer(APIView):
    def get_class(self,l):
        folder_path = os.path.join(settings.BASE_DIR, 'api\models\Brest cancer')
        file_name = 'lg_breast_cancer_model.joblib'
        model_path=os.path.join(folder_path, file_name)
        reg = joblib.load(model_path)
        l=np.array(l)
        l=l.reshape(1, -1)
        p=reg.predict(l)
        return p[0]
    def print_result(self,p):
        if p==0:
            re='your tumor is benign'
        else:
            re="your tumor is malignant"
        return re
    def post(self,request):
        data=request.data['data']
        print("typs is ",type(data))
        result=self.print_result(self.get_class(data))
        return Response({"message":f"{result}"})


class Covid(APIView):
    def preprocess_image(self,image_path, target_size=(224, 224)):
        img = cv2.imread(image_path)
        image = cv2.resize(img, target_size)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        return image
    def predict_image_class(self,img_path, model):
        img = self.preprocess_image(img_path)
        prediction = model.predict(img)
        class_labels = ['Covid', ' Normal', ' Pneumonia']  # Define your class labels
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    def post(self,request):
        folder_path = os.path.join(settings.BASE_DIR, 'api\models\Covied')
        file_name = 'vgg16_model_covied_v3.h5'
        model_path=os.path.join(folder_path, file_name)
        model=load_model(model_path)

        folder_path = os.path.join(settings.BASE_DIR, 'api\models')
        temp_image_path = os.path.join(folder_path, 'received_image.jpg')
        # print(temp_image_path)
        with open(temp_image_path, 'wb') as temp_image:
            temp_image.write(request.data['photo'].read())

        predicted_class = self.predict_image_class(temp_image_path, model)
        return Response({"message":f"{predicted_class}"})

class Obesity(APIView):
    def post(self,request):
        l=request.data['data']
        message=self.print_result(self.get_result(self.get_class(l)))
        return Response({"message":f"{message}"})

    def get_class(self,l):
        folder_path = os.path.join(settings.BASE_DIR, 'api\models\Obesity')
        file_name = 'obesity_rand_for.sav'
        model_path=os.path.join(folder_path, file_name)
        loaded_model=pickle.load(open(model_path, 'rb'))
        l=np.array(l)
        l=l.reshape(1, -1)
        p=loaded_model.predict(l)
        return p[0]
    def get_result(self,p):
        target=['Insufficient_Weight','Normal_Weight', 'Obesity_Type_I','Obesity_Type_II', 'Obesity_Type_III','Overweight_Level_I','Overweight_Level_II', ]
        x=target[p]
        return x
    def print_result(self,x):
        re="you are {}".format(x)
        return re