# Register API
from django.shortcuts import render
from django.http import HttpResponse

from rest_framework import generics, permissions
from rest_framework.response import Response
from knox.models import AuthToken
from .serializers import UserSerializer, RegisterSerializer 

#LoginAPI
from django.contrib.auth import login 

from rest_framework import permissions
from rest_framework.authtoken.serializers import AuthTokenSerializer
from knox.views import LoginView as KnoxLoginView 
#model 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics

class RegisterAPI(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({
        "user": UserSerializer(user, context=self.get_serializer_context()).data,
        "token": AuthToken.objects.create(user)[1]
        })

#LoginAPI
class LoginAPI(KnoxLoginView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request, format=None):
        serializer = AuthTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        login(request, user)
        return super(LoginAPI, self).post(request, format=None) 

def viewresult(request): 
    data="abdjkdk"
    data = pd.read_csv("C:/Users/Suraj S. Jha/Desktop/rapid-api-unesco/unescoapi/app1/diabetes.csv") 
    data.shape
    data.head(5)  
    data.isnull().values.any() 
    data.corr()
    diabetes_map = {True: 1, False: 0} 
    data['diabetes'] = data['Outcome'].map(diabetes_map) 
    diabetes_true_count = len(data.loc[data['Outcome'] == True])
    diabetes_false_count = len(data.loc[data['Outcome'] == False]) 
    ## Train Test Split

    
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    predicted_class = ['Outcome'] 
    X = data[feature_columns].values
    y= data[predicted_class].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=5)
    ## Apply Algorithm


    random_forest_model = RandomForestClassifier(random_state=5)

    random_forest_model.fit(X_train, y_train.ravel()) 
    predict_train_data = random_forest_model.predict(X_test)

    

    print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

    

    

    return HttpResponse(predict_train_data)
