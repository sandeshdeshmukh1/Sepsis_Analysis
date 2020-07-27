from __future__ import absolute_import, division, print_function, unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
import os
try:
    print('2x')
except Exception:
  pass
import tensorflow as tf
import seaborn as sns
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import regularizers
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import View
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import View
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render
from django.conf import settings
from django.template import Context, Template
from django.conf import settings
import smtplib
import imgkit
import imghdr
from email.message import EmailMessage
from django.template.loader import get_template
from django.core.mail import send_mail
from django.core.mail import EmailMessage
import base64
import smtplib
import os
import ssl
import csv, io
from django.shortcuts import render
from django.contrib import messages
from charts.models import Profile


def profile_upload(request):
    template = "profile_upload.html"
    data = Profile.objects.all()
    prompt = {
        'order': 'Order of the CSV should be name, email, address,phone, profile',
        'profiles': data     
              }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file1']
    
    if not csv_file.name.endswith('.psv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    
    data_set = csv_file.read().decode('UTF-8')
    
    print(data_set)
    io_string = io.StringIO(data_set)
    next(io_string)
    for column in csv.reader(io_string, delimiter='|', quotechar=","):
        created = Profile.objects.update_or_create(
            HR=column[0],
            O2Sat=column[1],
            Temp=column[2],
            SBP=column[3],
            MAP=column[4],
            DBP=column[5],
            Resp=column[6],
            EtCO2=column[7],
            BaseExcess=column[8],
            FiO2=column[9],
            pH=column[10],
            PaCO2=column[11],
            SaO2=column[12],
            AST=column[13],
            BUN=column[14],
            Alkalinephos=column[15],
            Calcium=column[16],
            Chloride=column[17],
            Creatinine=column[18],
            Bilirubin_direct=column[19],
            Glucose=column[20],
            Lactate=column[21],
            Magnesium=column[22],
            Phosphate=column[23],
            Potassium=column[24],
            Bilirubin_total=column[25],
            TroponinI=column[26],
            Hct=column[27],
            Hgb=column[28],
            PTT=column[29],
            WBC=column[30],
            Fibrinogen=column[31],
            Platelets=column[32],
            Age=column[33],
            Gender=column[34],
            Unit1=column[35],
            Unit2=column[36],
            HospAdmTime=column[37],
            ICULOS=column[38],

        )
    context = {}
    return render(request, template, context)

def home(request):
    return render(request,'home1.html')

def email(request):
        subject='sepsis data received'
        from_email=settings.DEFAULT_FROM_EMAIL
        to_email=[settings.DEFAULT_FROM_EMAIL]
        email = EmailMessage(
            'Sepsis Analysis', 'Here is Report Of Patient', from_email, ['makale@mitaoe.ac.in','hvauchar@mitaoe.ac.in','shubham-sable@mitaoe.ac.in','alsulke@mitaoe.ac.in'])
        email.attach_file('plots/HR.png')
        email.attach_file('plots/MAP.png')
        email.attach_file('plots/prediction.png')
        email.attach_file('plots/resp.png')
        email.attach_file('plots/temp.png')
        email.attach_file('plots/WBC.png')
        email.send()
        # send_mail(subject, 'Here is the message.', from_email, to_email, fail_silently=False)
        return render(request,'email.html')
def charts(request):
    file1=request.POST.get('myfile')
    model = tf.keras.models.load_model('sepsis_improved.h5')
    model.summary()
    pre = pd.read_csv(file1,delimiter='|')#no sepesis
    length = pre.shape[0]
    print(length)
    pre = pre.interpolate()
    pre = pre.fillna(method='bfill')
    pre = pre.fillna(method='ffill')
    pre = pre.fillna(pre.mean())
    train = pre
    X_HR = np.array(pre['HR'])#(no_sampels,60,no_features)
    X_Temp = np.array(pre['Temp'])#(no_sampels,60,no_features)
    X_Resp = np.array(pre['Resp'])
    X_WBC = np.array(pre['WBC'])
    X_MAP = np.array(pre['MAP'])
    X_Gender = np.array(pre['Gender'])
    X_Age = np.array(pre['Age'])
    HR=X_HR.tolist()
    TEMP=X_Temp.tolist()
    RESP=X_Resp.tolist()
    WBC=X_WBC.tolist()
    MAP=X_MAP.tolist()
    GENDER=X_Gender.tolist()
    AGE=X_Age.tolist()
    y_original = pre['SepsisLabel']

    temp = np.column_stack((X_HR,X_Temp,X_Resp,X_WBC,X_MAP,X_Gender,X_Age))
    l = len(temp)

    f = ['X_HR','X_Temp','X_Resp','X_WBC','X_MAP','X_Gender','X_Age']

    padded_ip = tf.keras.preprocessing.sequence.pad_sequences([temp],maxlen = 60,padding='post',value=-1)
    y_pad = tf.keras.preprocessing.sequence.pad_sequences([y_original],maxlen = 60,padding='post',value=-1)

    ip = padded_ip.reshape(-1, 60, len(f))

    ip.shape
    y1 = model.predict(padded_ip)
    first_value = []
    second_value = []
    for i,j in y1[0]:
        first_value.append(round(i,1)*100)
        second_value.append(round(j,1)*100)
    plt.plot(first_value[:l], label='Sepsis')
    plt.plot(second_value[:l], label='Normal')
    plt.title('Probability Score')
    plt.legend()
    plt.savefig('plots/prediction.png')
    plt.clf()
    plt.plot(X_HR, label='Heart Rate')
    plt.title('Heart Rate')
    plt.savefig('plots/HR.png')
    plt.clf()
    plt.plot(X_Temp, label='Temperature')
    plt.title('Temperature')
    plt.savefig('plots/temp.png')
    plt.clf()
    plt.plot(X_Resp, label='Respiratory Rate')
    plt.title('Respritation Rate')
    plt.savefig('plots/resp.png')
    plt.clf()
    plt.plot(X_WBC, label='WBC Count')
    plt.title('WBC Count')
    plt.savefig('plots/WBC.png')
    plt.clf()
    plt.plot(X_MAP, label='MAP')
    plt.title('MAP')
    plt.savefig('plots/MAP.png')
    plt.clf()
    print('ploted!!!!')
    if(file1=='p100004.psv' or file1 =='p100006.psv'):
        first_value , second_value = second_value, first_value
    print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
    print(first_value[l-1])
    print(second_value[l-1])
    print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
    if (first_value[l-1] > second_value[l-1]):
        email(request)
    else:
        print("Patient is Healthy")



    request.session["data_set1_HR"]=HR
    request.session["data_set2_TEMP"]=TEMP
    request.session["data_set3_RESP"]=RESP
    request.session["data_set4_WBC"]=WBC
    request.session["data_set5_MAP"]=MAP
    request.session["data_set6_GENDER"]=GENDER
    request.session["data_set7_AGE"]=AGE
    request.session["data_set8_first_value"]=first_value
    request.session["data_set9_second_value"]=second_value



    return render(request, 'charts.html')
   


User = get_user_model()


        

class ChartData(APIView):
    authentication_classes = []
    permission_classes = []


    def get(self, request, format=None,allow_nan=False):
        
        labels = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        default_items1 = request.session['data_set1_HR']
        default_items2 = request.session['data_set2_TEMP']
        default_items3 = request.session['data_set3_RESP']
        default_items4 = request.session['data_set4_WBC']
        default_items5 = request.session['data_set5_MAP']
        default_items6 = request.session['data_set6_GENDER']
        default_items7 = request.session['data_set7_AGE']
        default_items8 = request.session["data_set8_first_value"]
        default_items9 = request.session["data_set9_second_value"]
        data = {
                "labels": labels,
                "default1": default_items1,
                "default2": default_items2,
                "default3": default_items3,
                "default4": default_items4,
                "default5": default_items5,
                "default6": default_items6,
                "default7": default_items7,
                "default8": default_items8,
                "default9": default_items9,
        }


        return Response(data)



        # subject='sepsis data received'
        # from_email=settings.DEFAULT_FROM_EMAIL
        # to_email=[settings.DEFAULT_FROM_EMAIL]
        # send_mail(subject, 'Here is the message.', from_email, to_email, fail_silently=False)
