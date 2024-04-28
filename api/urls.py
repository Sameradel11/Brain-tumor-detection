from django.contrib import admin
from django.urls import path,include
from . import views


urlpatterns = [
    path('brain',views.Brain.as_view(),name='brain'),
    path('breast_cancer',views.BreastCancer.as_view()),
    path('covid',views.Covid.as_view()),
]
