from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("map/", views.map, name="map"),
    path("play/", views.play, name="play"),
    path("select/<int:i>/<int:j>/", views.select, name="select"),
    path("machine/", views.machine, name="machine"),
    path("machine-play/", views.machine_play, name="machine-play"),
    path("machine-select/", views.machine_select, name="machine-select"),
]
