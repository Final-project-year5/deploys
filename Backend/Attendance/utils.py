import socket
from django.utils import timezone
from .models import ActivityLog
#####################
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2

######################
def log_activity(name, action, resource, details, ip_address, status_code):
    timestamp = timezone.now()
    ActivityLog.objects.create(
        name=name,
        action=action,
        resource=resource,
        details=details,
        ip_address=ip_address,
        status_code=status_code,
        timestamp=timestamp
    )

def get_client_ip(request):
    # Get client's IP address
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
##attendance
model = load_model('Attendance/models/newmodel.h5')
# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (224, 224))  # Adjust size as per your model's requirement
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image / 255.0  # Normalize if required

def predict(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    return predictions