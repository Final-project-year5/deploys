from django.apps import AppConfig
import os
from tensorflow.keras.models import load_model
from django.conf import settings

class AttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Attendance'
    model = None

    def ready(self):
        # Construct the full path to the model file
        model_path = os.path.join(settings.BASE_DIR, 'attendance', 'models', 'newmodel.h5')
        self.model = load_model(model_path)
