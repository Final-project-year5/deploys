from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist

from .models import Student

class CustomAuthBackend(BaseBackend):
    def authenticate(self, request, student_id=None, password=None, **kwargs):
        UserModel = get_user_model()
        try:
            student = Student.objects.get(student_id=student_id)
            user = student.user
            print("Student ID:", student_id)
            print("User ID from Student:", user.id)
            if user.check_password(password):
                return user
        except ObjectDoesNotExist:
            pass  # Log this exception for debugging
        return None

    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except ObjectDoesNotExist:
            pass  # Log this exception for debugging
        return None