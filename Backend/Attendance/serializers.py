from rest_framework import serializers
from .models import AttendanceRecord, Attendance, Course, CustomUser

class AttendanceRecordSerializer(serializers.ModelSerializer):
    # id = serializers.CharField(source='_id')  # Map '_id' to 'id' field

    class Meta:
        model = AttendanceRecord
        fields = '__all__'

class AttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = ['date', 'status']

class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['name', 'user_type']  # Exclude 'id'

class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'email', 'name', 'phone_number', 'gender', 'department', 'college', 'role']