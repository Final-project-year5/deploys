from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from .models import CustomUser, Student, FaceData, ActivityLog, Section, PermissionRequest, Course, Attendance, Notification, AttendanceRecord,Schedule,Teacher
from django.contrib.auth import authenticate, login
from rest_framework.authtoken.models import Token
from .utils import log_activity, get_client_ip
from django.contrib.auth.decorators import login_required
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .decorators import token_required
from django.http import HttpResponseRedirect
from django.core.serializers.json import DjangoJSONEncoder
from django.forms.models import model_to_dict
from django.core.mail import send_mail
from django.http import JsonResponse,QueryDict
from django.template.loader import render_to_string
from django_otp import devices_for_user
from django_otp.oath import TOTP
from django.contrib.auth.hashers import make_password
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from django.core.cache import cache
import json
import base64
from django.views.decorators.http import require_http_methods
from rest_framework import generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .serializers import AttendanceRecordSerializer,AttendanceSerializer, CourseSerializer, UserSerializer ,AttendanceSerializer,CustomUserSerializer
from dns import resolver
from rest_framework import status
from rest_framework.views import APIView
from django.utils import timezone
import logging
from datetime import timedelta
from django.contrib import messages
from django.db.models import Case, When, F, Value, CharField
from django.db import IntegrityError
# from my_module import DatabaseError
from django.db.utils import DatabaseError
from django.utils.decorators import method_decorator
from tensorflow.keras.models import Model
from django.db import models

from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from keras.models import load_model
from scipy.spatial.distance import cosine

from numpy.linalg import norm
from django.core.files.base import ContentFile
from io import BytesIO

from .utils import get_client_ip
import csv
from django.core.files.base import BytesIO
import base64
from .models import Attendance, FaceData
from datetime import datetime
from .apps import AttendanceConfig
import pandas as pd
from django.utils.dateparse import parse_time

# Create your views here.
logger = logging.getLogger(__name__)

def home(request):
    return HttpResponse("hi merry:)")

@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        student_id = request.POST.get('student_id')
        email = request.POST.get('email')
        phonenumber = request.POST.get('phonenumber')
        password = request.POST.get('password')
        section = request.POST.get('section') 
        department = request.POST.get('department') 
        college = request.POST.get('college')
        gender = request.POST.get('gender') 
        year_semester = request.POST.get('year_semester') 
        face_image_file = request.FILES.get('face_image')

        if face_image_file:
            # Create a CustomUser instance
            user = CustomUser.objects.create_user(
                email=email,
                name=fullname,
                phone_number=phonenumber,
                department=department,
                college=college,
                gender=gender,
                role='student'
            )

            # Save the password
            user.set_password(password)
            user.save()

            # Create a Section instance
            section_instance = Section.objects.create(
                name=section,
            )

            # Create a Student instance
            student = Student.objects.create(
                student_id=student_id,
                section=section_instance,  # Link the student to the Section
                year_semester=year_semester,
                user=user  # Link the student to the CustomUser
            )

            # Create a FaceData instance with the uploaded face image file
            face_data = FaceData.objects.create(
                face_encoding=face_image_file,  # Associate the face image file with the face encoding attribute
                user=user  # Link the face data to the CustomUser
            )

            return JsonResponse({'message': 'Registration successful'}, status=201)
        else:
            return JsonResponse({'error': 'Face image file is required'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        password = request.POST.get('password')
        
        # Authenticate the user using the custom backend
        user = authenticate(request, student_id=student_id, password=password)
        
        if user is not None:
            # Generate a refresh token
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)
            
            # Update the token field in the user model with the new access token
            user.token = access_token
            user.save()
            
            # Log the login activity
            name = user.name
            action = "Login"
            resource = "User"
            details = f"User {name} logged in successfully"
            ip_address = get_client_ip(request)
            status_code = 200
            log_activity(name, action, resource, details, ip_address, status_code)
            return JsonResponse({'message': 'Login successful', 'access_token': access_token}, status=200)
        else:
            # Log the failed login attempt
            name = student_id
            action = "Login"
            resource = "User"
            details = f"Failed login attempt for user with student ID {student_id}"
            ip_address = get_client_ip(request)
            status_code = 401
            log_activity(name, action, resource, details, ip_address, status_code)
            return JsonResponse({'error': 'Invalid student ID or password.'}, status=401)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    
def get_activity_logs(request):
    if request.method == 'GET':
        logs = ActivityLog.objects.all().order_by('-timestamp')[:10]  # Fetching the latest 10 logs
        data = []
        for log in logs:
            log_data = {
                'name': log.name,
                'action': log.action,
                'resource': log.resource,
                'details': log.details,
                'ip_address': log.ip_address,
                'status_code': log.status_code,
                'timestamp': log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            data.append(log_data)
        return JsonResponse({'logs': data}, status=200)
    else:
        return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)


@token_required
def user_profile(request, user, *args, **kwargs):
    # Assuming a one-to-one relationship between CustomUser and Student models
    try:
        student = Student.objects.get(user=user)
    except Student.DoesNotExist:
        student = None

    # Assuming a one-to-one relationship between CustomUser and FaceData models
    try:
        face_data = FaceData.objects.get(user=user)
    except FaceData.DoesNotExist:
        face_data = None

    if student:
        profile_data = {
            'fullname': user.name,
            'student_id': student.student_id,
            'email': user.email,
            'phonenumber': user.phone_number,
            'section': model_to_dict(student.section),  # Convert Section object to dictionary
            # Add other fields as needed
        }

        if face_data:
            profile_data['face_encoding'] = face_data.face_encoding.url  # Assuming face_encoding is a FileField

        return JsonResponse(profile_data)
    else:
        return JsonResponse({'error': 'User profile not found'}, status=404)


@csrf_exempt
@token_required
def update_profile(request, user, *args, **kwargs):
    # Load JSON data from request.body
    data = json.loads(request.body)

    # Retrieve the current profile data
    try:
        student = Student.objects.get(user=user)
    except Student.DoesNotExist:
        student = None

    try:
        face_data = FaceData.objects.get(user=user)
    except FaceData.DoesNotExist:
        face_data = None

    # Update the user profile based on the provided data
    if student:
        # Update user fields
        if 'name' in data:
            user.name = data['name']
        if 'email' in data:
            user.email = data['email']
        if 'phone_number' in data:
            user.phone_number = data['phone_number']
        user.save()

        # Update student fields
        if 'student_id' in data:
            student.student_id = data['student_id']
        student.save()

        # Update face data if available
        if face_data:
            if 'face_encoding' in data:
                face_data.face_encoding = data['face_encoding']
            face_data.save()

        # Construct and return the updated profile data
        profile_data = {
            'fullname': user.name,
            'student_id': student.student_id if student else None,
            'email': user.email,
            'phonenumber': user.phone_number,
            'section': model_to_dict(student.section) if student else None,
            'face_encoding': face_data.face_encoding.url if face_data else None,
            # Add other fields as needed
        }
        return JsonResponse({'success': True, 'message': 'Profile updated successfully', 'profile_data': profile_data})
    else:
        return JsonResponse({'error': 'User profile not found'}, status=404)




@csrf_exempt
def send_otp(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            if email:
                try:
                    user = CustomUser.objects.get(email=email)
                except CustomUser.DoesNotExist:
                    return JsonResponse({'success': False, 'message': 'User with this email does not exist'})
                
                # Generate OTP
                totp = TOTP(key=user.name.encode('utf-8'))
                otp_code = totp.token()

                # Store OTP in cache
                cache.set(email, otp_code, timeout=3000)  # Set expiration time to 5 minutes
                
                # Send OTP via email
                subject = 'Password Reset OTP'
                message = render_to_string('password_reset_email.html', {'otp_code': otp_code})
                try:
                    send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])
                    return JsonResponse({'success': True, 'message': 'OTP sent successfully'})
                except Exception as e:
                    return JsonResponse({'success': False, 'message': f'Failed to send OTP: {str(e)}'})
            else:
                return JsonResponse({'success': False, 'message': 'Email not provided'})
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'message': 'Invalid JSON format'})
    else:
        return JsonResponse({'success': False, 'message': 'Only POST requests are allowed'})


@csrf_exempt    
def update_password(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        otp = data.get('otp')
        new_password = data.get('new_password')

        print("Type of otp:", type(otp))

        if email and otp and new_password:
            try:
                user = CustomUser.objects.get(email=email)
            except CustomUser.DoesNotExist:
                return JsonResponse({'success': False, 'message': 'User with this email does not exist'})

            # Retrieve OTP from cache
            cached_otp = cache.get(email)
            print("Type of cached_otp:", type(cached_otp))
            print("Cached OTP:", cached_otp)

            if cached_otp is None or str(cached_otp) != otp:
                return JsonResponse({'success': False, 'message': 'Invalid OTP'})

            # Update password
            user.set_password(new_password)
            user.save()

            # Clear OTP from cache after successful password update
            cache.delete(email)

            return JsonResponse({'success': True, 'message': 'Password updated successfully'})
        else:
            return JsonResponse({'success': False, 'message': 'Missing email, OTP, or new password'})
    else:
        return JsonResponse({'success': False, 'message': 'Only POST requests are allowed'})
    
#Permission create
@csrf_exempt  
def create_permission_request(request):
    if request.method == 'POST':
        teacher = request.POST.get('teacher')
        reason = request.POST.get('reason')
        evidence = request.FILES.get('evidence')
        sick_leave = request.POST.get('sickLeave', False)
        
        # Save the form data to the database
        permission_request = PermissionRequest.objects.create(
            teacher=teacher,
            reason=reason,
            evidence=evidence,
            sick_leave=sick_leave
        )
        
        return JsonResponse({'message': 'Permission request submitted successfully.'})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
#Course 
@csrf_exempt
def course_list(request):
    if request.method == 'GET':
        courses = Course.objects.all()
        data = list(courses.values())
        return JsonResponse(data, safe=False)

    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            course = Course.objects.create(
                college=data['college'],
                department=data['department'],
                name=data['name'],
                code=data['code'],
                duration=data['duration'],
                year=data['year'],
                prerequest=data['prerequest']
            )
            return JsonResponse({'message': 'Course created successfully'}, status=201)
        except KeyError as e:
            return JsonResponse({'error': f'Missing key in request data: {e}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        

@csrf_exempt
@require_http_methods(["GET", "PUT", "PATCH"])
def course_detail(request, pk):
    try:
        course = Course.objects.get(pk=pk)
    except Course.DoesNotExist:
        return JsonResponse({'message': 'Course not found'}, status=404)

    if request.method == 'GET':
        data = {
            'college': course.college,
            'department': course.department,
            'name': course.name,
            'code': course.code,
            'duration': course.duration,
            'year': course.year,
            'prerequest': course.prerequest
        }
        return JsonResponse(data)

    elif request.method == 'PUT':
        data = json.loads(request.body.decode('utf-8'))

        course.college = data['college']
        course.department = data['department']
        course.name = data['name']
        course.code = data['code']
        course.duration = data['duration']
        course.year = data['year']
        course.prerequest = data['prerequest']
        course.save()

        return JsonResponse({'message': 'Course updated successfully'})

    elif request.method == 'PATCH':
        data = json.loads(request.body.decode('utf-8'))

        # Update only the specified field if present in the request data
        if 'college' in data:
            course.college = data['college']
        elif 'department' in data:
            course.department = data['department']
        elif 'name' in data:
            course.name = data['name']
        elif 'code' in data:
            course.code = data['code']
        elif 'duration' in data:
            course.duration = data['duration']
        elif 'year' in data:
            course.year = data['year']
        elif 'prerequest' in data:
            course.prerequest = data['prerequest']

        course.save()

        return JsonResponse({'message': 'Course updated successfully'})
    

@csrf_exempt
def approve_permission(request, attendance_id):
    try:
        attendance = Attendance.objects.get(id=attendance_id)
        attendance.status = "Present"  # Update status to Present when approved
        attendance.save()

        # Create notification for the student
        Notification.objects.create(
            title="Permission Approved",
            message="Your permission request has been approved.",
            status="unread",
            link="/your-profile-page",  # Change this to the appropriate link
            user=attendance.student.user
        )

        return JsonResponse({'message': 'Permission approved successfully'})
    except Attendance.DoesNotExist:
        return JsonResponse({'error': 'Attendance record not found'}, status=404)

@csrf_exempt
def reject_permission(request, attendance_id):
    try:
        attendance = Attendance.objects.get(id=attendance_id)
        attendance.status = "Absent"  # Update status to Absent when rejected
        attendance.save()

        # Create notification for the student
        Notification.objects.create(
            title="Permission Rejected",
            message="Your permission request has been rejected.",
            status="unread",
            link="/your-profile-page",  # Change this to the appropriate link
            user=attendance.student.user
        )

        return JsonResponse({'message': 'Permission rejected successfully'})
    except Attendance.DoesNotExist:
        return JsonResponse({'error': 'Attendance record not found'}, status=404)
    


class UserList(APIView):
    def get(self, request):
        users = CustomUser.objects.all()
        data = [{'name': user.name, 'user_type': user.user_type} for user in users]
        return Response(data)

# API endpoint to fetch data from Attendance model
class AttendanceList(APIView):
    def get(self, request):
        attendance_records = Attendance.objects.all()
        data = [{'date': record.date, 'status': record.status, 'student_id': record.student_id} for record in attendance_records]
        print("attendancelist")
        print(attendance_records)
        return Response(data)

# API endpoint to fetch data from Course model
class CourseList(APIView):
    def get(self, request):
        courses = Course.objects.all()
        data = [{'course_name': course.course_name, 'teacher_id': course.teacher_id, 'schedule_id': course.schedule_id} for course in courses]
        return Response(data)

class AttendanceRecordList(generics.ListAPIView):
    queryset = AttendanceRecord.objects.all()
    serializer_class = AttendanceRecordSerializer
class AttendanceCreate(APIView):
    def post(self, request, format=None):
        serializer = AttendanceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)     
class AttendanceListByStatus(generics.ListAPIView):
    serializer_class = AttendanceSerializer  
    def get(self, request):
        attendance_records = Attendance.objects.all()
        data = [{'date': record.date, 'status': record.status} for record in attendance_records]
        return Response(data) 


class UserListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = CustomUser.objects.all()
        serializer = CustomUserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = CustomUserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

class UserDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        try:
            return CustomUser.objects.get(pk=pk)
        except CustomUser.DoesNotExist:
            return None

    def get(self, request, pk):
        user = self.get_object(pk)
        if user is None:
            return Response(status=404)
        serializer = CustomUserSerializer(user)
        return Response(serializer.data)

    def put(self, request, pk):
        user = self.get_object(pk)
        if user is None:
            return Response(status=404)
        serializer = CustomUserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    def delete(self, request, pk):
        user = self.get_object(pk)
        if user is None:
            return Response(status=404)
        user.delete()
        return Response(status=204)
    


@csrf_exempt
def add_teacher(request):
    if request.method == 'POST':
        try:
            # Print the data received from the frontend
            print('Data received from frontend:', request.POST)

            # Retrieve data from the POST request
            email = request.POST.get('email')
            name = request.POST.get('name')
            phone_number = request.POST.get('phone_number')
            gender = request.POST.get('gender')
            department = request.POST.get('department')
            college = request.POST.get('college')
            qualifications = request.POST.get('qualifications')
            semester = request.POST.get('semester')
            profile_picture = request.FILES.get('profile_picture')

            # Ensure that all required fields are present
            if not (email and name and phone_number and gender and department and college and qualifications and semester and profile_picture):
                return JsonResponse({'error': 'Missing required fields'}, status=400)

            # Create a CustomUser instance
            user = CustomUser.objects.create(
                email=email,
                name=name,
                phone_number=phone_number,
                department=department,
                college=college,
                gender=gender,
                role='teacher'
            )

            # Create a Teacher instance and assign values to its fields
            teacher = Teacher.objects.create(
                user=user,
                qualifications=qualifications,
                semester=semester,
                profile_picture=profile_picture
            )

            return JsonResponse({'message': 'Teacher added successfully'}, status=201)
        except IntegrityError as e:
            logger.error(f"IntegrityError occurred: {e}")
            return JsonResponse({'error': 'IntegrityError occurred. Please check if the data is valid.'}, status=400)
        except DatabaseError as e:
            logger.error(f"DatabaseError occurred: {e}")
            return JsonResponse({'error': 'DatabaseError occurred. Please try again later.'}, status=500)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return JsonResponse({'error': 'An unexpected error occurred. Please try again later.'}, status=500)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

@csrf_exempt
def edit_teacher(request, id):
    try:
        teacher = Teacher.objects.get(user_id=id)  # Retrieve Teacher instance by user_id
    except Teacher.DoesNotExist:
        return JsonResponse({'error': 'Teacher not found'}, status=404)

    if request.method == 'PUT':
        put_data = json.loads(request.body.decode('utf-8'))
        try:
            # Parse the form data from the request
            # put_data = request.POST

            # Extracting data from the request
            name = put_data.get('name')
            email = put_data.get('email')
            phone_number = put_data.get('phone_number')
            qualifications = put_data.get('qualifications')
            semester = put_data.get('semester')

            # Update teacher's information
            teacher.user.name = name
            teacher.user.email = email
            teacher.user.phone_number = phone_number
            teacher.user.save()

            # Update additional fields in Teacher model
            teacher.qualifications = qualifications
            teacher.semester = semester
            teacher.save()
            
            return JsonResponse({'message': 'Teacher updated successfully'}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    elif request.method == 'PATCH':
        patch_data = json.loads(request.body.decode('utf-8'))
        try:
            # Parse the form data from the request
            # patch_data = request.POST

            # Extracting data from the request
            name = patch_data.get('name')
            email = patch_data.get('email')
            phone_number = patch_data.get('phone_number')
            qualifications = patch_data.get('qualifications')
            semester = patch_data.get('semester')

            # Update teacher's information if the data is provided
            if name:
                teacher.user.name = name
            if email:
                teacher.user.email = email
            if phone_number:
                teacher.user.phone_number = phone_number
            if qualifications:
                teacher.qualifications = qualifications
            if semester:
                teacher.semester = semester

            # Save the changes
            teacher.user.save()
            teacher.save()

            return JsonResponse({'message': 'Teacher updated successfully'}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Only PUT or PATCH requests are allowed'}, status=405)

@csrf_exempt
def delete_teacher(request, id):
    try:
        teacher = Teacher.objects.get(user_id=id)
    except Teacher.DoesNotExist:
        return JsonResponse({'error': 'Teacher not found'}, status=404)

    if request.method == 'DELETE':
        # Delete the associated Teacher object
        teacher.delete()

        # Delete the CustomUser object
        teacher.user.delete()

        return JsonResponse({'message': 'Teacher deleted successfully'}, status=200)
    else:
        return JsonResponse({'error': 'Only DELETE requests are allowed'}, status=405)

@csrf_exempt
def add_schedule(request):
    if request.method == 'POST':
        # Extract schedule data from the POST request
        course_id = request.POST.get('course_id')
        day_of_the_week = request.POST.get('day_of_the_week')
        start_time = request.POST.get('start_time')
        end_time = request.POST.get('end_time')

        # Create a new Schedule object with the extracted data
        schedule = Schedule(course_id=course_id, day_of_the_week=day_of_the_week, 
                            start_time=start_time, end_time=end_time)

        # Save the Schedule object to the database
        schedule.save()

        # Return a JSON response indicating success
        return JsonResponse({'message': 'Schedule added successfully'}, status=201)

    else:
        # Handle the case where the request method is not POST
        # This can happen if someone tries to access the view via GET or other methods
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    # return render(request, 'add_schedule.html')


def display_schedule(request):
    user = request.user

    # Initialize queryset for schedules
    schedules = Schedule.objects.all()

    # Filter schedules based on user type
    if user.is_authenticated:
        if user.user_type == 'teacher':
            # Filter schedules for teacher based on courses taught
            schedules = Schedule.objects.filter(course__teacher__user=user)
        elif user.user_type == 'student':
            # Filter schedules for student based on section courses
                schedules = Schedule.objects.filter(course__sections__students=user)

    # Annotate schedule queryset with section/course name and teacher name
    schedules = schedules.annotate(
        section_course_name=Case(
            When(course__sections__isnull=False, then=F('course__sections__name')),
            default=F('course__name'),
            output_field=CharField(),
        ),
        teacher_name=F('course__teacher__user__name')
    )

    # Serialize schedule data
    schedule_data = []
    for schedule in schedules:
        start_time = schedule.start_time.strftime('%H:%M:%S') if schedule.start_time else None
        end_time = schedule.end_time.strftime('%H:%M:%S') if schedule.end_time else None
        schedule_data.append({
            'id': schedule.id,
            'section_course_name': schedule.section_course_name,
            'teacher_name': schedule.teacher_name,
            'day_of_the_week': schedule.day_of_the_week,
            'start_time': start_time,
            'end_time': end_time
        })

    # Return JSON response with schedule data
    return JsonResponse({'schedules': schedule_data})

@login_required
def edit_schedule(request, schedule_id):
    schedule = get_object_or_404(Schedule, id=schedule_id)
    user = request.user
    if user.user_type != 'teacher':
        messages.error(request, "You don't have permission to edit this schedule.")
        return redirect('display_schedule')

    form = ScheduleForm(request.POST or None, instance=schedule)
    if form.is_valid():
        form.save()
        messages.success(request, "Schedule updated successfully.")
        return redirect('display_schedule')

    context = {
        'form': form,
        'schedule': schedule
    }
    return render(request, 'edit_schedule.html', context)

@login_required
@csrf_exempt
def delete_schedule(request, schedule_id):
    schedule = get_object_or_404(Schedule, id=schedule_id)
    user = request.user
    if user.user_type != 'teacher':
        messages.error(request, "You don't have permission to delete this schedule.")
        return redirect('display_schedule')

    if request.method == 'POST':
        schedule.delete()
        messages.success(request, "Schedule deleted successfully.")
        return redirect('display_schedule')

    context = {
        'schedule': schedule
    }
    return render(request, 'delete_schedule.html', context)

# reminder for schedule

@csrf_exempt
def send_schedule_reminder(request):
    # Get the current datetime
    current_datetime = timezone.now()

    # Define the duration before the schedule that the reminder should be sent
    reminder_duration = timedelta(hours=1)  # Adjust as needed

    # Get schedules that are nearing
    nearing_schedules = Schedule.objects.filter(start_time__gt=current_datetime, start_time__lte=current_datetime + reminder_duration)

    # Iterate over nearing schedules
    for schedule in nearing_schedules:
        # Send reminders to associated teachers
        teachers = Teacher.objects.filter(course__schedules=schedule)
        for teacher in teachers:
            send_email_to_teacher(teacher.user.email, schedule)

        # Send reminders to associated students
        students = Student.objects.filter(course__schedules=schedule)
        for student in students:
            send_email_to_student(student.user.email, schedule)

    return JsonResponse({'message': 'Schedule reminders sent successfully'}, status=200)

def send_email_to_teacher(email, schedule):
    # Customize the email subject and content for teachers
    subject = f"Reminder: Your class is about to start"
    message = f"Dear Teacher,\n\nThis is a reminder that your class is about to start.\n\nSchedule Details:\nDay: {schedule.day_of_the_week}\nStart Time: {schedule.start_time}\nEnd Time: {schedule.end_time}\n\nBest Regards,\nYour School"

    # Send email
    send_mail(subject, message, settings.EMAIL_HOST_USER, ['meradongwook@gmail.com', 'kynthia369@gmail.com'], fail_silently=False)

def send_email_to_student(email, schedule):
    # Customize the email subject and content for students
    subject = f"Reminder: Your class is about to start"
    message = f"Dear Student,\n\nThis is a reminder that your class is about to start.\n\nSchedule Details:\nDay: {schedule.day_of_the_week}\nStart Time: {schedule.start_time}\nEnd Time: {schedule.end_time}\n\nBest Regards,\nYour School"

    # Send email
    send_mail(subject, message, settings.EMAIL_HOST_USER, ['meradongwook@gmail.com', 'kynthia369@gmail.com'], fail_silently=False)
# views.py
@csrf_exempt
def handle_uploaded_file(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

    print("Column Names:", df.columns)  # Print column names for debugging

    # Ensure columns are as expected
    expected_columns = ['Course Code', 'Course Name', 'Day', 'Start Time', 'End Time', 'Location', 'Instructor']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("Invalid column names in the file")

    return df

@csrf_exempt
def upload_schedule(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file part in the request"}, status=400)

        file = request.FILES['file']
        try:
            df = handle_uploaded_file(file)
            for _, row in df.iterrows():
                course, _ = Course.objects.get_or_create(code=row['Course Code'], defaults={'name': row['Course Name']})
                
                # Check if location and instructor fields are present in the DataFrame
                if 'Location' in df.columns and 'Instructor' in df.columns:
                    location = row['Location']
                    instructor = row['Instructor']
                    print("Location:", location)
                    print("Instructor:", instructor)
                    schedule = Schedule(
                        day_of_the_week=row['Day'],
                        start_time=row['Start Time'],
                        end_time=row['End Time'],
                        location=location,
                        instructor=instructor,
                        course=course
                    )
                else:
                    # If location and instructor fields are not present, create Schedule without them
                    schedule = Schedule(
                        day_of_the_week=row['Day'],
                        start_time=row['Start Time'],
                        end_time=row['End Time'],
                        course=course
                    )
                    
                schedule.save()
                
            return JsonResponse({"message": "Schedules uploaded successfully."}, status=201)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request"}, status=400)
