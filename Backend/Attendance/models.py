from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.core.validators import RegexValidator

# Custom user manager for the custom user model
class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, role='student', **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user_type = extra_fields.pop('user_type', None)
        semester = extra_fields.pop('semester', None)
        qualification = extra_fields.pop('qualification', None)
        user = self.model(email=email, role=role, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        if user_type == 'teacher' and semester:
            Teacher.objects.create(user=user, qualifications=qualification, semester=semester)
        return user

    def create_superuser(self, email, password=None, role='admin', **extra_fields):
        if role != 'admin':
            raise ValueError('Superuser must have role="admin".')

        return self.create_user(email, password=password, role=role, **extra_fields)

# Custom user model extending AbstractBaseUser
class CustomUser(AbstractBaseUser):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=100, validators=[RegexValidator(regex=r'^\+?1?\d{9,15}$', message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")])
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female')], blank=True)
    department = models.CharField(max_length=200,default=None)
    college = models.CharField(max_length=200,default=None)
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('teacher', 'Teacher'),
        ('student', 'Student'),
    )
    role = models.CharField(max_length=50, choices=ROLE_CHOICES, default='student')
    token = models.CharField(max_length=255, blank=True, null=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']

    def __str__(self):
        return self.email

# Teacher Model
class Teacher(models.Model): 
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    qualifications = models.CharField(max_length=200, blank=True)
    semester = models.CharField(max_length=50, default=None) 
    profile_picture = models.ImageField(upload_to='teacher_profiles/', blank=True)

    def __str__(self):
        return self.user.name 

# Section Model
class Section(models.Model):  
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name

# FaceData Model
class FaceData(models.Model):
    face_encoding = models.FileField(upload_to='facedata')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

# Student Model
class Student(models.Model):
    student_id = models.CharField(max_length=100, unique=True, default=None)
    section = models.ForeignKey(Section, on_delete=models.CASCADE)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, default=None)
    year_semester = models.CharField(max_length=50, default=None)

    def __str__(self):
        return self.name

# Course Model
class Course(models.Model):
    college = models.CharField(max_length=100, default=None)
    department = models.CharField(max_length=100, default=None)
    name = models.CharField(max_length=100, default=None)
    code = models.CharField(max_length=10, default=None)
    duration = models.CharField(max_length=50, default=None)
    year = models.CharField(max_length=10, default=None)
    prerequest = models.CharField(max_length=100, default=None)
    sections = models.ManyToManyField(Section, related_name='courses')

    def __str__(self):
        return self.course_name

# Schedule Model
class Schedule(models.Model):
    day_of_the_week = models.CharField(max_length=50)
    start_time = models.TimeField()
    end_time = models.TimeField()
    location = models.CharField(max_length=100)  # Add location field
    instructor = models.CharField(max_length=100)  # Add instructor field
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='schedules')

    def __str__(self):
        return f"{self.day_of_the_week} ({self.start_time} - {self.end_time}) - Course: {self.course}"

# Notification Model
class Notification(models.Model):
    title = models.CharField(max_length=100)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50)
    link = models.URLField()
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

# Attendance Model
class Attendance(models.Model):
    date = models.DateField()
    status = models.CharField(max_length=50)
    student = models.ForeignKey(Student, on_delete=models.CASCADE, default=None)

# CourseStudent Model
class CourseStudent(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)

# Permission Model
class Permission(models.Model):
    user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    teacher_id = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    reason = models.CharField(max_length=255)
    evidence = models.FileField(upload_to='permissions')


# Activity log Model
class ActivityLog(models.Model):
    name = models.CharField(max_length=100)
    action = models.CharField(max_length=100)
    resource = models.CharField(max_length=100)
    details = models.TextField()
    ip_address = models.CharField(max_length=50)
    status_code = models.PositiveIntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.action} - {self.resource} - {self.timestamp}" 

#Attendance record model
class AttendanceRecord(models.Model):
    date = models.DateField()
    check_in = models.TimeField()
    check_out = models.TimeField()
    total_hours = models.DecimalField(max_digits=5, decimal_places=2)
    notes = models.TextField()

    def str(self):
        return f"{self.date} - {self.check_in}"
    
#Permission request model
class PermissionRequest(models.Model):
    teacher = models.CharField(max_length=100)
    reason = models.TextField()
    evidence = models.FileField(upload_to='evidence/')
    sick_leave = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)