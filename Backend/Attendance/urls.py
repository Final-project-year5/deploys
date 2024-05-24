from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.home, name="home"),
    path('api/register/', views.register_user, name='register_user'),
    path('api/login/', views.login_user, name='login_user'),
    path('api/activity-logs/', views.get_activity_logs, name='get_activity_logs'),
    path('api/profile/', views.user_profile, name='user_profile'),
    path('api/updateprofile/', views.update_profile, name='update_profile'),
    path('api/generate_otp/', views.send_otp, name='send_otp'),
    path('api/update_password/', views.update_password, name='update_password'),
    path('api/create_permission/', views.create_permission_request, name='create_permission_request'),
    path('api/approve-permission/<int:attendance_id>/', views.approve_permission, name='approve_permission'),
    path('api/reject-permission/<int:attendance_id>/', views.reject_permission, name='reject_permission'),
    path('api/courses/', views.course_list, name='course-list'),
    path('api/courses/<int:pk>/', views.course_detail, name='course-detail'),
    # path('api/join-course/', views.join_course, name='join-course'),
     path("api/add_teacher/", views.add_teacher, name="add_teacher"),
     path("api/edit_teacher/<int:id>/", views.edit_teacher, name="edit_teacher"),
     path('api/display_schedule/', views.display_schedule, name='display_schedule_api'),
     path('api/send_schedule_reminder/', views.send_schedule_reminder, name='send_schedule_reminder'),
     path('api/delete_teacher/<str:id>/', views.delete_teacher, name='delete_teacher'),
     path('delete_schedule/<int:schedule_id>/', views.delete_schedule, name='delete_schedule'),
     path('add_schedule/', views.add_schedule, name='add_schedule'),
     path('api/delete_schedule/<str:id>/', views.delete_schedule, name='delete_schedule'),
     ##attendance
    # path('api/mark_attendance/', views.mark_attendance, name='mark_attendance'),
    # path('api/recognize/', views.recognize_face, name='recognize_face'),
    path('api/upload_schedule/', views.upload_schedule, name='upload_schedule'),


    path('api/attendance/', views.AttendanceRecordList.as_view(), name='attendance-list'),
    path('api/attendance/create/', views.AttendanceCreate.as_view(), name='attendance-create'),
    path('Users/', views.UserList.as_view(), name='user_list'),  
    path('api/Attendance/', views.AttendanceList.as_view(), name='attendance_list'), #returns attendance list
    path('Courses/', views.CourseList.as_view(), name='course_list'),
    path('', views.AttendanceRecordList.as_view(), name='attendance-list'),
    path('api/attendance/', views.AttendanceListByStatus.as_view(), name='attendance-list'),
    # path('users/', views.UserListView.as_view(), name='user-list'),
    # path('users/<int:pk>/', views.UserDetailView.as_view(), name='user-detail'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)