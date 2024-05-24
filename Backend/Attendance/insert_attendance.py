from datetime import date

from Attendance.models import Attendance, Student

# Assuming you have a student instance
student = Student.objects.get(pk=4)  # Replace 1 with the ID of the student

# Create an attendance record
attendance_record = Attendance.objects.create(
    date=date.today(),  # Today's date
    status='Present',   # Example status
    student=student     # Link the attendance record to the student
)

# Save the attendance record
attendance_record.save()