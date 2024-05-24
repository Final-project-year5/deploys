import React from 'react';
import './SchedulePage.css';

function SchedulePage() {
  const courses = [
    {
      id: 1,
      courseCode: 'course1',
      courseTitle: 'Introduction to English Literature',
      instructor: 'Professor Smith',
      schedule: 'MWF 9:00 AM - 10:30 AM',
    },
    {
      id: 2,
      courseCode: 'course2',
      courseTitle: 'Calculus II',
      instructor: 'Professor Johnson',
      schedule: 'TTH 1:00 PM - 2:30 PM',
    },
    {
      id: 3,
      courseCode: 'course3',
      courseTitle: 'World History: 20th Century',
      instructor: 'Professor Davis',
      schedule: 'MWF 11:00 AM - 12:30 PM',
    },
    { 
    id: 4,
    courseCode: 'course4',
    courseTitle: 'World History: 20th Century',
    instructor: 'Professor Davis',
    schedule: 'MWF 11:00 AM - 12:30 PM',
  },
    // Add more course objects as needed
  ];

  return (
    <div className="schedule-page">
      <h1>Schedule</h1>
      <table>
        <thead>
          <tr>
            <th>Course Code</th>
            <th>Course Title</th>
            <th>Instructor</th>
            <th>Schedule</th>
          </tr>
        </thead>
        <tbody>
          {courses.map((course) => (
            <tr key={course.id}>
              <td>{course.courseCode}</td>
              <td>{course.courseTitle}</td>
              <td>{course.instructor}</td>
              <td>{course.schedule}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default SchedulePage;