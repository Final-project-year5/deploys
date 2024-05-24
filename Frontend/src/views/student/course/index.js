import React, { useState } from 'react';
import './CoursePage.css';

function CoursePage() {
  const [selectedCourse, setSelectedCourse] = useState(null);

  const courses = [
    {
      id: 1,
      title: 'Course 1',
      syllabus: 'Course 1 syllabus details...',
      instructor: 'Instructor 1',
      schedule: 'Course 1 schedule...',
    },
    {
      id: 2,
      title: 'Course 2',
      syllabus: 'Course 2 syllabus details...',
      instructor: 'Instructor 2',
      schedule: 'Course 2 schedule...',
    },
    {
      id: 3,
      title: 'Course 3',
      syllabus: 'Course 3 syllabus details...',
      instructor: 'Instructor 3',
      schedule: 'Course 3 schedule...',
    },
    {
      id: 4,
      title: 'Course 4',
      syllabus: 'Course 4 syllabus details...',
      instructor: 'Instructor 4',
      schedule: 'Course 4 schedule...',
    },
    {
      id: 5,
      title: 'Course 5',
      syllabus: 'Course 5 syllabus details...',
      instructor: 'Instructor 5',
      schedule: 'Course 5 schedule...',
    },
  ];

  const handleCourseClick = (courseId) => {
    setSelectedCourse(courseId);
  };

  return (
    <div className="course-page">
      <div className="course-list">
        <h1>Course List</h1>
        <ul>
          {courses.map((course) => (
            <li
              key={course.id}
              onClick={() => handleCourseClick(course.id)}
              className={selectedCourse === course.id ? 'selected' : ''}
            >
              {course.title}
            </li>
          ))}
        </ul>
      </div>
      <div className="course-details">
        {selectedCourse && (
          <div>
            <h2>{courses.find((course) => course.id === selectedCourse).title}</h2>
            <div>
              <h3>Syllabus</h3>
              <p>{courses.find((course) => course.id === selectedCourse).syllabus}</p>
            </div>
            <div>
              <h3>Instructor Information</h3>
              <p>{courses.find((course) => course.id === selectedCourse).instructor}</p>
            </div>
            <div>
              <h3>Course Schedule</h3>
              <p>{courses.find((course) => course.id === selectedCourse).schedule}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CoursePage;