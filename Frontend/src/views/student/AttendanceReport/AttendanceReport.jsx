import React from 'react';
import { BrowserRouter as Router, Route, Switch, Link, useParams } from 'react-router-dom';
import './AttendanceReport.css';

const CourseSelection = () => {
    return (
        <div className="container">
            <h1>Select Course</h1>
            <div className="course-list">
                <Link to="/report/course1" className="course-item">
                    <h2>Course Name 1</h2>
                    <p>Course Code 1</p>
                </Link>
                <Link to="/report/course2" className="course-item">
                    <h2>Course Name 2</h2>
                    <p>Course Code 2</p>
                </Link>
                <Link to="/report/course3" className="course-item">
                    <h2>Course Name 3</h2>
                    <p>Course Code 3</p>
                </Link>
                {/* Add more course items as needed */}
            </div>
        </div>
    );
}


const ReportDisplay = () => {
  const { courseName } = useParams();

  // Sample attendance data for demonstration
  const attendanceData = [
      { date: '2024-04-01', status: 'Present' },
      { date: '2024-04-02', status: 'Absent' },
      { date: '2024-04-03', status: 'Present' },
      // Add more attendance data as needed
  ];

  return (
      <div className="container">
          <h1>Attendance Report for {courseName}</h1>
          <div className="report-container">
            
              <div className="report-card">
                  <table className="report-table">
                      <thead>
                          <tr>
                              <th>Date</th>
                              <th>Status</th>
                          </tr>
                      </thead>
                      <tbody>
                          {attendanceData.map((attendance, index) => (
                              <tr key={index}>
                                  <td>{attendance.date}</td>
                                  <td>{attendance.status}</td>
                              </tr>
                          ))}
                      </tbody>
                  </table>
              </div>
              <Link to="/" className="back-link">Back to Select Course</Link> {/* Link back to the select course page */}
          </div>
          
      </div>
  );
}

const AttendanceReport = () => {
    return (
        <Router>
            <Switch>
                <Route exact path="/" component={CourseSelection} />
                <Route path="/report/:courseName" component={ReportDisplay} />
            </Switch>
        </Router>
    );
}

export default AttendanceReport;