// AttendanceHistory.js
import React from "react";
import { Link } from 'react-router-dom'; // Import Link component
import AttendanceItem from "./AtendanceItem";

const AttendanceHistory = ({ attendanceData }) => {
  const courseName = "Database Management";
  const totalAttendance = 65;

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  const getCurrentMonthAndYear = () => {
    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ];
    const date = new Date();
    const month = months[date.getMonth()];
    const year = date.getFullYear();
    return `${month}, ${year}`;
  };

  return (
    <div className="container mx-auto">
      <div className="flex justify-between ">
        
        <div className="mt-24">
          <p className="text-l ml-2 font-bold text-gray-400 mb-4 ">
            Today {today}
          </p>
        </div>
        <div className="grid grid-cols-3 mt-8 gap-4">
          <div className="mt-12">
            <h2 className="text-sm font-bold text-gray-400">Course Name</h2>
            <p className="text-sm font-bold">{courseName}</p>
          </div>
          <div className="flex flex-col items-end mt-12">
            <h2 className="text-sm font-bold text-gray-400">
              Total Attendance
            </h2>
            <p className="text-sm mr-14 font-bold">{totalAttendance}</p>
          </div>
        </div>
      </div>
      <div className="mt-8 mb-4 flex items-center">
        <p className="text-sm font-bold text-gray-400 mr-4">
          {getCurrentMonthAndYear()}
        </p>
        <hr className="flex-grow border-gray-300" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {attendanceData && attendanceData.map((item, index) => (
          <Link key={index} to={`/detailed-report/${index}`}> {/* Update to include Link */}
            <AttendanceItem {...item} />
          </Link>
        ))}
      </div>
    </div>
  );
};

export default AttendanceHistory;
