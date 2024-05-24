import React from 'react';

const AttendanceItem = ({ date, checkIn, checkOut, totalHours, notes, itemIndex }) => {
  const section = 'C';
  const totalStudents = 25;

  const formattedDate = new Date(date).toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <div className="bg-white p-4 shadow-md mb-4 border-blue-300 border-l-4">
      <div className="mb-4">
        <p className="text-lg font-bold text-gray-500">{formattedDate}</p>
      </div>
      <hr className="my-2 border-gray-200" />
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-xs mb-2 font-bold text-gray-400">Recorded time</p>
          <p className="font-bold">{checkIn}</p>
        </div>
        <div>
          <p className="text-xs font-bold mb-2 text-gray-400">Section</p>
          <p className="font-bold ml-3">{section}</p>
        </div>
        <div>
          <p className="text-xs font-bold mb-2 text-gray-400">Students</p>
          <p className="font-bold ml-3">{totalStudents}</p>
        </div>
      </div>
      <hr className="my-4 border-gray-200" />
      <div className="mt-2 flex">
        <p className="text-sm text-black font-bold">Notes: </p>
        <p className="text-sm text-gray-400 ml-1">{notes}</p>
      </div>
    </div>
  );
}; export default  AttendanceItem;