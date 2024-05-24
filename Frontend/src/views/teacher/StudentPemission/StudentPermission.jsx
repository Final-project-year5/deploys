import React from "react";
// import { useHistory } from "react--dom";
import { useHistory } from "react-router-dom";

import { MdOutlineEdit } from "react-icons/md"; // Import MdOutlineEdit icon

const StudentPermission = (props) => {
  const courseName = "Database Management";
  const { ...rest } = props;
  const history = useHistory();

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  const getCurrentMonthAndYear = () => {
    const months = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    const date = new Date();
    const month = months[date.getMonth()];
    const year = date.getFullYear();
    return `${month}, ${year}`;
  };

  const samplePermissionData = [
    {
      no: 1,
      firstName: "meron",
      lastName: "abera",
      ID: "ets 0451/12",
      Section: "C",
      submittedDate: today,
      status: "pending",
    },
    {
      no: 2,
      firstName: "meron",
      lastName: "edea",
      ID: "ets045o/12",
      Section: "C",
      submittedDate: today,
      status: "approved",
    },
    {
      no: 3,
      firstName: "tsedey",
      lastName: "abera",
      ID: "ets5451/12",
      Section: "C",
      submittedDate: today,
      status: "rejected",
    },
  ];

  const handleEditClick = () => {
    console.log("Edit button clicked");
    history.push("/view-permission");
  };

  return (
    <div className="container mx-auto">
      
      <div className="mt-20 mb-4 flex items-center">
        <p className="text-sm font-bold text-gray-400 mr-4">
          {getCurrentMonthAndYear()}
        </p>
        <hr className="flex-grow border-gray-300" />
      </div>

      <div className="mt-8">
        <h2 className="text-lg font-bold mb-2">Student List</h2>
        <div className="p-6 px-0 overflow-scroll">
          <table className="w-full mt-4 text-left table-auto min-w-max">
            <thead>
              <tr>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  No
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  First Name
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  Last Name
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  ID
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  Section
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  Submitted Date
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  Status
                </th>
                <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                  Actions
                </th>{" "}
                {/* Add Actions column */}
              </tr>
            </thead>
            <tbody>
              {samplePermissionData.map((permission, index) => (
                <tr
                  key={index}
                  className={index % 2 === 0 ? "bg-gray-100" : ""}
                >
                  <td className="p-4 border-b border-blue-gray-50">
                    {permission.no}
                  </td>
                  <td className="p-4 border-b border-blue-gray-50">
                    {permission.firstName}
                  </td>
                  <td className="p-4 border-b border-blue-gray-50">
                    {permission.lastName}
                  </td>
                  <td className="p-4 border-b border-blue-gray-50">
                    {permission.ID}
                  </td>
                  <td className="p-4 border-b border-blue-gray-50 ml-8">
                    {permission.Section}
                  </td>
                  <td className="p-4 border-b border-blue-gray-50">
                    {permission.submittedDate}
                  </td>
                  <td className={`p-4 border-b border-blue-gray-50 ml-8 w-max`}>
                    <div
                      className={`relative grid items-center px-2 py-1 font-sans text-xs font-bold uppercase rounded-md select-none whitespace-nowrap ${
                        permission.status === "pending"
                          ? "bg-blue-200 text-blue-900"
                          : permission.status === "approved"
                          ? "bg-green-200 text-green-900"
                          : permission.status === "rejected"
                          ? "bg-red-200 text-red-900"
                          : ""
                      }`}
                    >
                      <span>{permission.status}</span>
                    </div>
                  </td>
                  <td className="p-4 border-b border-blue-gray-50">
                    <button
                      className="p-2 hover:bg-blue-300 text-white font-bold py-1 px-2 rounded-full focus:outline-none focus:shadow-outline"
                      onClick={handleEditClick}
                    >
                      <MdOutlineEdit size={20} color="blue" />{" "}
                      {/* Render the MdOutlineEdit icon */}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default StudentPermission;
