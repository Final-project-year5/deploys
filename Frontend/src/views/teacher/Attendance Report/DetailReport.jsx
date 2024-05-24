import React, { useState } from "react";
import Sidebar from "../../../components/sidebar/Sidebar";
import { SidebarContext } from "../../../contexts/SidebarContext";
import routes from "routes.js";
import {
  Portal,
  Box,
  useDisclosure,
  
} from "@chakra-ui/react";
import Navbar from "components/navbar/NavbarAdmin.js";

const DetailReport = (props) => {
  // Get today's date in the specified format
  const courseName = "Database Management";
  const totalAttendance = 65;
  const { ...rest } = props;
  const [fixed] = useState(false);

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

  // Sample student data
  const sampleStudentData = [
    {
      firstName: "meron",
      lastName: "abera",
      ID: "ets 0451/12",
      status: "present",
    },
    {
      firstName: "tsedet",
      lastName: "mekonnen",
      ID: "ets0450/12",
      status: "absent",
    },
    {
      firstName: "meron",
      lastName: "edea",
      ID: "ets1050/12",
      status: "permission",
    },
    {
      firstName: "kaleb",
      lastName: "abera",
      ID: "ets 0451/12",
      status: "present",
    },
    {
      firstName: "joni",
      lastName: "mekonnen",
      ID: "ets0450/12",
      status: "absent",
    },
    {
      firstName: "migbar",
      lastName: "edea",
      ID: "ets1050/12",
      status: "permission",
    },
    {
      firstName: "meron",
      lastName: "abera",
      ID: "ets 0451/12",
      status: "present",
    },
    {
      firstName: "tsedet",
      lastName: "mekonnen",
      ID: "ets0450/12",
      status: "absent",
    },
    {
      firstName: "meron",
      lastName: "edea",
      ID: "ets1050/12",
      status: "permission",
    },
    {
      firstName: "meron",
      lastName: "abera",
      ID: "ets 0451/12",
      status: "present",
    },
    {
      firstName: "tsedet",
      lastName: "mekonnen",
      ID: "ets0450/12",
      status: "absent",
    },
    {
      firstName: "meron",
      lastName: "edea",
      ID: "ets1050/12",
      status: "permission",
    },
  ];
  const getActiveNavbar = (routes) => {
    let activeNavbar = false;
    for (let i = 0; i < routes.length; i++) {
      if (routes[i].collapse) {
        let collapseActiveNavbar = getActiveNavbar(routes[i].items);
        if (collapseActiveNavbar !== activeNavbar) {
          return collapseActiveNavbar;
        }
      } else if (routes[i].category) {
        let categoryActiveNavbar = getActiveNavbar(routes[i].items);
        if (categoryActiveNavbar !== activeNavbar) {
          return categoryActiveNavbar;
        }
      } else {
        if (
          window.location.href.indexOf(routes[i].layout + routes[i].path) !== -1
        ) {
          return routes[i].secondary;
        }
      }
    }
    return activeNavbar;
  };
  const getActiveNavbarText = (routes) => {
    let activeNavbar = false;
    for (let i = 0; i < routes.length; i++) {
      if (routes[i].collapse) {
        let collapseActiveNavbar = getActiveNavbarText(routes[i].items);
        if (collapseActiveNavbar !== activeNavbar) {
          return collapseActiveNavbar;
        }
      } else if (routes[i].category) {
        let categoryActiveNavbar = getActiveNavbarText(routes[i].items);
        if (categoryActiveNavbar !== activeNavbar) {
          return categoryActiveNavbar;
        }
      } else {
        if (
          window.location.href.indexOf(routes[i].layout + routes[i].path) !== -1
        ) {
          return routes[i].messageNavbar;
        }
      }
    }
    return activeNavbar;
  };
  const getActiveRoute = (routes) => {
    let activeRoute = "Default Brand Text";
    for (let i = 0; i < routes.length; i++) {
      if (routes[i].collapse) {
        let collapseActiveRoute = getActiveRoute(routes[i].items);
        if (collapseActiveRoute !== activeRoute) {
          return collapseActiveRoute;
        }
      } else if (routes[i].category) {
        let categoryActiveRoute = getActiveRoute(routes[i].items);
        if (categoryActiveRoute !== activeRoute) {
          return categoryActiveRoute;
        }
      } else {
        if (
          window.location.href.indexOf(routes[i].layout + routes[i].path) !== -1
        ) {
          return routes[i].name;
        }
      }
    }
  };
  // Sort students by first name
  sampleStudentData.sort((a, b) => {
    const firstNameA = a.firstName.toLowerCase();
    const firstNameB = b.firstName.toLowerCase();
    if (firstNameA < firstNameB) return -1;
    if (firstNameA > firstNameB) return 1;
    return 0;
  });
  document.documentElement.dir = "ltr";
  const { onOpen } = useDisclosure();
  document.documentElement.dir = "ltr";
  return (
    <Box marginRight="10">
      <Box>
        <SidebarContext.Provider value={{ isOpen: true, toggleOpen: () => {} }}>
          <Sidebar routes={routes} display="none" {...rest} />
        </SidebarContext.Provider>
        <Box marginLeft="80">
          <Portal>
            <Box>
              <Navbar
                onOpen={onOpen}
                brandText={getActiveRoute(routes)}
                secondary={getActiveNavbar(routes)}
                message={getActiveNavbarText(routes)}
                fixed={fixed}
                {...rest}
              />
            </Box>
          </Portal>
          <div className="container mx-auto">
            <div className="flex justify-between">
              <div>
                <h1 className="text-2xl mt-24 mb-2 font-bold">
                  Attendance History
                </h1>
                <p className="text-sm ml-2 font-bold text-gray-400 mb-4 ">
                  Today {today}
                </p>
              </div>
              <div className="grid grid-cols-3 mt-8 gap-4">
                <div>
                  <h2 className="text-sm mt-24 font-bold text-gray-400">
                    Course Name
                  </h2>
                  <p className=" text-sm font-bold">{courseName}</p>
                </div>
                <div className="flex flex-col items-end">
                  <h2 className="text-sm mt-24 font-bold text-gray-400">
                    Total Attendance
                  </h2>
                  <p className=" text-sm mr-14 font-bold">{totalAttendance}</p>
                </div>
              </div>
            </div>
            <div className="mt-8 mb-4 flex items-center">
              <p className="text-sm font-bold text-gray-400 mr-4">
                {getCurrentMonthAndYear()}
              </p>
              <hr className="flex-grow border-gray-300" />
            </div>

            <div className="mt-8">
              <h2 className="text-lg font-bold mb-2">Student List</h2>
              <p className="text-xs text-gray-500 mt-2">
                Recorded time: {new Date().toLocaleTimeString()}
              </p>

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
                        Status
                      </th>
                      <th className="p-4 border-y border-blue-gray-100 bg-blue-gray-50/50">
                        Date
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {sampleStudentData.map((student, index) => (
                      <tr
                        key={index}
                        className={index % 2 === 0 ? "bg-gray-100" : ""}
                      >
                        <td className="p-4 border-b border-blue-gray-50">
                          {index + 1}.
                        </td>
                        <td className="p-4 border-b border-blue-gray-50">
                          {student.firstName}
                        </td>
                        <td className="p-4 border-b border-blue-gray-50">
                          {student.lastName}
                        </td>
                        <td className="p-4 border-b border-blue-gray-50">
                          {student.ID}
                        </td>
                        <td className="p-4 border-b border-blue-gray-50">
                          <div className="w-max">
                            <div
                              className={`relative grid items-center px-2 py-1 font-sans text-xs font-bold uppercase rounded-md select-none whitespace-nowrap ${
                                student.status === "present"
                                  ? "bg-green-200 text-green-900"
                                  : student.status === "absent"
                                  ? "bg-red-200 text-red-900"
                                  : student.status === "permission"
                                  ? "bg-blue-200 text-blue-900"
                                  : ""
                              }`}
                            >
                              <span>{student.status}</span>
                            </div>
                          </div>
                        </td>
                        <td className="p-4 border-b border-blue-gray-50">
                          {today}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </Box>
      </Box>
    </Box>
  );
};

export default DetailReport;

