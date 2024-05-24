import React from "react";

import { Icon } from "@chakra-ui/react";
import {
  MdPerson,
  MdHome,
  MdLock,
  MdSubject,
  MdEventNote,
  MdGroupWork, 
  MdCheckCircleOutline,
  MdOutlineSchedule,
  MdOutlinePeople,
} from "react-icons/md";

import Course from "views/teacher/courses";
import ActivityLog from "views/admin/activitylog/components/activitylog";
import Role from "views/admin/role/components/role";
import AttendanceHistory from "views/teacher/Attendance Report/AttendanceHistory";
import StudentPermission from "views/teacher/StudentPemission/StudentPermission"; 
import Schedule from "views/teacher/Schedule/ViewSchedule";

import Profile from "views/student/profile";
import Studentcourse from "views/student/course";
import studentschedule from "views/student/Schedule";
import Permission from "views/admin/Permission";

import AttendanceReport from "views/student/AttendanceReport/AttendanceReport";
import AdminHR from "views/admin/AdminHR/AdminHR";
import PermissionPage from "views/student/PermissionPage/PermissionPage";

import SignInCentered from "views/auth/signIn";
import {tableData} from "views/teacher/courses/variables/tableData";
import {activityData} from "views/admin/activitylog/variables/activityData";
import {roleData} from "views/admin/role/variables/roleData";

const attendanceData = [
  {
    date: "2024-03-25",
    checkIn: "09:00 AM",
    checkOut: "06:00 PM",
    totalHours: "8 hours",
    notes: "Worked on project X",
  },
  {
    date: "2024-03-26",
    checkIn: "09:00 AM",
    checkOut: "06:00 PM",
    totalHours: "8 hours",
    notes: "Worked on project X",
  },
  {
    date: "2024-03-26",
    checkIn: "09:00 AM",
    checkOut: "06:00 PM",
    totalHours: "8 hours",
    notes: "Worked on project X",
  },
];

const routes = [
  {
    name: "Main Dashboard",
    layout: "/admin",
    path: "/default",
    icon: <Icon as={MdHome} width='20px' height='20px' color='inherit' />,
    // component: MainDashboard,
  },
  {
    name: "Course",
    layout: "/admin",
    path: "/Course",
    icon: (
      <Icon
        as={MdSubject}
        width='20px'
        height='20px'
        color='inherit'
      />
    ),
    component: () => <Course courses={tableData} />,
  },
  {
    name: "Student Course",
    layout: "/admin",
    path: "/nft-marketplace",
    icon: (
      <Icon
        as={MdSubject}
        width='20px'
        height='20px'
        color='inherit'
      />
    ),
    component: Studentcourse,
    secondary: true,
    
  },
  {
    name: "Attendance report",
    layout: "/admin",
    icon: <Icon as={MdCheckCircleOutline} width='20px' height='20px' color='inherit' />,
    path: "/attendance-history",
    component: () => <AttendanceHistory attendanceData={attendanceData} />, // Removed component
  },
  {
    name: "Student Attendance report",
    layout: "/admin",
    icon: <Icon as={MdCheckCircleOutline} width='20px' height='20px' color='inherit' />,
    path: "/attendance-report",
    component: AttendanceReport, // Updated component
  },
  {
    name: "Admin HR",
    layout: "/admin",
    icon: <Icon as={MdCheckCircleOutline} width='20px' height='20px' color='inherit' />,
    path: "/admin-hr",
    component: AdminHR, // Updated component
  },
  {
    name: "Schedule",
    layout: "/admin",
    icon: <Icon as={MdOutlineSchedule} width='20px' height='20px' color='inherit' />,
    path: "/view-schedule",
    component: () => <Schedule />, // Removed component
  },
  {
    name: "Students Permission",
    layout: "/admin",
    icon: <Icon as={MdOutlinePeople} width='20px' height='20px' color='inherit' />,
    path: "/student-permissions",
    component: () => <StudentPermission />, // Removed component
  },
  { 
    name: "Permission Form",
    layout: "/admin",
    icon: <Icon as={MdCheckCircleOutline} width='20px' height='20px' color='inherit' />,
    path: "/permission-page",
    component: PermissionPage

   
  },
  {
    name: "Student Schedule",
    layout: "/admin",
    icon: <Icon as={MdOutlineSchedule} width='20px' height='20px' color='inherit' />,
    path: "/data-tables",
    component: studentschedule,
  },
  {
    name: "Profile",
    layout: "/admin",
    path: "/profile",
    icon: <Icon as={MdPerson} width='20px' height='20px' color='inherit' />,
    component: Profile,
  },
  {
    name: "Role",
    layout: "/admin",
    path: "/role",
    icon: (
      <Icon
        as={MdGroupWork }
        width='20px'
        height='20px'
        color='inherit'
      />
    ),
    component: () => <Role roleData={roleData} />,
  },
  {
    name: "Permission",
    layout: "/admin",
    path: "/Permission",
    icon: <Icon as={MdPerson} width='20px' height='20px' color='inherit' />,
    component: Permission,
  },
  {
    name: "Activity Log",
    layout: "/admin",
    path: "/activitylog",
    icon: (
      <Icon
        as={MdEventNote}
        width='20px'
        height='20px'
        color='inherit'
      />
    ),
    component: () => <ActivityLog activityData={activityData} />,
  },
  {
    name: "Sign In",
    layout: "/auth",
    path: "/sign-in",
    icon: <Icon as={MdLock} width='20px' height='20px' color='inherit' />,
    component: SignInCentered,
  },
];

export default routes;
