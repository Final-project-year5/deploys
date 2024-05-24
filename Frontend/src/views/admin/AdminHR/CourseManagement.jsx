import React, { useState } from 'react';
import './CourseManagement.css';
import Sidebar from '../../../components/sidebar/Sidebar';
import { SidebarContext } from '../../../contexts/SidebarContext';
import { MdOutlineEdit, MdMoreVert } from "react-icons/md"; // Import MdOutlineEdit and MdMoreVert icons
import { Portal,
  Box,
  useDisclosure, } from '@chakra-ui/react'; // Import Box component from Chakra UI
import routes from 'routes.js';
import Navbar from "components/navbar/NavbarAdmin.js";

const CourseManagement = (props) => {
  const [courses, setCourses] = useState([]);
  const [college, setCollege] = useState('');
  const [department, setDepartment] = useState('');
  const [courseName, setCourseName] = useState('');
  const [courseCode, setCourseCode] = useState('');
	const { ...rest } = props;
  const [fixed] = useState(false);
  
  const addCourse = () => {
    if (college.trim() !== '' && department.trim() !== '' && courseName.trim() !== '' && courseCode.trim() !== '') {
      const newCourse = { college, department, name: courseName, code: courseCode };
      setCourses([...courses, newCourse]);
      // Reset form fields after submission
      setCollege('');
      setDepartment('');
      setCourseName('');
      setCourseCode('');
    }
  };
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
  document.documentElement.dir = "ltr";
  const { onOpen } = useDisclosure();
  document.documentElement.dir = "ltr";
  return (
    <Box  marginRight='10'>
    <Box>
      <SidebarContext.Provider value={{ isOpen: true, toggleOpen: () => {} }}>
        <Sidebar routes={routes} display='none' {...rest} />
      </SidebarContext.Provider>
      <Box marginLeft='80'>
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
    <div className="course-management-page">
      <h2>Course Management</h2>
      <div className="add-course-form">
        <input
          type="text"
          value={college}
          onChange={(e) => setCollege(e.target.value)}
          placeholder="College"
          className="college-input"
        />
        <input
          type="text"
          value={department}
          onChange={(e) => setDepartment(e.target.value)}
          placeholder="Department"
          className="department-input"
        />
        <input
          type="text"
          value={courseName}
          onChange={(e) => setCourseName(e.target.value)}
          placeholder="Course Name"
          className="course-name-input"
        />
        <input
          type="text"
          value={courseCode}
          onChange={(e) => setCourseCode(e.target.value)}
          placeholder="Course Code"
          className="course-code-input"
        />
        <button onClick={addCourse} className="add-course-button">Add Course</button>
      </div>
      <table className="course-list-table">
        <thead>
          <tr>
            <th className="college-column">College</th>
            <th className="department-column">Department</th>
            <th className="course-name-column">Course Name</th>
            <th className="course-code-column">Course Code</th>
          </tr>
        </thead>
        <tbody>
          {courses.map((course, index) => (
            <tr key={index} className="course-row">
              <td className="college-cell">{course.college}</td>
              <td className="department-cell">{course.department}</td>
              <td className="course-name-cell">{course.name}</td>
              <td className="course-code-cell">{course.code}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </Box>
    </Box>
    </Box>
  );
};

export default CourseManagement;