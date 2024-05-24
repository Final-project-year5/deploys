import React, { useState } from "react";
import "./AdminHR.css";
import axios from "axios";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const AdminHR = () => {
  const [activeTab, setActiveTab] = useState("course");
  const [courses, setCourses] = useState([
    {
      id: 1,
      college: "College A",
      department: "Department A",
      name: "Database",
      code: "SWEG101",
    },
    {
      id: 2,
      college: "College B",
      department: "Department B",
      name: "Software evolution",
      code: "SWEG101",
    },
    {
      id: 3,
      college: "College C",
      department: "Department C",
      name: "Machine learning",
      code: "SWEG231",
    },
  ]);
  const [teachers, setTeachers] = useState([
    {
      id: 1,
      name: "Solomon Ashenafi",
      department: "Software",
      email: "Database",
      course: "SWEG101",
    },
    {
      id: 2,
      name: "Hana chanie",
      department: "Electrical",
      email: "mechanics",
      course: "SWEG101",
    },
    {
      id: 3,
      name: "Tsedey Mekonnen",
      department: "Software",
      email: "evolution",
      course: "SWEG101",
    },
  ]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  const addCourse = (course) => {
    setCourses([...courses, course]);
  };

  const deleteCourse = (id) => {
    if (window.confirm("Are you sure you want to delete this course?")) {
      setCourses(courses.filter((course) => course.id !== id));
    }
  };

  const editCourse = (id, updatedCourse) => {
    const updatedCourses = courses.map((course) =>
      course.id === id ? updatedCourse : course
    );
    setCourses(updatedCourses);
  };

  const addTeacher = (teacher) => {
    setTeachers([...teachers, teacher]);
  };

  const deleteTeacher = (index) => {
    if (window.confirm("Are you sure you want to delete this teacher?")) {
      setTeachers(teachers.filter((_, i) => i !== index));
    }
  };

  const editTeacher = (index, updatedTeacher) => {
    const updatedTeachers = teachers.map((teacher, i) =>
      i === index ? updatedTeacher : teacher
    );
    setTeachers(updatedTeachers);
  };

  return (
    <div className="admin-hr-page">
      <div className="flex justify-center items-center space-x-24 mb-4 ">
        <div
          className={`py-2 px-4 rounded-lg cursor-pointer ${
            activeTab === "course" ? "bg-blue-200" : "bg-blue-500"
          }`}
          onClick={() => handleTabChange("course")}
        >
          Course Management
        </div>
        <div
          className="w-4" // Adjust width for spacing
        ></div>{" "}
        {/* Spacer */}
        <div
          className={`py-2 px-4 rounded-lg cursor-pointer ${
            activeTab === "teacher" ? "bg-blue-200" : "bg-blue-500"
          }`}
          onClick={() => handleTabChange("teacher")}
        >
          Teacher Management
        </div>
        <div
          className="w-4" // Adjust width for spacing
        ></div>{" "}
        {/* Spacer */}
        <div
          className={`py-2 px-4 rounded-lg cursor-pointer ${
            activeTab === "schedule" ? "bg-blue-200" : "bg-blue-500"
          }`}
          onClick={() => handleTabChange("schedule")}
        >
          Schedule Management
        </div>
      </div>

      <div className="admin-hr-content">
        {activeTab === "course" ? (
          <CourseManagement
            courses={courses}
            addCourse={addCourse}
            deleteCourse={deleteCourse}
            editCourse={editCourse}
          />
        ) : activeTab === "teacher" ? (
          <TeacherManagement
            teachers={teachers}
            addTeacher={addTeacher}
            deleteTeacher={deleteTeacher}
            editTeacher={editTeacher}
            courses={courses}
          />
        ) : activeTab === "schedule" ? (
          <ScheduleManagement /> // Render the ScheduleManagement component
        ) : null}
      </div>
    </div>
  );
};

const CourseManagement = ({ courses, addCourse, deleteCourse, editCourse }) => {
  const [editingId, setEditingId] = useState(null);
  const [editedCollege, setEditedCollege] = useState("");
  const [editedDepartment, setEditedDepartment] = useState("");
  const [editedName, setEditedName] = useState("");
  const [editedCode, setEditedCode] = useState("");
  const [editedDuration, setEditedDuration] = useState("");
  const [editedYear, setEditedYear] = useState("");
  const [editedPrerequest, setEditedPrerequest] = useState("");

  const handleEditClick = (id) => {
    setEditingId(id);
    const courseToEdit = courses.find((course) => course.id === id);
    setEditedCollege(courseToEdit.college);
    setEditedDepartment(courseToEdit.department);
    setEditedName(courseToEdit.name);
    setEditedDuration(courseToEdit.code);
    setEditedYear(courseToEdit.code);
    setEditedPrerequest(courseToEdit.code);
  };

  const handleSaveEdit = (id) => {
    const updatedCourse = {
      id,
      college: editedCollege,
      department: editedDepartment,
      name: editedName,
      code: editedCode,
      duration: editedDuration,
      year: editedYear,
      prerequest: editedPrerequest,
    };
    editCourse(id, updatedCourse);
    setEditingId(null);
  };

  const handleAddCourse = () => {
    if (
      editedCollege.trim() !== "" &&
      editedDepartment.trim() !== "" &&
      editedName.trim() !== "" &&
      editedCode.trim() !== "" &&
      editedDuration.trim() !== "" &&
      editedYear.trim() !== "" &&
      editedPrerequest.trim() !== ""
    ) {
      const newCourse = {
        college: editedCollege,
        department: editedDepartment,
        name: editedName,
        code: editedCode,
        duration: editedDuration,
        year: editedYear,
        prerequest: editedPrerequest,
      };
      addCourse(newCourse);
      // Reset form fields after submission
      setEditedCollege("");
      setEditedDepartment("");
      setEditedName("");
      setEditedCode("");
      setEditedDuration("");
      setEditedYear("");
      setEditedPrerequest("");
    }
  };

  return (
    <div className="course-management">
      <h3>Add Course</h3>
      <div className="add-course-form">
        <input
          type="text"
          value={editedCollege}
          onChange={(e) => setEditedCollege(e.target.value)}
          placeholder="College"
          className="college-input"
        />
        <input
          type="text"
          value={editedDepartment}
          onChange={(e) => setEditedDepartment(e.target.value)}
          placeholder="Department"
          className="department-input"
        />
        <input
          type="text"
          value={editedName}
          onChange={(e) => setEditedName(e.target.value)}
          placeholder="Course Name"
          className="course-name-input"
        />
        <input
          type="text"
          value={editedCode}
          onChange={(e) => setEditedCode(e.target.value)}
          placeholder="Course Code"
          className="course-code-input"
        />
        <input
          type="text"
          value={editedDuration}
          onChange={(e) => setEditedDuration(e.target.value)}
          placeholder="Course Duration"
          className="course-duration-input"
        />
        <input
          type="text"
          value={editedYear}
          onChange={(e) => setEditedYear(e.target.value)}
          placeholder="Course Year"
          className="course-year-input"
        />
        <input
          type="text"
          value={editedPrerequest}
          onChange={(e) => setEditedPrerequest(e.target.value)}
          placeholder="Course Prerequest"
          className="course-prerequest-input"
        />
        <button onClick={handleAddCourse} className="add-course-button">
          Add Course
        </button>
      </div>
      <h3>List of Courses</h3>
      <table className="course-list-table">
        <thead>
          <tr>
            <th>College</th>
            <th>Department</th>
            <th>Course Name</th>
            <th>Course Code</th>
            <th>Duration</th>
            <th>Year</th>
            <th>Prerequest</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {courses.map((course) => (
            <tr key={course.id}>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedCollege}
                    onChange={(e) => setEditedCollege(e.target.value)}
                  />
                ) : (
                  course.college
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedDepartment}
                    onChange={(e) => setEditedDepartment(e.target.value)}
                  />
                ) : (
                  course.department
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedName}
                    onChange={(e) => setEditedName(e.target.value)}
                  />
                ) : (
                  course.name
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedCode}
                    onChange={(e) => setEditedCode(e.target.value)}
                  />
                ) : (
                  course.code
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedDuration}
                    onChange={(e) => setEditedDuration(e.target.value)}
                  />
                ) : (
                  course.duration
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedYear}
                    onChange={(e) => setEditedYear(e.target.value)}
                  />
                ) : (
                  course.year
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <input
                    type="text"
                    value={editedPrerequest}
                    onChange={(e) => setEditedPrerequest(e.target.value)}
                  />
                ) : (
                  course.prerequest
                )}
              </td>
              <td>
                {editingId === course.id ? (
                  <button onClick={() => handleSaveEdit(course.id)}>
                    Save
                  </button>
                ) : (
                  <>
                    <button
                      className="edit-button"
                      onClick={() => handleEditClick(course.id)}
                    >
                      Edit
                    </button>
                    <button
                      className="delete-button"
                      onClick={() => deleteCourse(course.id)}
                    >
                      Delete
                    </button>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
const TeacherManagement = ({
  teachers,
  addTeacher,
  deleteTeacher,
  editTeacher,
  courses,
}) => {
  const initialTeacherState = {
    name: "",
    email: "",
    department: "",
    semester: "", // Changed from selectedCourse
    phone_number: "",
    gender: "",
    college: "",
    qualifications: "",
  };

  const [teacherData, setTeacherData] = useState(initialTeacherState);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setTeacherData({
      ...teacherData,
      [name]: value,
    });
  };
  const handleAddTeacher = async () => {
    try {
      const formData = new FormData();
      formData.append("name", teacherData.name);
      formData.append("email", teacherData.email);
      formData.append("department", teacherData.department);
      formData.append("phone_number", teacherData.phone_number);
      formData.append("gender", teacherData.gender);
      formData.append("college", teacherData.college);
      formData.append("qualifications", teacherData.qualifications);
      formData.append("selectedCourse", teacherData.selectedCourse);

      // Append profile picture if selected
      if (teacherData.profilePicture) {
        formData.append("profile_picture", teacherData.profilePicture);
      }

      // Log form data before making the request
      console.log("Form Data:", formData);

      const response = await axios.post(
        "http://localhost:8000/api/add_teacher/",
        formData
      );

      console.log("Response from backend:", response.data);

      // Assuming addTeacher and setTeacherData are defined elsewhere
      addTeacher(response.data);
      setTeacherData(initialTeacherState);
    } catch (error) {
      console.error("Error adding teacher:", error.message);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error("Response status:", error.response.status);
        console.error("Response data:", error.response.data);
      } else if (error.request) {
        // The request was made but no response was received
        console.error("No response received:", error.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error("Error in request setup:", error.message);
      }
      // Handle error
    }
  };

  return (
    <div className="teacher-management">
      <h3>Add Teacher</h3>
      <div className="add-teacher-form">
        <input
          type="text"
          name="name"
          value={teacherData.name}
          onChange={handleChange}
          placeholder="Name"
          className="name-input"
        />
        <input
          type="email"
          name="email"
          value={teacherData.email}
          onChange={handleChange}
          placeholder="Email"
          className="email-input"
        />
        <input
          type="text"
          name="department"
          value={teacherData.department}
          onChange={handleChange}
          placeholder="Department"
          className="department-input"
        />
        <input
          type="text"
          name="semester" // Changed from selectedCourse
          value={teacherData.semester} // Changed from selectedCourse
          onChange={handleChange} // Changed from selectedCourse
          placeholder="Semester" // Changed from selectedCourse
          className="semester-input" // Changed from selectedCourse
        />
        <input
          type="text"
          name="phone_number"
          value={teacherData.phone_number}
          onChange={handleChange}
          placeholder="Phone Number"
          className="phone-number-input"
        />
        <input
          type="text"
          name="gender"
          value={teacherData.gender}
          onChange={handleChange}
          placeholder="Gender"
          className="gender-input"
        />
        <input
          type="text"
          name="college"
          value={teacherData.college}
          onChange={handleChange}
          placeholder="College"
          className="college-input"
        />
        <input
          type="text"
          name="qualifications"
          value={teacherData.qualifications}
          onChange={handleChange}
          placeholder="Qualifications"
          className="qualifications-input"
        />
        <button onClick={handleAddTeacher} className="add-teacher-button">
          Add Teacher
        </button>
      </div>
      <h3>List of Teachers</h3>
      <table className="teacher-list-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Email</th>
            <th>Department</th>
            <th>Semester</th>
            <th>Phone Number</th>
            <th>Gender</th>
            <th>College</th>
            <th>Qualifications</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {teachers.map((teacher, index) => (
            <tr key={index}>
              <td>{teacher.name}</td>
              <td>{teacher.email}</td>
              <td>{teacher.department}</td>
              <td>{teacher.semester}</td>
              <td>{teacher.phone_number}</td>
              <td>{teacher.gender}</td>
              <td>{teacher.college}</td>
              <td>{teacher.qualifications}</td>
              <td>
                <button
                  className="edit-button"
                  onClick={() => editTeacher(index)}
                >
                  Edit
                </button>
                <button
                  className="delete-button"
                  onClick={() => deleteTeacher(index)}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const ScheduleManagement = () => {
  // State for managing schedules
  const [schedules, setSchedules] = useState([]);
  const [file, setFile] = useState(null);

  // Function to handle file upload
  const handleFileUpload = (uploadedFile) => {
    if (uploadedFile) {
      if (validateFileType(uploadedFile)) {
        setFile(uploadedFile);
      } else {
        alert("Please select only Excel or CSV file.");
      }
    }
  };

  // Function to handle schedule upload
 const uploadSchedule = async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);

      // Make a POST request to upload the schedule
      const response = await axios.post(
        "http://localhost:8000/api/upload_schedule/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      // Handle success
      console.log("Response from backend:", response.data);
      // Update the schedules state with the response data
      setSchedules(response.data.schedules);

      // Reset the file state
      setFile(null);

      // Show success toast message
      toast.success("File uploaded successfully!", {
        position: toast.POSITION.TOP_CENTER,
      });
    } catch (error) {
      // Handle error
      console.error("Error uploading schedule:", error);

      // Show error toast message
      toast.error("Failed to upload file. Please try again later.", {
        position: toast.POSITION.TOP_CENTER,
      });
    }
  };

  // Function to validate file type
  const validateFileType = (file) => {
    const allowedExtensions = ["xlsx", "xls", "csv"];
    const fileType = file.name.split(".").pop();
    return allowedExtensions.includes(fileType);
  };

  // Function to handle drag over event
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Function to handle drop event
  const handleDrop = (event) => {
    event.preventDefault();
    const uploadedFile = event.dataTransfer.files[0];
    handleFileUpload(uploadedFile);
  };

  return (
    <div
      className="bg-gray-100 p-8 rounded-lg shadow-md text-center"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <div className="border-dashed border-2 border-gray-300 p-8 rounded-lg mb-4">
        <label
          htmlFor="fileInput"
          className="text-gray-500 text-lg block mb-2 cursor-pointer"
        >
          Drag & drop files to upload or click here
        </label>
        <input
          type="file"
          id="fileInput"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          className="hidden"
          accept=".csv,.xlsx,.xls"
        />
        <div
          className="flex items-center justify-center"
          onClick={() => document.getElementById("fileInput").click()}
        >
          <div className="bg-white border border-gray-300 p-4 rounded-lg cursor-pointer">
            Browse Files
          </div>
        </div>
        {file && (
          <div className="mt-4">
            <span className="text-gray-500">Selected file:</span>{" "}
            <span>{file.name}</span>
          </div>
        )}
      </div>
      <button
        onClick={uploadSchedule}
        className="bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition duration-300"
      >
        Upload Schedule
      </button>
      {/* Display schedules here */}
    </div>
  );
};

export default AdminHR;
