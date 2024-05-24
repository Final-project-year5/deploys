import React, { useState } from "react";
import "./PermissionPage.css"; // Import the CSS file

function PermissionPage() {
  // Sample list of teachers
  const teachers = ["Teacher 1", "Teacher 2", "Teacher 3"];

  // State variables to store form data and validation status
  const [teacher, setTeacher] = useState("");
  const [reason, setReason] = useState("");
  const [evidence, setEvidence] = useState(null); // Change to null initially
  const [sickLeave, setSickLeave] = useState(false);
  const [teacherError, setTeacherError] = useState("");
  const [reasonError, setReasonError] = useState("");
  const [fileSizeError, setFileSizeError] = useState("");

  // Function to handle form submission
  const handleSubmit = (event) => {
    event.preventDefault();
    // Perform form validation
    if (!teacher) {
      setTeacherError("Please select a teacher.");
      return;
    } else {
      setTeacherError("");
    }
    if (!reason) {
      setReasonError("Please provide a reason.");
      return;
    } else {
      setReasonError("");
    }
    if (!evidence) {
      setFileSizeError("Please select an evidence file.");
      return;
    } else if (evidence.size > 5242880) {
      // 5MB size limit
      setFileSizeError("File size exceeds the limit (5MB).");
      return;
    } else {
      setFileSizeError("");
    }
    // Perform submission logic here (e.g., send data to server)
    console.log("Teacher:", teacher);
    console.log("Reason:", reason);
    console.log("Evidence:", evidence);
    console.log("Sick leave requested:", sickLeave);
    // Reset form fields after submission
    setTeacher("");
    setReason("");
    setEvidence(null);
    setSickLeave(false);
  };

  // Function to handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setEvidence(file);
  };

  return (
    <div>
      <h2 className="title">Permission Form</h2>
      <div className="permission-form">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="teacher">Select Teacher:</label>
            <select
              id="teacher"
              value={teacher}
              onChange={(e) => setTeacher(e.target.value)}
              required
            >
              <option value="">Select Teacher</option>
              {teachers.map((teacher, index) => (
                <option key={index} value={teacher}>
                  {teacher}
                </option>
              ))}
            </select>
            {teacherError && <p className="error-message">{teacherError}</p>}
          </div>
          <div className="form-group">
            <label htmlFor="reason">Reason:</label>
            <input
              type="text"
              id="reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              required
            />
            {reasonError && <p className="error-message">{reasonError}</p>}
          </div>
          <div className="form-group">
            <label htmlFor="evidence">Evidence:</label>
            <input
              type="file"
              id="evidence"
              onChange={handleFileChange}
              accept=".jpg, .jpeg, .png, .pdf" // Limit accepted file types
            />
            {fileSizeError && <p className="error-message">{fileSizeError}</p>}
          </div>
          <div className="form-group">
            <input
              type="checkbox"
              id="sickLeave"
              checked={sickLeave}
              onChange={(e) => setSickLeave(e.target.checked)}
            />
            <label htmlFor="sickLeave">Request Sick Leave</label>
          </div>
          <button
            type="submit"
            className="bg-blue-500 hover:bg-blue-300 text-white font-bold py-2 px-2 rounded"
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}

export default PermissionPage;
