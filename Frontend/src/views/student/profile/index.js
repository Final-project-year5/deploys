import React, { useState } from 'react';
import avatars7 from './avatar7.png';
function ProfilePage() {
  const [fullName, setFullName] = useState('John Doe');
  const [profilePicture, setProfilePicture] = useState(avatars7);
 const [employeeID, setEmployeeID] = useState('EMP12345');
  const [department, setDepartment] = useState('Engineering');
  const [position, setPosition] = useState('Software Engineer');
  const [contactInfo, setContactInfo] = useState('john.doe@example.com');
  const [isEditMode, setIsEditMode] = useState(false);

  const handleEditFullName = (event) => {
    setFullName(event.target.value);
  };

  const handleEditEmployeeID = (event) => {
    setEmployeeID(event.target.value);
  };

  const handleEditDepartment = (event) => {
    setDepartment(event.target.value);
  };

  const handleEditPosition = (event) => {
    setPosition(event.target.value);
  };

  const handleEditContactInfo = (event) => {
    setContactInfo(event.target.value);
  };

  const handleEditClick = () => {
    setIsEditMode(true);
  };

  const handleSaveClick = () => {
    setIsEditMode(false);
  };

  return (
    <div
      style={{
     
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: '#f1f1f1',
       
      }}
    >
      <div
        style={{
          border: '1px solid #ccc',
          padding: '20px',
         
          borderRadius: '4px',
          background: '#fff',
          maxWidth: '400px',
           
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
         
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '20px',
            width:'400px' ,
            height: '200px' ,
          }}
        >
          <img
            src={profilePicture}
            alt="Profile Picture"
            style={{
              width: '100px',
              height: '100px',
              borderRadius: '50%',
              marginBottom: '10px',
            }}
          />
          <h2 style={{ margin: 0 }}>{fullName}</h2>
          <p style={{ color: '#888' }}>{position}</p>
        </div>
        <div style={{ marginBottom: '10px' }}>
          <label style={{ fontWeight: 'bold' }}>Employee ID:</label>
          <div>{employeeID}</div>
        </div>
        <div style={{ marginBottom: '10px' }}>
          <label style={{ fontWeight: 'bold' }}>Department:</label>
          <div>{department}</div>
        </div>
        <div style={{ marginBottom: '10px' }}>
          <label style={{ fontWeight: 'bold' }}>Contact Information:</label>
          <div>{contactInfo}</div>
        </div>
        {isEditMode ? (
          <>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ fontWeight: 'bold' }}>Full Name:</label>
              <input
                type="text"
                value={fullName}
                onChange={handleEditFullName}
                style={{ width: '100%', padding: '8px' }}
              />
            </div>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ fontWeight: 'bold' }}>Profile Picture:</label>
              <input
                type="file"
                accept="image/*"
                onChange={(event) =>
                  setProfilePicture(event.target.files[0])
                }
                style={{ width: '100%', padding: '8px' }}
              />
            </div>
          </>
        ) : null}
        {isEditMode ? (
          <button
            onClick={handleSaveClick}
            style={{
              width: '100%',
              padding: '8px',
              background: '#4caf50',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Save
          </button>
        ) : (
          <button
            onClick={handleEditClick}
            style={{
              width: '100%',
              padding: '8px',
              background: '#2196f3',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Edit
          </button>
        )}
      </div>
    </div>
  );
}

export default ProfilePage;