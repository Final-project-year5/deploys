import React, { useState } from 'react';
import './PermissionPage.css';

function PermissionPage() {
  const [roles, setRoles] = useState([
    {
      id: 1,
      name: 'Admin',
      permissions: [
        { id: 'manageUsers', label: 'Manage Users', description: 'Add, delete, and modify user accounts', enabled: false },
        { id: 'manageStudents', label: 'Manage Students', description: 'Add, delete, and modify student records', enabled: false },
        { id: 'manageAttendance', label: 'Manage Attendance', description: 'View and manage student attendance data', enabled: false }
      ]
    },
    {
      id: 2,
      name: 'Teacher',
      permissions: [
        { id: 'manageStudents', label: 'Manage Students', description: 'View and manage student records', enabled: false },
        { id: 'manageAttendance', label: 'Manage Attendance', description: 'View and manage student attendance data', enabled: false }
      ]
    },
    {
      id: 3,
      name: 'Student',
      permissions: [
        { id: 'manageProfile', label: 'Manage Profile', description: 'Update personal profile information', enabled: false }
      ]
    },
  ]);

  const handlePermissionChange = (roleId, permissionId) => {
    setRoles(prevRoles =>
      prevRoles.map(role => {
        if (role.id === roleId) {
          return {
            ...role,
            permissions: role.permissions.map(permission => {
              if (permission.id === permissionId) {
                return {
                  ...permission,
                  enabled: !permission.enabled
                };
              }
              return permission;
            })
          };
        }
        return role;
      })
    );
  };

  return (
    <div className="permission-page">
      <h1>Role and Permission Management</h1>
      <div className="role-list">
        <table>
          <thead>
            <tr>
              <th>Role</th>
              <th>Permissions</th>
            </tr>
          </thead>
          <tbody>
            {roles.map(role => (
              <tr key={role.id}>
                <td>{role.name}</td>
                <td>
                  <ul>
                    {role.permissions.map(permission => (
                      <li key={permission.id}>
                        <label>
                          <input
                            type="checkbox"
                            checked={permission.enabled}
                            onChange={() => handlePermissionChange(role.id, permission.id)}
                          />
                          {permission.label} - {permission.description}
                        </label>
                      </li>
                    ))}
                  </ul>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default PermissionPage;