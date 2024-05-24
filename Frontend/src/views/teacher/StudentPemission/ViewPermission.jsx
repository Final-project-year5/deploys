import React, { useState } from "react";
import { useHistory } from "react-router-dom";
import {
  Box,
  Button,
  useDisclosure,
  Container,
  Divider,
  Text,
} from "@chakra-ui/react";
import Sidebar from "../../../components/sidebar/Sidebar";
import { SidebarContext } from "../../../contexts/SidebarContext";
import Navbar from "components/navbar/NavbarAdmin.js";
import routes from "routes.js";

const ViewPermission = (
  {
    fullName = "John Doe",
    studentID = "123456",
    section = "A",
    reason = "Medical Leave",
    attachedFile = "medical_certificate.pdf",
  },
  props
) => {
  const history = useHistory();
  const { ...rest } = props;
  const [fixed] = useState(false);

  const today = new Date();
  const getRandomColor = (type) => {
    let color = "";
    if (type === "Present") {
      // Shades of green for Present
      color = `rgba(0, 128, 0, ${Math.random() * 0.3 + 0.2})`;
    } else if (type === "Absent") {
      // Shades of red for Absent
      color = `rgba(255, 69, 0, ${Math.random() * 0.3 + 0.2})`;
    } else {
      // Shades of blue for Permission
      color = `rgba(30, 144, 255, ${Math.random() * 0.3 + 0.2})`;
    }
    return color;
  };

  const approvePermission = () => {
    // Implement approve permission functionality
  };

  const rejectPermission = () => {
    // Implement reject permission functionality
  };

  const { onOpen } = useDisclosure();

  // Sample attendance data
  const attendanceHistory = [
    { date: "2024-04-01", status: "Present" },
    { date: "2024-04-02", status: "Absent" },
    { date: "2024-04-03", status: "Present" },
    { date: "2024-04-04", status: "Present" },
    { date: "2024-04-05", status: "Absent" },
  ];

  // Calculate number of absent, present, and permission days
  const absentDays = attendanceHistory.filter(
    (day) => day.status === "Absent"
  ).length;
  const presentDays = attendanceHistory.filter(
    (day) => day.status === "Present"
  ).length;
  const permissionDays = attendanceHistory.filter(
    (day) => day.status === "Permission"
  ).length;

  return (
    <Box>
      <SidebarContext.Provider value={{ isOpen: true, toggleOpen: () => {} }}>
        <Sidebar routes={routes} display="none" {...rest} />
      </SidebarContext.Provider>
      <Box className="ml-80">
        <Navbar onOpen={onOpen} brandText="Edit Schedule" {...rest} />
        <Container maxWidth={false}>
          <Divider style={{ marginBottom: 20 }} />
          <Box className="mt-2 mb-2 flex-grow py-8">
            <div className="container mx-auto">
              <div className="flex justify-between">
                <div className="grid grid-cols-3 mt-8 gap-4">
                  <div>
                    <h2 className="text-2xl mt-24 font-bold text-blue-600">
                      Student Permission
                    </h2>
                  </div>
                </div>
              </div>
              <div className="mt-4 mb-4 flex items-center">
                <hr className="flex-grow border-gray-300" />
              </div>
              <div className="mt-8 p-6 px-0 overflow-scroll">
                <div className="flex flex-wrap justify-between">
                  <div className="w-48">
                    <p className="text-lg font-semibold text-blue-400 mt-2">Full Name:</p>
                    <p className="text-lg font-bold text-blue-400 mt-4">Section:</p>
                    <p className="text-lg font-bold text-blue-400 mt-6">Attached File:</p>
                  </div>
                  <div className="flex-1">
                    <p className="text-lg  mb-4 bg-blue-100 text-gray-500 rounded-md w-56 pl-4 pr-4 pt-1 pb-1">
                      {fullName}
                    </p>
                    <p className="text-lg  mb-4 bg-blue-100 text-gray-500 rounded-md w-56 pl-4 pr-4 pt-1 pb-1">
                      {section}
                    </p>
                    <p className="text-lg  mb-4 bg-blue-100 text-gray-500 rounded-md w-56 pl-4 pr-4 pt-1 pb-1">
                      {attachedFile}
                    </p>
                  </div>
                  <div className="w-20">
                    <p className="text-lg font-bold text-blue-400 mt-2">ID:</p>
                    <p className="text-lg font-bold  text-blue-400 mt-6">Reason:</p>
                  </div>
                  <div className="flex-1">
                    <p className="text-lg  mb-4 text-gray-500 bg-blue-100 rounded-md w-56 pl-4 pr-4 pt-1 pb-1">{studentID}</p>
                    <p className="text-lg  text-gray-500 mb-4 bg-blue-100 rounded-md w-56 pl-4 pr-4 pt-1 pb-1">{reason}</p>

                  </div>
                </div>

                <div className="flex mt-4">
                  <Button
                    colorScheme="blue"
                    onClick={approvePermission}
                    size="md"
                    color="white"
                    className="ml-64"
                    _hover={{ bg: "white", color: "blue.500" }}
                  >
                    Approve
                  </Button>
                  <Button
                    colorScheme="red"
                    onClick={rejectPermission}
                    size="md"
                    className="ml-72"
                    color="white"
                    _hover={{ bg: "white", color: "red.500" }}
                  >
                    Reject
                  </Button>
                </div>

                <br />
                <hr className="mt-8" />
                <Text className="text-2xl font-bold mt-8 mb-8 text-blue-500"
                  
                >
                  Attendance History of {fullName}:
                </Text>
                <div className="flex flex-wrap gap-4">
                  <Box
                    bgColor={getRandomColor("Present")}
                    padding="30px"
                    borderRadius="md"
                    width="333px"
                  >
                    <Text fontSize="lg" fontWeight="bold" color="green">
                      Present Days:
                    </Text>
                    <Text>{presentDays}</Text>
                  </Box>
                  <Box
                    bgColor={getRandomColor("Absent")}
                    padding="30px"
                    borderRadius="md"
                    width="333px"
                  >
                    <Text fontSize="lg" fontWeight="bold" color="red">
                      Absent Days:
                    </Text>
                    <Text>{absentDays}</Text>
                  </Box>
                  <Box
                    bgColor={getRandomColor("Permission")}
                    padding="30px"
                    borderRadius="md"
                    width="333px"
                  >
                    <Text fontSize="lg" fontWeight="bold" color="blue">
                      Permission Days:
                    </Text>
                    <Text>{permissionDays}</Text>
                  </Box>
                </div>
              </div>
            </div>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default ViewPermission;
