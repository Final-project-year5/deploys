import React, { useState, useRef } from "react";
import { useParams } from "react-router-dom";
import {
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  Box,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
} from "@chakra-ui/react";
import { ChevronDownIcon } from "@chakra-ui/icons";
import Sidebar from "../../../../components/sidebar/Sidebar";
import { SidebarContext } from "../../../../contexts/SidebarContext";
import routes from "../../../../routes";
import AdminNavbar from "components/navbar/NavbarAdmin";

const ViewCoursePage = ({ tableData, students }) => {
  const { index } = useParams();

  // Ensure that index is a valid index
  const courseIndex = parseInt(index);
  const modalBodyRef = useRef(null);
  const [selectedSection, setSelectedSection] = useState(null);
  const [isCameraOpen, setCameraOpen] = useState(false);

  if (
    isNaN(courseIndex) ||
    courseIndex < 0 ||
    courseIndex >= tableData.length
  ) {
    return <div>Error: Invalid course index</div>;
  }

  const course = tableData[courseIndex];

  // Function to handle taking attendance
  const handleTakeAttendance = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setCameraOpen(true);

      // Display the camera feed in a video element
      const videoElement = document.createElement('video');
      videoElement.srcObject = stream;
      videoElement.autoplay = true;
      videoElement.play();

      // Append the video element to the modal body
      if (modalBodyRef.current) {
        modalBodyRef.current.appendChild(videoElement);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const handleCloseCamera = () => {
    setCameraOpen(false);
    // Remove the video element from the modal body
    if (modalBodyRef.current) {
      modalBodyRef.current.innerHTML = '';
    }
  };

  const handleFilterBySection = (section) => {
    setSelectedSection(section);
    // Implement filter logic here based on the selected section
    console.log("Filtering by section:", section);
  };

  return (
    <Box display="flex" position="relative">
      <AdminNavbar />
      <Box flex="1" ml="320px" mt="100px" mr="10">
        <Box mb="4">
          <Box justifyContent="space-between">
            <div style={{ position: 'absolute', top: '100px', right: '10px', zIndex: '999' }}>
            <Menu>
              <MenuButton
                as={Button}
                colorScheme="teal"
                marginLeft="650"
                rightIcon={<ChevronDownIcon />}
              >
                Filter by Section
              </MenuButton>
              <MenuList>
                {["A", "B", "C", "D"].map((section) => (
                  <MenuItem
                    key={section}
                    onClick={() => handleFilterBySection(section)}
                  >
                    {section}
                  </MenuItem>
                ))}
              </MenuList>
            </Menu>
            <Button colorScheme="blue" onClick={handleTakeAttendance}>
              Take Attendance
            </Button>
            </div>
          </Box>
          <Box mb="4">
          <Box display="flex" justifyContent="space-between" mb="2" mt="10">
              <Box>
                <h1>Course: {course.name}</h1>
                <p>Course Code: {course.code}</p>
              </Box>
              <Box>
                <p>Duration: {course.duration}</p>
                <p>Pre-requisite: {course.prerequisite}</p>
              </Box>
            </Box>
          </Box>
          <h2>List of Students</h2>
          <Box
            bg="white"
            borderRadius="md"
            boxShadow="md"
            p="4"
            maxW="xxl"
            mx="auto"
            overflowX="auto"
          >
            <Table variant="striped" colorScheme="gray" color="gray.500" mb="24px">
              <Thead>
                <Tr>
                  <Th bg="blue.200">
                    No
                  </Th>
                  <Th bg="blue.200">
                    First Name
                  </Th>
                  <Th bg="blue.200">
                    Last Name
                  </Th>
                  <Th bg="blue.200">
                    Student Id
                  </Th>
                  <Th bg="blue.200">
                    Section
                  </Th>
                </Tr>
              </Thead>
              <Tbody>
                {students.map((student, index) => (
                  <Tr key={index}>
                    <Td>{index + 1}</Td>
                    <Td>{student.fname}</Td>
                    <Td>{student.lname}</Td>
                    <Td>{student.studentId}</Td>
                    <Td>{student.section}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Box>
        </Box>
      </Box>
      <SidebarContext.Provider value={{ isOpen: true, toggleOpen: () => {} }}>
        <Sidebar routes={routes} />
      </SidebarContext.Provider>
      <Modal isOpen={isCameraOpen} onClose={handleCloseCamera}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Take Attendance</ModalHeader>
          <ModalCloseButton />
          <ModalBody ref={modalBodyRef}>
            {/* The ref attribute will assign the modal body reference to modalBodyRef */}
          </ModalBody>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default ViewCoursePage;