import React, { useState } from "react";
import { useHistory } from "react-router-dom";
import {
  Box,
  Button,
  Container,
  Divider,
  FormControl,
  FormLabel,
  Input,
  Select,
  useDisclosure,
  Portal,
} from "@chakra-ui/react";
import Sidebar from "../../../components/sidebar/Sidebar";
import { SidebarContext } from "../../../contexts/SidebarContext";
import Navbar from "components/navbar/NavbarAdmin.js";
import routes from "routes.js";

const EditSchedule = ({ sampleSchedule, setSampleSchedule }, props) => {
  const [courseName, setCourseName] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [selectedSection, setSelectedSection] = useState("");
  const [selectedDay, setSelectedDay] = useState("");
  const history = useHistory();
  const { ...rest } = props;
  const [fixed] = useState(false);

  

  // Function to handle updating the schedule
  const handleEditSchedule = () => {
    // Update the schedule logic here
    // Redirect to view schedule page after updating
    history.push("/view-schedule"); // Assuming "/add-schedule" is the route path for the AddSchedule page
  };

  const handleCancel = () => {
    // Redirect to view schedule page without saving changes
    history.push("/view-schedule"); // Assuming "/add-schedule" is the route path for the AddSchedule page
  };

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
                brandText="Edit Schedule"
                {...rest}
              />
            </Box>
          </Portal>
          <Box mt={2} mb={2} component="main" className="flex-grow py-8">
            <Container maxWidth={false}>
              <Divider style={{ marginBottom: 20 }} />
              <Box mt={10} mx="auto" maxW="500px">
                <FormControl className="mt-24">
                  <Box mt={10} textAlign="center">
                    <h1 className="text-blue-600 font-bold text-2xl">
                      Edit Schedule
                    </h1>
                  </Box>
                  <FormLabel color="blue">Course Name</FormLabel>
                  <Input
                    value={courseName}
                    onChange={(e) => setCourseName(e.target.value)}
                  />
                </FormControl>
                <FormControl mt={4}>
                  <FormLabel color="blue">Section</FormLabel>
                  <Select
                    value={selectedSection}
                    onChange={(e) => setSelectedSection(e.target.value)}
                  >
                    <option value="A">Section A</option>
                    <option value="B">Section B</option>
                    <option value="C">Section C</option>
                    <option value="D">Section D</option>
                  </Select>
                </FormControl>
                <FormControl mt={4}>
                  <FormLabel color="blue">Time</FormLabel>
                  <Input
                    type="time"
                    value={selectedTime}
                    onChange={(e) => setSelectedTime(e.target.value)}
                  />
                </FormControl>
                <FormControl mt={4}>
                  <FormLabel color="blue">Day</FormLabel>
                  <Select
                    value={selectedDay}
                    onChange={(e) => setSelectedDay(e.target.value)}
                  >
                    <option value="Monday">Monday</option>
                    <option value="Tuesday">Tuesday</option>
                    <option value="Wednesday">Wednesday</option>
                    <option value="Thursday">Thursday</option>
                    <option value="Friday">Friday</option>
                  </Select>
                </FormControl>
                <Box mt={6}>
                  <Button
                    onClick={handleEditSchedule}
                    colorScheme="blue"
                    mr={4}
                    variant="outline"
                  >
                    Save Changes
                  </Button>
                  <Button
                    onClick={handleCancel}
                    variant="outline"
                    colorScheme="red"
                  >
                    Cancel
                  </Button>
                </Box>
              </Box>
            </Container>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default EditSchedule;
