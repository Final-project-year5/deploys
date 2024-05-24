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
import { format, addDays } from "date-fns";
import Sidebar from "../../../components/sidebar/Sidebar";
import { SidebarContext } from "../../../contexts/SidebarContext";
import Navbar from "components/navbar/NavbarAdmin.js";
import routes from "routes.js";
import { FiChevronLeft } from 'react-icons/fi'; // Import ChevronLeftIcon from react-icons library

const AddSchedule = ({ sampleSchedule, setSampleSchedule }, props) => {
  const [courseName, setCourseName] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [selectedSection, setSelectedSection] = useState("");
  const [selectedDay, setSelectedDay] = useState("");
  const history = useHistory();
  const { ...rest } = props;
  const [fixed] = useState(false);

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  const handleAddSchedule = () => {
    const updatedSchedule = [...sampleSchedule];
    const scheduleIndex = updatedSchedule.findIndex(
      (schedule) => schedule.time === selectedTime
    );
    if (scheduleIndex !== -1) {
      updatedSchedule[scheduleIndex][selectedDay] = courseName;
      setSampleSchedule(updatedSchedule);
    }
    history.push(routes.viewSchedule);
  };

  const handleCancel = () => {
    history.push(routes.viewSchedule);
  };

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
                brandText="Add Schedule"
                leftElement={
                  <Button
                    variant="link"
                    onClick={() => history.goBack()} // Redirect to the previous page
                    leftIcon={<FiChevronLeft />} // Set ChevronLeftIcon as the left icon
                  >
                    Back
                  </Button>
                }
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
                      Add Schedule
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
                    onClick={handleAddSchedule}
                    colorScheme="blue"
                    mr={4}
                    variant="outline"
                  >
                    Add Schedule
                  </Button>
                  <Button
                    onClick={handleCancel}
                    variant="outline"
                    colorScheme="blue"
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

export default AddSchedule;
