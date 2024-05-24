import React, { useState, useEffect } from "react";
import { useHistory } from "react-router-dom";
import axios from "axios"; // Import Axios for making HTTP requests
import {
  Box,
  Button,
  Container,
  Divider,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
} from "@chakra-ui/react";
import { Calendar, dateFnsLocalizer } from "react-big-calendar";
import { format, parse, startOfWeek, getDay, addDays } from "date-fns";
import { enUS } from "date-fns/locale";
import { Portal, useDisclosure } from "@chakra-ui/react";
import routes from "../../../routes.js";
import Navbar from "../../../components/navbar/NavbarAdmin.js";
import { DeleteIcon } from "@chakra-ui/icons";

const ViewSchedule = () => {
  const locales = {
    "en-US": enUS,
  };

  const localizer = dateFnsLocalizer({
    format,
    parse,
    startOfWeek,
    getDay,
    locales,
  });

  const [openSlot, setOpenSlot] = useState(false);
  const [openTodoModal, setOpenTodoModal] = useState(false);
  const [currentEvent, setCurrentEvent] = useState(null);
  const [eventInfoModal, setEventInfoModal] = useState(false);
  const [events, setEvents] = useState([]);
  const [scheduleData, setScheduleData] = useState([]);
  const [schedules, setSchedules] = useState([]); // Define setSchedules function here

  const history = useHistory();

  useEffect(() => {
    // Fetch schedule data
    const fetchScheduleData = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/api/display_schedule/");
        const data = response.data;
        if (data && data.schedules) {
          // Set the schedules state
          setSchedules(data.schedules);
          // Update the scheduleData state for rendering
          const updatedScheduleData = data.schedules.map(schedule => ({
            // Map the fetched schedule data to the format expected by your table
            time: `${schedule.start_time} - ${schedule.end_time}`, // Assuming you want to display the time range
            Monday: schedule.day_of_the_week === 'Monday' ? 'Available' : '', // Example for Monday, adjust as needed
            // Add more days as needed
          }));
          setScheduleData(updatedScheduleData);
        } else {
          console.error("Invalid data format:", data);
        }
      } catch (error) {
        console.error("Error fetching schedule data:", error);
      }
    };
  
    fetchScheduleData();
  }, []);
  
  

  const handleAddEventClick = () => {
    console.log("Add Event button clicked");
    history.push("/add-schedule");
  };

  const today = new Date();
  const dates = Array.from({ length: 5 }, (_, i) => addDays(today, i));
  const formattedDates = dates.map((date) => format(date, "EEE, MMM dd"));

  const handleScheduleItemClick = (event) => {
    history.push(`/edit-schedule/${event.id}`);
  };

  const getRandomColor = () => {
    const letters = "0123456789ABCDEF";
    let color = "#";
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return `${color}20`;
  };

  return (
    <Box mt={2} mb={2} component="main" className="flex-grow py-8">
      <Container maxWidth={false}>
        <Divider style={{ marginBottom: 20 }} />
        <Box className="flex justify-between mb-4 mt-20">
          <div className="flex items-center mb-4">
            <p className="text-sm font-bold text-gray-400 mr-4">
              Today {format(today, "EEE, MMM dd")}
            </p>
            <hr className="flex-grow border-gray-300" />
          </div>
          <Button
            onClick={handleAddEventClick}
            size="l"
            className="bg-blue-500 text-blue-900 hover:bg-blue-300 hover:text-blue-500 py-2 px-4 shadow-md font-semibold"
          >
            Add Event
          </Button>
        </Box>
        <Divider style={{ marginBottom: 20 }} />

        <Table variant="simple" color="black" borderColor="black">
          <Thead>
            <Tr>
              <Th borderColor="white">Time</Th>
              {formattedDates.map((date, index) => (
                <Th key={index} borderColor="black">
                  {date}
                </Th>
              ))}
            </Tr>
          </Thead>
          <Tbody>
            {scheduleData.map((row, index) => (
              <Tr
                key={index}
                className="mb-2"
                style={{
                  margin: "8px 0",
                  cursor: "pointer",
                  transition: "background-color 0.3s ease",
                  height: "80px",
                  borderRadius: "8px",
                  overflow: "hidden",
                }}
                onClick={() => handleScheduleItemClick(row)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "#F0F0F0";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "";
                }}
              >
                <Td>{row.time}</Td>
                {Object.keys(row)
                  .slice(1)
                  .map((day, index) => (
                    <Td
                      key={index}
                      style={{
                        backgroundColor: row[day]
                          ? getRandomColor()
                          : "#FFFFFF",
                        position: "relative",
                      }}
                    >
                      {row[day]}
                      {row[day] && (
                        <DeleteIcon
                          style={{
                            position: "absolute",
                            top: "4px",
                            right: "4px",
                            color: "rgba(255, 0, 0, 0.5)",
                            cursor: "pointer",
                            fontSize: "13px",
                          }}
                        />
                      )}
                    </Td>
                  ))}
              </Tr>
            ))}
          </Tbody>
        </Table>
      </Container>
    </Box>
  );
};

export default ViewSchedule;
