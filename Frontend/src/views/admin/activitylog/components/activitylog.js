import React, { useState } from "react";
import {
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  Box,
  Button,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
} from "@chakra-ui/react";
import DateRangePicker from "react-daterange-picker";
import "react-daterange-picker/dist/css/react-calendar.css";

export default function CheckTable({ activityData }) {
  const [dateRange, setDateRange] = useState();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleDateRangeChange = (newDateRange) => {
    setDateRange(newDateRange);
    // Implement filtering logic based on the selected date range
    console.log("Selected date range:", newDateRange);
  };

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen);
  };

  return (
    <Box
      bg="white"
      borderRadius="md"
      boxShadow="md"
      p="4"
      maxW="xxl"
      mx="auto"
      overflowX="auto"
    >
      <Box mb="4" mt="20">
        <Button onClick={toggleModal} bg="blue.100">
          {dateRange
            ? `${dateRange.start.format("YYYY-MM-DD")} to ${dateRange.end.format(
                "YYYY-MM-DD"
              )}`
            : "Filter By Date"}
        </Button>
        <Modal isOpen={isModalOpen} onClose={toggleModal}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Select Date Range</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <DateRangePicker
                value={dateRange}
                onSelect={handleDateRangeChange}
                singleDateRange={true}
              />
            </ModalBody>
            <ModalFooter>
              <Button colorScheme="blue" mr={3} onClick={toggleModal}>
                Close
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Box>
      <Table variant="simple" color="gray.500" mb="24px">
        <Thead>
          <Tr>
            <Th minW="50px" bg="blue.200">
              No
            </Th>
            <Th minW="100px" bg="blue.200">
              User
            </Th>
            <Th minW="100px" bg="blue.200">
              Action
            </Th>
            <Th minW="100px" bg="blue.200">
              Resource
            </Th>
            <Th minW="150px" bg="blue.200">
              Details
            </Th>
            <Th minW="150px" bg="blue.200">
              Ip Address
            </Th>
            <Th minW="150px" bg="blue.200">
              Status Code
            </Th>
            <Th minW="200px" bg="blue.200">
              Time Stamp
            </Th>
          </Tr>
        </Thead>
        <Tbody>
          {activityData &&
            activityData.map((activitylog, index) => (
              <Tr key={index}>
                <Td>{index + 1}</Td>
                <Td>{activitylog.user}</Td>
                <Td>{activitylog.action}</Td>
                <Td>{activitylog.resource}</Td>
                <Td>{activitylog.details}</Td>
                <Td>{activitylog.ipaddress}</Td>
                <Td>{activitylog.statuscode}</Td>
                <Td>{activitylog.timestamp}</Td>
              </Tr>
            ))}
        </Tbody>
      </Table>
    </Box>
  );
}