import React, { useState } from "react";
import {
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useColorModeValue,
  Box,
  IconButton,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverArrow,
  PopoverCloseButton,
  PopoverBody,
} from "@chakra-ui/react";
import { MdMoreVert } from "react-icons/md";
import { useHistory } from "react-router-dom";

export default function CheckTable({ tableData }) {
  const textColor = useColorModeValue("secondaryGray.900", "white");
  const borderColor = useColorModeValue("gray.200", "whiteAlpha.100");
  const history = useHistory();

  const [selectedCourse, setSelectedCourse] = useState(null);

  const handleSelectCourse = (index) => {
    console.log("Handle select course function called with index:", index);
    const selectedCourse = tableData[index];
    // Use history to navigate to the view page
    history.push(`/courses/${index}`);
    console.log("after history push");
  };

  const handleCloseMenu = () => {
    setSelectedCourse(null);
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
      <Table variant="simple" color="gray.500" mb="24px">
        <Thead>
          <Tr>
            <Th minW="50px"></Th>
            <Th minW="50px" bg="blue.200">No</Th>
            <Th minW="150px" bg="blue.200">Course Code</Th>
            <Th minW="200px" bg="blue.200">Course Name</Th>
            <Th minW="100px" bg="blue.200">Duration</Th>
            <Th minW="100px" bg="blue.200">Year</Th>
            <Th minW="200px" bg="blue.200">Pre-request</Th>
          </Tr>
        </Thead>
        <Tbody>
          {tableData.map((course, index) => (
            <Tr key={index}>
              <Td>
              <Popover maxW="xs" maxH="xs">
                  <PopoverTrigger>
                    <IconButton
                      aria-label="More options"
                      icon={<MdMoreVert />}
                      variant="ghost"
                      size="sm"
                    />
                  </PopoverTrigger>
                  <PopoverContent>
                    <PopoverArrow/>
                    <PopoverCloseButton/>
                    <PopoverBody>
                      <Text
                        cursor="pointer"
                        onClick={() => handleSelectCourse(index)}
                      >
                        View
                      </Text>
                      <Text
                        cursor="pointer"
                      >
                        Schedule
                      </Text>
                    </PopoverBody>
                  </PopoverContent>
                </Popover>
              </Td>
              <Td>{index + 1}</Td>
              <Td>{course.code}</Td>
              <Td>{course.name}</Td>
              <Td>{course.duration}</Td>
              <Td>{course.year}</Td>
              <Td>{course.prerequest}</Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Box>
  );
}