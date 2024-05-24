import React, { useState } from "react";
import {
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  Box,
} from "@chakra-ui/react";
import { studentData } from 'views/teacher/courses/variables/studentData'; 

export default function StudentTable() {

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
            <Th minW="50px" bg="blue.200">No</Th>
            <Th minW="150px" bg="blue.200">Student ID</Th>
            <Th minW="200px" bg="blue.200">Name</Th>
            <Th minW="100px" bg="blue.200">Age</Th>
            <Th minW="100px" bg="blue.200">Grade</Th>
            <Th minW="200px" bg="blue.200">Status</Th>
          </Tr>
        </Thead>
        <Tbody>
          {studentData.map((student, index) => (
            <Tr key={index}>
              <Td>{index + 1}</Td>
              <Td>{student.id}</Td>
              <Td>{student.name}</Td>
              <Td>{student.age}</Td>
              <Td>{student.grade}</Td>
              <Td>{student.status}</Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Box>
  );
}