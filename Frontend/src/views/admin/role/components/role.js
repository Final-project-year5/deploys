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
  Flex,
} from "@chakra-ui/react";

export default function CheckTable({ roleData }) {
  // Placeholder function for handling adding a new role
  const handleAddRole = () => {
    // Implement your logic for adding a new role here
    console.log("Add new role functionality");
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
      <Flex justify="flex-end" mb="4" mt="14">
        {/* Button for adding a new role */}
        <Button onClick={handleAddRole} bg="blue.300" color="white" _hover={{ bg: "blue.500" }}>
            + New Role
        </Button>
      </Flex>
      <Table variant="simple" color="gray.500" mb="24px">
        <Thead>
          <Tr>
            <Th minW="50px" bg="blue.200">
              No
            </Th>
            <Th minW="100px" bg="blue.200">
              Name
            </Th>
            <Th minW="100px" bg="blue.200">
              Action
            </Th>
            <Th minW="100px" bg="blue.200">
              Permission
            </Th>
          </Tr>
        </Thead>
        <Tbody>
          {roleData.map((role, index) => (
            <Tr key={index}>
              <Td>{index + 1}</Td>
              <Td>{role.username}</Td>
              <Td>{role.action}</Td>
              <Td>{role.permission}</Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Box>
  );
}