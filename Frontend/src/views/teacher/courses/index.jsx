import { Box, SimpleGrid, Text } from "@chakra-ui/react";
import CheckTable from "views/teacher/courses/components/table";
import { tableData } from "views/teacher/courses/variables/tableData";
import React from "react";

export default function CoursesPage({ courses }) {
  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
      {courses.length === 0 ? (
        <Text>No courses assigned to you.</Text>
      ) : (
        <SimpleGrid
          mb="20px"
          columns={{ sm: 1, md: 2 }}
          spacing={{ base: "20px", xl: "20px" }}
        >
          <CheckTable tableData={tableData} />
        </SimpleGrid>
      )}
    </Box>
  );
}