import { Box, SimpleGrid, Text } from "@chakra-ui/react";
import CheckTable from "views/admin/activitylog/components/activitylog";
import {activityData} from "views/admin/activitylog/variables/activityData";
import React from "react";

export default function ActivityLog({ activitylogs }) {
  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
      {activitylogs.length === 0 ? (
        <Text>No activities.</Text>
      ) : (
        <SimpleGrid
          mb="20px"
          columns={{ sm: 1, md: 2 }}
          spacing={{ base: "20px", xl: "20px" }}
        >
          <CheckTable tableData={activityData} />
        </SimpleGrid>
      )}
    </Box>
  );
}