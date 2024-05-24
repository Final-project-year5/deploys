import { Box, SimpleGrid, Text } from "@chakra-ui/react";
import CheckTable from "views/admin/activitylog/components/activitylog";
import {roleData} from "views/admin/role/variables/activityData";
import React from "react";

export default function ActivityLog({ roles }) {
  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
      {roles.length === 0 ? (
        <Text>No roles.</Text>
      ) : (
        <SimpleGrid
          mb="20px"
          columns={{ sm: 1, md: 2 }}
          spacing={{ base: "20px", xl: "20px" }}
        >
          <CheckTable tableData={roleData} />
        </SimpleGrid>
      )}
    </Box>
  );
}