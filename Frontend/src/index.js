import React from 'react';
import ReactDOM from 'react-dom';
import 'assets/css/App.css';
import { HashRouter, Route, Switch, Redirect } from 'react-router-dom';
import AuthLayout from 'layouts/auth';
import AdminLayout from 'layouts/admin';
import TeacherLayout from 'layouts/teacher';
import { ChakraProvider } from '@chakra-ui/react';
import theme from 'theme/theme';
import { ThemeEditorProvider } from '@hypertheme-editor/chakra-ui';
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import ViewCoursePage from "./views/teacher/courses/components/view";
import {tableData} from "views/teacher/courses/variables/tableData";
import {studentData} from "views/teacher/courses/variables/studentData";
import AttendanceHistory from "./views/teacher/Attendance Report/AttendanceHistory";
import DetailedReport from "./views/teacher/Attendance Report/DetailReport";
import StudentPermission from "./views/teacher/StudentPemission/StudentPermission";
import AddSchedule from "./views/teacher/Schedule/AddSchedule";
import EditSchedule from "./views/teacher/Schedule/EditSchedule";
import ViewSchedule from "./views/teacher/Schedule/ViewSchedule";
import ViewPermission from "./views/teacher/StudentPemission/ViewPermission";
import CourseManagement from './views/admin/AdminHR/CourseManagement';
import LandingPage from "./views/LandingPage/LandingPage";
import Register from "./views/auth/signIn/index";
import Login from "./views/auth/signIn/LoginForm";
import ForgotPwd from "./views/auth/signIn/ForgotPasswordProcess";



ReactDOM.render(
	<ChakraProvider theme={theme}>
		<React.StrictMode>
			<ThemeEditorProvider>
				<HashRouter>
					<Switch>
						<Route exact path="/" component={LandingPage} />
						<Route path={`/auth`} component={AuthLayout} />
						<Route path={`/admin`} component={AdminLayout} />
						<Route path={`/teacher`} component={TeacherLayout} />
						<Route path="/courses/:index" render={(props) => <ViewCoursePage {...props} tableData={tableData} students={studentData} />} />
						<Route path="/attendance-history" component={AttendanceHistory} />
						<Route path="/detailed-report/:id" component={DetailedReport} />
						<Route path="/student-permission" component={StudentPermission} />
						<Route path="/view-permission" component={ViewPermission} />
						<Route path="/course-management" component={CourseManagement} />
						<Route path="/view-schedule" component={ViewSchedule} />
						<Route path="/add-schedule" component={AddSchedule} />
						<Route path="/edit-schedule/:id" component={EditSchedule} />
						<Route path="/signup" component={Register} />
						<Route path="/signin" component={Login} />
						<Route path="/forgotpwd" component={ForgotPwd} />
						{/* <ToastContainer /> */}

						<Redirect from='/' to='/admin' />
					</Switch>
				</HashRouter>
			</ThemeEditorProvider>
		</React.StrictMode>
	</ChakraProvider>,
	document.getElementById('root')
);
