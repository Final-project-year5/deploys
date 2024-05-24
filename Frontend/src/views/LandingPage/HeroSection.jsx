import { Link } from 'react-router-dom'; // Import Link from react-router-dom
import attendanceVideo1 from "../../assets/img/video1.mp4";
import attendanceVideo2 from "../../assets/img/child.mp4";

const HeroSection = () => {
  return (
    <div className="flex flex-col items-center mt-6 lg:mt-20">
      <h1 className="text-4xl sm:text-6xl font-semibold lg:text-7xl text-center tracking-wide">
        Simplify Attendance Management
        <span className="bg-gradient-to-r from-blue-500 to-blue-800 text-transparent bg-clip-text">
          {" "}
          with Scan2mark
        </span>
      </h1>
      <p className="mt-10 text-lg text-center text-neutral-500 max-w-4xl">
        Streamline your attendance tracking process and enhance productivity
        with our comprehensive tools. Get started today and revolutionize
        attendance management!
      </p>
      <div className="flex justify-center my-10">
      <Link to="/signup" className="bg-gradient-to-r from-blue-500 to-blue-800 py-3 px-4 mx-3 rounded-md">

          Get started
        </Link>
       
      </div>
      <div className="flex mt-10 justify-center">
        <video
          autoPlay
          loop
          muted
          className="rounded-lg w-1/2 border border-blue-700 shadow-sm shadow-blue-400 mx-2 my-4"
        >
          <source src={attendanceVideo1} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <video
          autoPlay
          loop
          muted
          className="rounded-lg w-1/2 border border-blue-700 shadow-sm shadow-blue-400 mx-2 my-4"
        >
          <source src={attendanceVideo2} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
};

export default HeroSection;
