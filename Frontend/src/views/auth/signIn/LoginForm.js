import React, { useState, useRef, useEffect } from "react";
import loginImage from "./../../../assets/img/auth/class.jpg";
import { useHistory } from "react-router-dom";
import { FaEnvelope, FaLock, FaUser } from "react-icons/fa";
import movingImage from "../../../assets/img/auth/class.jpg";

const LoginForm = () => {
  const [loginMethod, setLoginMethod] = useState("");
  const [loginId, setLoginId] = useState("");
  const [loginIdError, setLoginIdError] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginPasswordError, setLoginPasswordError] = useState("");
  const [showCamera, setShowCamera] = useState(false);
  const [videoStream, setVideoStream] = useState(null);
  const videoRef = useRef(null);
  const history = useHistory();

  useEffect(() => {
    if (showCamera) {
      startVideo();
    } else {
      stopVideo();
    }
    return () => {
      stopVideo();
    };
  }, [showCamera]);

  const startVideo = async () => {
    try {
      const constraints = { video: true };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setVideoStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const stopVideo = () => {
    if (videoStream) {
      videoStream.getTracks().forEach((track) => {
        track.stop();
      });
      setVideoStream(null);
    }
  };

  const handleLoginMethodChange = (method) => {
    setLoginMethod(method);
    if (method === "faceScan") {
      setShowCamera(true);
    } else {
      setShowCamera(false);
    }
  };

  const handleLoginIdChange = (e) => {
    setLoginId(e.target.value);
  };

  const handleLoginPasswordChange = (e) => {
    setLoginPassword(e.target.value);
  };

  const validateLoginForm = () => {
    let isValid = true;

    if (loginMethod === "email") {
      if (!/^(ETS|Ets)\d{4}\/\d{2}$/.test(loginId.trim())) {
        setLoginIdError("The format should be ETS0451/12 or Ets0451/12");
        isValid = false;
      } else {
        setLoginIdError("");
      }
    }

    if (loginMethod === "email") {
      if (
        !/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[A-Za-z\d!@#$%^&*()_+]{8,}$/.test(
          loginPassword.trim()
        )
      ) {
        setLoginPasswordError(
          "Password must contain at least 8 characters, one uppercase, one lowercase, one number, and one special character."
        );
        isValid = false;
      } else {
        setLoginPasswordError("");
      }
    }

    return isValid;
  };

  const handleLoginSubmit = (e) => {
    e.preventDefault();

    if (!validateLoginForm()) {
      return;
    }

    if (
      loginMethod === "email" &&
      loginId.trim() !== "" &&
      loginPassword.trim() !== ""
    ) {
      setLoginMethod("");
      setLoginId("");
      setLoginPassword("");

      history.push(`/admin`);
    }
  };

  const startCamera = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
      })
      .catch((error) =>
        console.error("Error accessing media devices: ", error)
      );
  };

  const handleCameraToggle = () => {
    setShowCamera(true);
    startCamera();
  };

  const handlePictureChange = async () => {
    try {
      if (!videoRef.current || !videoRef.current.srcObject) {
        console.error("No video stream available.");
        return;
      }
      const mediaStream = videoRef.current.srcObject;
      const videoTrack = mediaStream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(videoTrack);

      const blob = await imageCapture.takePhoto();
      const file = new File([blob], "profile_picture.jpg", {
        type: "image/jpeg",
      });

      // You can set this file to state or use it as needed
    } catch (error) {
      console.error("Error capturing picture:", error);
    }
  };

  return (
<div className="registration-container flex min-h-screen w-full items-center justify-center bg-cover bg-no-repeat" style={{ backgroundImage: `url(${movingImage})` }}>
    <div className="flex flex-col justify-between h-full py-8">
        <div
          className="rounded-xl bg-gray-600 bg-opacity-50 px-16 w-96 py-10 shadow-lg backdrop-blur-md max-sm:px-8"
          style={{ width: "500px" }}
        >
          <div className="text-white">
            <div className="mb-8 flex flex-col items-center">
              <h1 className="mb-2 text-blue-500 text-2xl font-bold">Login</h1>
              <span className="text-gray-300">
                {loginMethod ? "Login here" : "Choose one"}
              </span>
            </div>
            <div className="mb-10 flex justify-between">
              <button
                className={`text-lg ${
                  loginMethod === "email"
                    ? "text-blue-500 font-bold px-3"
                    : "text-white px-3"
                }`}
                onClick={() => handleLoginMethodChange("email")}
              >
                Login with email
              </button>
              <button
                className={`text-lg ${
                  loginMethod === "faceScan"
                    ? "text-blue-500 font-bold px-3"
                    : "text-white px-3"
                }`}
                onClick={() => handleLoginMethodChange("faceScan")}
              >
                Login with face scan
              </button>
            </div>
            {loginMethod === "email" && (
              <form onSubmit={handleLoginSubmit}>
                <div className="mb-4 text-lg">
                  <div className="relative">
                    <input
                      type="text"
                      value={loginId}
                      onChange={handleLoginIdChange}
                      placeholder=" "
                      className="registration-input text-white pl-4 mb-4 block w-full px-0 mt-0 bg-transparent border-0 border-b-2 appearance-none focus:outline-none focus:ring-0 focus:border-black border-gray-200"
                    />
                    <label className="absolute top-0 left-0 -mt-2 ml-2 text-gray-400">
                      <FaEnvelope className="inline-block mr-2 mt-1" /> Enter
                      your email
                    </label>
                  </div>
                  {loginIdError && (
                    <p className="text-red-500 text-sm">{loginIdError}</p>
                  )}
                </div>
                <div className="mb-4 text-lg">
                  <div className="relative">
                    <input
                      type="password"
                      value={loginPassword}
                      onChange={handleLoginPasswordChange}
                      placeholder=" "
                      className="registration-input text-white pl-4 mb-4 block w-full px-0 mt-0 bg-transparent border-0 border-b-2 appearance-none focus:outline-none focus:ring-0 focus:border-black border-gray-200"
                    />
                    <label className="absolute top-0 left-0 -mt-2 ml-2 text-gray-400">
                      <FaLock className="inline-block mr-2 mt-1" /> Enter your
                      password
                    </label>
                  </div>
                  {loginPasswordError && (
                    <p className="text-red-500 text-sm">
                      {loginPasswordError}
                    </p>
                  )}
                </div>
                <div className="mb-4 text-lg flex flex-col ">
                  <span
                    className="text-blue-500 cursor-pointer mb-4 ml-56"
                    onClick={() => history.push("/forgotpwd")}
                  >
                    Forgot password?
                  </span>
                  <button
                    type="submit"
                    className="take-picture-button ml-12 mr-12 mt-4 rounded-lg bg-blue-500 text-white px-4 py-1 transition-colors duration-300 hover:bg-transparent hover:text-blue-500"
                  >
                    Login
                  </button>
                </div>
              </form>
            )}
            {loginMethod === "faceScan" && (
              <div className="mb-4 text-lg flex justify-center">
                {showCamera && (
                  <div className="camera-container mt-2">
                    <video ref={videoRef} className="camera-feed" autoPlay />
                    <button
                      type="button"
                      onClick={handlePictureChange}
                      className="take-picture-button ml-32 mt-4 rounded-lg bg-blue-500 text-white px-4 py-1 transition-colors duration-300 hover:bg-transparent hover:text-blue-500"
                    >
                      Take Picture
                    </button>
                  </div>
                )}
              </div>
            )}
            <p className="text-center font-bold text-blue-500">
              Don't have an account?{" "}
              <span
                className="cursor-pointer"
                onClick={() => history.push("/signup")}
              >
                Sign up here
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;
