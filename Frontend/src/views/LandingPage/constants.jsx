import { BotMessageSquare } from "lucide-react";
import { BatteryCharging } from "lucide-react";
import { Fingerprint } from "lucide-react";
import { ShieldHalf } from "lucide-react";
import { PlugZap } from "lucide-react";
import { GlobeLock } from "lucide-react";

import faceRecognitionIcon from "../../assets/img/facial-recognition.png";
import secureAccessIcon from "../../assets/img/folder.png";
import realTimeMonitoringIcon from "../../assets/img/data-analytics.png";
import automatedReportsIcon from "../../assets/img/report.png";
import integrationIcon from "../../assets/img/digital-transformation.png";
import userExperienceIcon from "../../assets/img/rating.png";

import user1 from "../../assets/img/profile-pictures/user1.jpg";
import user2 from "../../assets/img/profile-pictures/user2.jpg";
import user3 from "../../assets/img/profile-pictures/user3.jpg";
import user4 from "../../assets/img/profile-pictures/user4.jpg";
import user5 from "../../assets/img/profile-pictures/user5.jpg";
import user6 from "../../assets/img/profile-pictures/user6.jpg";

export const navItems = [
  { label: "Features", href: "#" },
  { label: "Workflow", href: "#" },
  { label: "Pricing", href: "#" },
  { label: "Testimonials", href: "#" },
];

export const aboutUs = [
 
  {
    text: "Our team is composed of talented and dedicated professionals who are experts in their fields. We work together to create innovative solutions that meet the unique needs of each client.",
    teamMembers: [
      {
        name: "John Doe",
        position: "CEO",
        image: user1,
      },
      {
        name: "Jane Smith",
        position: "CTO",
        image: user2,
      },
      {
        name: "David Johnson",
        position: "Lead Developer",
        image: user3,
      },
      {
        name: "Ronee Brown",
        position: "Project Manager",
        image: user4,
      },
      {
        name: "Michael Wilson",
        position: "Design Lead",
        image: user5,
      },
      {
        name: "Emily Davis",
        position: "Marketing Head",
        image: user6,
      },
    ],
  },
];


export const features = [
  {
    icon: <img src={faceRecognitionIcon} alt="Face Recognition" />,
    text: "Face Recognition Attendance Tracking",
    description:
      "Effortlessly track attendance using advanced face recognition technology, ensuring accurate and efficient attendance management.",
  },
  {
    icon: <img src={secureAccessIcon} alt="Secure Access" />,
    text: "Secure Access Control",
    description:
      "Enhance security with biometric access control, allowing only authorized individuals to enter designated areas.",
  },
  {
    icon: <img src={realTimeMonitoringIcon} alt="Real-Time Monitoring" />,
    text: "Real-Time Monitoring",
    description:
      "Monitor attendance in real-time with live updates and notifications, enabling proactive management and intervention.",
  },
  {
    icon: <img src={automatedReportsIcon} alt="Automated Reporting" />,
    text: "Automated Reporting",
    description:
      "Generate automated attendance reports with detailed insights and analytics, simplifying administrative tasks.",
  },
  {
    icon: <img src={integrationIcon} alt="Integration Capabilities" />,
    text: "Integration Capabilities",
    description:
      "Integrate seamlessly with existing systems and databases for streamlined attendance tracking and management.",
  },
  {
    icon: <img src={userExperienceIcon} alt="User-Friendly Interface" />,
    text: "User-Friendly Interface",
    description:
      "Enjoy a user-friendly interface designed for ease of use and intuitive navigation, ensuring a seamless experience for administrators and users alike.",
  },
];


export const checklistItems = [
  {
    title: "Automated Attendance",
    description:
      "Efficiently track attendance with facial recognition technology, reducing manual work.",
  },
  {
    title: "High Accuracy",
    description:
      "Advanced algorithms ensure accurate identification, minimizing errors in attendance records.",
  },
  {
    title: "Real-time Monitoring",
    description:
      "Monitor attendance in real-time with instant notifications and updates.",
  },
  {
    title: "Secure and Private",
    description:
      "Ensure data security and privacy with encrypted storage and strict access controls.",
  },
];

export const pricingOptions = [
  {
    title: "Free",
    price: "$0",
    features: [
      "Private board sharing",
      "5 Gb Storage",
      "Web Analytics",
      "Private Mode",
    ],
  },
  {
    title: "Pro",
    price: "$10",
    features: [
      "Private board sharing",
      "10 Gb Storage",
      "Web Analytics (Advance)",
      "Private Mode",
    ],
  },
  {
    title: "Enterprise",
    price: "$200",
    features: [
      "Private board sharing",
      "Unlimited Storage",
      "High Performance Network",
      "Private Mode",
    ],
  },
];

export const resourcesLinks = [
  { href: "#", text: "Getting Started" },
  { href: "#", text: "Documentation" },
  { href: "#", text: "Tutorials" },
  { href: "#", text: "API Reference" },
  { href: "#", text: "Community Forums" },
];

export const platformLinks = [
  { href: "#", text: "Features" },
  { href: "#", text: "Supported Devices" },
  { href: "#", text: "System Requirements" },
  { href: "#", text: "Downloads" },
  { href: "#", text: "Release Notes" },
];
