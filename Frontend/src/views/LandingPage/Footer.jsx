import { resourcesLinks } from "./constants";
import { FaFacebook, FaTwitter, FaLinkedin, FaGithub } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="py-4 bg-[#D1D2F9]">
      <div className="container mx-auto px-4 grid grid-cols-2 lg:grid-cols-4 gap-8">
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2 text-black">Resources</h3>
          <ul className="space-y-2">
            {resourcesLinks.map((link, index) => (
              <li key={index}>
                <a
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors bg-transparent"
                >
                  {link.text}
                </a>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2 text-black">Connect with Us</h3>
          <ul className="flex space-x-4">
            <li>
              <a
                href="https://facebook.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                <FaFacebook size={24} />
              </a>
            </li>
            <li>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                <FaTwitter size={24} />
              </a>
            </li>
            <li>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                <FaLinkedin size={24} />
              </a>
            </li>
            <li>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                <FaGithub size={24} />
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div className="container mx-auto px-4 mt-0 text-center text-neutral-400 text-sm">
        &copy; {new Date().getFullYear()} XYZ Company. All rights reserved.
      </div>
    </footer>
  );
};

export default Footer;
