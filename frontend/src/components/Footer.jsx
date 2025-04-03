import React from "react";
import { Link } from "react-router-dom";
import { FaInstagram, FaTwitter, FaLinkedin } from "react-icons/fa";
import { MdEmail, MdLocationOn } from "react-icons/md";
import { FaPhoneAlt } from "react-icons/fa";
import Logo from "../assets/headphone.png";

const Footer = () => {
  return (
    <footer className="bg-black text-white py-8 px-6">
      <div className="flex justify-center mb-6">
        <img src={Logo} alt="Detection Icon" className="w-6 h-6" />
      </div>

      {/* Navigation Links - Styled like Navbar */}
      <div className="flex justify-center space-x-8 text-gray-300 font-medium text-lg">
        <Link to="/" className="hover:text-lime-400 transition duration-300">Home</Link>
        <Link to="/about" className="hover:text-lime-400 transition duration-300">About</Link>
        <Link to="/product" className="hover:text-lime-400 transition duration-300">Product</Link>
        <Link to="/contact" className="hover:text-lime-400 transition duration-300">Contact</Link>
      </div>

      <div className="mt-8"></div>

      {/* Contact Information in One Line */}
      <div className="flex justify-center items-center space-x-8 text-gray-400 mb-6">
        <div className="flex items-center space-x-2">
          <MdEmail className="text-lime-400 text-xl" />
          <span>beproject_p03@gmail.com</span>
        </div>
        <div className="flex items-center space-x-2">
          <FaPhoneAlt className="text-lime-400 text-xl" />
          <span>+91 60705-96244</span>
        </div>
        <div className="flex items-center space-x-2">
          <MdLocationOn className="text-lime-400 text-xl" />
          <span>Pune, India</span>
        </div>
      </div>

      <hr className="border-gray-700 mx-12" />

      {/* Social Media Icons, Centered Rights & Policies in One Row */}
      <div className="flex justify-between items-center px-12 mt-6">
        {/* Social Media Icons (Left) */}
        <div className="flex space-x-4">
          <a href="#" className="bg-lime-400 text-black p-3 rounded-full text-2xl">
            <FaInstagram />
          </a>
          <a href="#" className="bg-lime-400 text-black p-3 rounded-full text-2xl">
            <FaTwitter />
          </a>
          <a href="#" className="bg-lime-400 text-black p-3 rounded-full text-2xl">
            <FaLinkedin />
          </a>
        </div>

        {/* "All Rights Reserved" (Centered) */}
        <p className="text-gray-300 text-center flex-grow">All Rights Reserved</p>

        {/* Privacy Policy & Terms (Right) */}
        <div className="flex space-x-6 text-gray-300">
          <a href="#" className="hover:text-lime-400">Privacy Policy</a>
          <span>|</span>
          <a href="#" className="hover:text-lime-400">Terms of Service</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
