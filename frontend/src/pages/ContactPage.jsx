import React from "react";
import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import Footer from "../components/Footer";
import {
  FaPhone,
  FaEnvelope,
  FaMapMarkerAlt,
  FaTwitter,
  FaInstagram,
  FaLinkedin,
} from "react-icons/fa";

const ContactPage = () => {
  return (
    <div className="bg-[#282828] text-gray-200 font-sans">
      <Navbar />
      <Hero />

      {/* Contact Section */}
      <section className="py-16 px-8">
        <h2 className="text-4xl font-bold text-center">Contact Us</h2>
        <p className="text-gray-400 text-center mt-2">
          Any questions or remarks? Just write us a message!
        </p>

        <div className="mt-12 h-[500px] flex flex-col md:flex-row bg-gray-900 rounded-xl overflow-hidden shadow-lg">
          {/* Left - Contact Info */}
          <div className="relative bg-lime-400 text-gray-900 p-8 md:w-1/3 flex flex-col justify-between rounded-xl shadow-xl overflow-hidden">
            <div>
              <h3 className="text-xl font-bold">Contact Information</h3>
              <p className="text-sm mt-1">Say something to start a live chat!</p>

              <div className="mt-6 space-y-4">
                <p className="flex items-center space-x-3">
                  <FaPhone /> <span>+91 60705-96244</span>
                </p>
                <p className="flex items-center space-x-3">
                  <FaEnvelope /> <span>beproject_p03@gmail.com</span>
                </p>
                <p className="flex items-center space-x-3">
                  <FaMapMarkerAlt /> <span>Pune, Maharashtra, India</span>
                </p>
              </div>
            </div>

            {/* Social Media Icons */}
            <div className="flex space-x-4 text-black mt-8">
              <FaInstagram className="text-2xl cursor-pointer hover:scale-110 transition" />
              <FaTwitter className="text-2xl cursor-pointer hover:scale-110 transition" />
              <FaLinkedin className="text-2xl cursor-pointer hover:scale-110 transition" />
            </div>

            {/* Circular Background Elements */}
            <div className="absolute bottom-[-30px] right-[-30px]">
              <div className="w-28 h-28 bg-gray-900 opacity-50 rounded-full absolute bottom-16 right-16 z-10"></div>
              <div className="w-44 h-44 bg-black opacity-90 rounded-full absolute -bottom-7 -right-7"></div>
            </div>
          </div>

          {/* Right - Contact Form */}
          <div className="p-8 md:w-2/3 bg-[#1a1a1a]">
            <form className="space-y-6">
              <div className="flex space-x-4">
                <input
                  type="text"
                  placeholder="First Name"
                  className="w-1/2 bg-transparent border-b-2 border-gray-500 focus:border-lime-400 outline-none text-white placeholder-gray-400 py-2"
                />
                <input
                  type="text"
                  placeholder="Last Name"
                  className="w-1/2 bg-transparent border-b-2 border-gray-500 focus:border-lime-400 outline-none text-white placeholder-gray-400 py-2"
                />
              </div>
              <div className="flex space-x-4">
                <input
                  type="email"
                  placeholder="Email"
                  className="w-1/2 bg-transparent border-b-2 border-gray-500 focus:border-lime-400 outline-none text-white placeholder-gray-400 py-2"
                />
                <input
                  type="text"
                  placeholder="Phone Number"
                  className="w-1/2 bg-transparent border-b-2 border-gray-500 focus:border-lime-400 outline-none text-white placeholder-gray-400 py-2"
                />
              </div>
              <textarea
                placeholder="Message"
                rows="4"
                className="w-full bg-transparent border-b-2 border-gray-500 focus:border-lime-400 outline-none text-white placeholder-gray-400 py-2 resize-none"
              ></textarea>
              <button className="bg-lime-400 text-gray-900 px-6 py-3 rounded-md font-bold w-full hover:bg-lime-500 transition">
                Send Message
              </button>
            </form>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default ContactPage;
