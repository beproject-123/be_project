import React from "react";
import { Link, useLocation } from "react-router-dom";
import Logo from "../assets/headphone.png";

const Navbar = () => {
  const location = useLocation();

  return (
    <header className="flex justify-center py-4 px-8">
      <div className="flex items-center justify-between w-full max-w-4xl bg-black text-white px-6 py-3 rounded-full shadow-lg">
        {/* Logo */}
        <div className="text-2xl font-bold">
          <img src={Logo} alt="Detection Icon" className="w-6 h-6" />
        </div>

        {/* Navigation Links */}
        <nav className="flex space-x-6">
          {["Home", "About", "Product", "Contact"].map((page) => {
            const path = page === "Home" ? "/" : `/${page.toLowerCase()}`;
            return (
              <Link
                key={page}
                to={path}
                className={`px-4 py-2 rounded-full ${
                  location.pathname === path ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"
                }`}
              >
                {page}
              </Link>
            );
          })}
        </nav>

        {/* Buttons */}
        <div className="space-x-4">
          <Link
            to="/signup"
            className={`px-4 py-2 rounded-full ${
              location.pathname === "/signup"
                ? "bg-gray-800 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Sign Up
          </Link>

          <Link
            to="/login"
            className="bg-lime-400 text-black px-4 py-2 rounded-full hover:bg-lime-500 transition duration-300"
          >
            Login
          </Link>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
