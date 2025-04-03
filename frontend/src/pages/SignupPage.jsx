import React, { useState } from "react";
import { Link } from "react-router-dom";
import Logo from "../assets/headphone.png";

const SignupPage = () => {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#282828]">
      <div className="bg-white p-8 rounded-lg shadow-lg w-[500px]">
        {/* Logo */}
        <div className="flex mb-3">
          <div className="bg-black w-12 h-12 flex items-center justify-center rounded-full">
            <img src={Logo} alt="Detection Icon" className="w-6 h-6" />
          </div>
        </div>

        {/* Title */}
        <h2 className="text-2xl font-semibold text-black">Create an account</h2>
        <p className="text-gray-600 mt-1">
          Already have an account?{" "}
          <Link to="/login" className="text-black font-semibold hover:underline">
            Log in
          </Link>
        </p>

        {/* Name Fields */}
        <div className="grid grid-cols-2 gap-4 mt-6">
          <div>
            <label className="text-gray-600">First name</label>
            <input
              type="text"
              className="w-full text-black border border-gray-300 rounded-md p-2 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-600">Last name</label>
            <input
              type="text"
              className="w-full text-black border border-gray-300 rounded-md p-2 focus:outline-none"
            />
          </div>
        </div>

        {/* Email Field */}
        <div className="mt-4">
          <label className="text-gray-600">Email address</label>
          <input
            type="email"
            className="w-full text-black border border-gray-300 rounded-md p-2 focus:outline-none"
          />
        </div>

        {/* Password Fields */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div>
            <label className="text-gray-600">Password</label>
            <input
              type={showPassword ? "text" : "password"}
              className="w-full text-black border border-gray-300 rounded-md p-2 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-600">Confirm your password</label>
            <input
              type={showPassword ? "text" : "password"}
              className="w-full text-black border border-gray-300 rounded-md p-2 focus:outline-none"
            />
          </div>
        </div>

        {/* Password Requirements */}
        <p className="text-gray-500 text-sm mt-2">
          Use 8 or more characters with a mix of letters, numbers & symbols
        </p>

        {/* Show Password Checkbox */}
        <div className="mt-2 flex items-center">
          <input
            type="checkbox"
            checked={showPassword}
            onChange={() => setShowPassword(!showPassword)}
            className="mr-2"
          />
          <span className="text-gray-700">Show password</span>
        </div>

        {/* Login Instead & Create Account (same row) */}
        <div className="mt-6 flex items-center justify-between">
          <Link to="/login" className="text-black font-semibold hover:underline text-sm">
            Log in instead
          </Link>
          <button className="bg-gray-300 text-gray-600 py-2 px-6 rounded-lg text-lg font-semibold hover:bg-lime-400">
            Create an account
          </button>
        </div>

        {/* Footer */}
        {/* Footer - Everything in One Line */}
        <div className="mt-6 flex items-center justify-between text-gray-600 text-sm">
          <span>English (United States) ▼</span>
          <div className="flex space-x-4">
            <a href="#" className="hover:underline">
              Help
            </a>
            <a href="#" className="hover:underline">
              Privacy
            </a>
            <a href="#" className="hover:underline">
              Terms
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
