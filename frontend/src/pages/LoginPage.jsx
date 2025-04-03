import React from "react";
import { Link } from "react-router-dom";
import { FaFacebook, FaGoogle } from "react-icons/fa";

const Login = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-[#282828]">
      <div className="bg-white w-[400px] p-8 rounded-xl shadow-lg">
        <h2 className="text-center text-black text-2xl font-semibold">Log in</h2>
        <p className="text-center text-gray-500 text-sm mt-1">
          Don't have an account?{" "}
          <Link to="/signup" className="text-black font-semibold hover:underline">
            Sign up
          </Link>
        </p>
        
        <div className="mt-5 space-y-3">
          <button className="flex items-center justify-center w-full border border-gray-400 py-2 rounded-full text-gray-700">
            <FaFacebook className="text-blue-600 mr-2" /> Log in with Facebook
          </button>
          <button className="flex items-center justify-center w-full border border-gray-400 py-2 rounded-full text-gray-700">
            <FaGoogle className="text-red-500 mr-2" /> Log in with Google
          </button>
        </div>
        
        <div className="relative flex items-center my-5">
          <div className="flex-grow border-t border-gray-300"></div>
          <span className="mx-3 text-gray-400">OR</span>
          <div className="flex-grow border-t border-gray-300"></div>
        </div>

        <form className="space-y-4">
          <div>
            <label className="block text-sm text-gray-500">Your email</label>
            <input type="email" className="w-full text-black border-b border-gray-400 focus:outline-none py-1" />
          </div>
          <div className="relative">
            <label className="block text-sm text-gray-500">Your password</label>
            <input type="password" className="w-full text-black border-b border-gray-400 focus:outline-none py-1" />
          </div>
          <div className="text-right text-sm text-black font-medium cursor-pointer">
            Forget your password
          </div>
          <button className="w-full bg-gray-300 text-gray-500 py-2 rounded-full text-lg font-medium hover:bg-lime-400">
            Log in
          </button>
        </form>
        
        <div className="mt-6 text-sm text-gray-500 flex justify-between px-3">
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

export default Login;
