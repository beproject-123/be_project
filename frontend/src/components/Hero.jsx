import React from "react";
import { Link } from "react-router-dom";

const Hero = () => {
  return (
    <section className="flex flex-col items-center justify-center py-36 text-center bg-[#434343] text-white">
      <h1 className="text-4xl font-bold mb-4">
        Want to test your audio file is fake????
      </h1>
      <p className="text-xl text-gray-300 mb-6">
        Your one stop to check your deep-fake audio's...
      </p>

      {/* Button that directs to the Upload Section */}
      <Link
        to="/product#upload-section"
        className="bg-black px-4 py-2 text-white rounded-md text-sm hover:bg-gray-900 transition duration-300"
      >
        Click here to find out!!!
      </Link>
    </section>
  );
};

export default Hero;
