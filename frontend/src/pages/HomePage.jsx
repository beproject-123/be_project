import React from "react";
import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import Footer from "../components/Footer";
import Product1 from "../assets/product1.png";
import Product2 from "../assets/product2.png";
import Product3 from "../assets/product3.png";

const HomePage = () => {
  return (
    <div className="bg-[#282828] text-gray-200 font-sans">
      <Navbar />
      <Hero />

      {/* Product Section */}
      <section id="product" className="py-16 bg-[#1a1a1a] text-white">
        <h2 className="text-3xl text-center font-bold mb-10">
          Our <span className="text-lime-400">Product</span>
        </h2>
        <div className="grid md:grid-cols-3 gap-12 px-8">
          {/* Card 1 */}
          <div className="text-center p-12">
            <div className="mb-4 flex justify-center">
              <img src={Product1} alt="Detection Icon" className="w-16 h-16" />
            </div>
            <h3 className="text-xl font-bold mb-2">Accurate Deepfake Detection</h3>
            <p className="text-gray-300">
              Our cutting-edge technology quickly analyzes audio files to determine their authenticity.
              Using advanced machine learning algorithms, we can differentiate between real and fake
              audio with remarkable precision, ensuring that you have the most reliable results for your analysis needs.
            </p>
          </div>

          {/* Card 2 */}
          <div className="text-center p-12">
            <div className="mb-4 flex justify-center">
              <img src={Product2} alt="Detection Icon" className="w-16 h-16" />
            </div>
            <h3 className="text-xl font-bold mb-2">User-Friendly Experience</h3>
            <p className="text-gray-300">
              Uploading and checking your audio files is a breeze with our intuitive platform. Simply
              drag and drop your audio, and our system will take care of the rest. With a sleek and
              minimalistic interface, you can focus on the results without any distractions.
            </p>
          </div>

          {/* Card 3 */}
          <div className="text-center p-12">
            <div className="mb-4 flex justify-center">
              <img src={Product3} alt="Detection Icon" className="w-16 h-16" />
            </div>
            <h3 className="text-xl font-bold mb-2">Why Trust Us</h3>
            <p className="text-gray-300">
              Built on the latest advancements in AI and audio analysis, our platform offers a
              seamless way to verify audio files. We prioritize user privacy and accuracy, making
              our service the ideal solution for journalists, researchers, and anyone looking to
              identify deepfake audio content.
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default HomePage;
