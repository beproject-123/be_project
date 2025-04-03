import React from "react";
import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import Footer from "../components/Footer";
import AboutImage from "../assets/Image.png";

const AboutPage = () => {
  return (
    <div className="bg-[#282828] text-gray-200 min-h-screen font-sans">
      <Navbar />
      <Hero />

      <section className="text-white py-16 px-6 md:px-12 lg:px-24 flex flex-col md:flex-row items-center gap-12">
        {/* Left Content */}
        <div className="md:w-1/2 bg-[#1a1a1a] p-10 rounded-2xl shadow-lg relative z-10">
          <p className="text-sm text-gray-300 mb-2">Welcome</p>
          <h1 className="text-4xl md:text-5xl font-bold leading-tight">
            Where Authenticity <br />
            Meets <span className="text-lime-400">Precision!</span>
          </h1>
          <p className="text-gray-300 mt-4 leading-relaxed">
            We believe that audio verification should be more than just a process—
            it should be an empowering experience that ensures trust and authenticity.
            Our mission is to deliver reliable, state-of-the-art deepfake detection for audio,
            offering peace of mind to individuals, businesses, and organizations.
            With a commitment to accuracy, user-friendly design, and advanced technology,
            we aim to transform the way audio authenticity is verified. Join us in setting
            new standards for audio analysis and experience the assurance that comes
            with knowing the truth behind every sound.
          </p>
        </div>

        {/* Right Image Section */}
        <div className="md:w-1/2 relative -mb-50">
          <img
            src={AboutImage}
            alt="Business Professionals"
            className="rounded-xl"
          />
        </div>
      </section>

      <section className="py-16 px-8 max-w-6xl mx-auto">
        {/* Section Title */}
        <h2 className="text-3xl font-bold text-lime-400 mb-12">Mission & Vision</h2>

        <div className="flex flex-col gap-12">
          {/* Mission Section */}
          <div className="flex flex-col md:flex-row items-center p-6 rounded-2xl relative">
            {/* Image on Left */}
            <div className="w-full md:w-1/2 p-4">
              <img src={AboutImage} alt="Mission" className="rounded-lg w-full h-auto" />
            </div>

            {/* Text on Right */}
            <div className="w-full md:w-1/2 text-white p-6">
              <h3 className="text-2xl font-bold mb-4">
                <span className="border-l-4 border-lime-400 pl-3">Mission</span>
              </h3>
              <p className="text-gray-300 leading-relaxed">
                We aim to empower our users to ensure the authenticity of audio
                content with confidence. We are dedicated to delivering innovative
                deepfake detection solutions tailored to their needs. Our mission is
                to be a trusted partner in navigating the complexities of audio
                verification with accuracy and transparency.
              </p>
            </div>
          </div>

          {/* Vision Section */}
          <div className="flex flex-col md:flex-row-reverse items-center p-6 rounded-2xl relative">
            {/* Image on Right */}
            <div className="w-full md:w-1/2 p-4">
              <img src={AboutImage} alt="Vision" className="rounded-lg w-full h-auto" />
            </div>

            {/* Text on Left */}
            <div className="w-full md:w-1/2 text-white p-6">
              <h3 className="text-2xl font-bold mb-4">
                <span className="border-l-4 border-lime-400 pl-3">Vision</span>
              </h3>
              <p className="text-gray-300 leading-relaxed">
                Our vision is to redefine audio verification by providing a seamless
                and user-centric experience. We aim to set new standards in
                deepfake detection by ensuring accessibility, transparency, and
                cutting-edge technology to make audio authentication reliable for
                everyone.
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default AboutPage;
