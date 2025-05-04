import React from 'react';
import { Link } from 'react-router-dom';
import './App.css'; // Make sure to import your CSS file

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="container">
        <Link to="/" className="navbar-brand">
          Deepfake Detector
        </Link>
        <div className="nav-links">
          <Link to="/" className="nav-link">Home</Link>
          <Link to="/about" className="nav-link">About</Link>
          <Link to="/how-it-works" className="nav-link">How It Works</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;