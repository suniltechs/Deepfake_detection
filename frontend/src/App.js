import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import Results from './components/Results';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="App">
      <Navbar />
      <div className="container">
        <h1>Deepfake Detection</h1>
        <p>Upload an image or video to check if it's a deepfake</p>
        <FileUpload onResult={setResult} />
        <Results result={result} />
      </div>
    </div>
  );
}

export default App;