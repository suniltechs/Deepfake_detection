import React from 'react';

const Results = ({ result }) => {
  if (!result) return null;

  return (
    <div className="results">
      <h2>Analysis Results</h2>
      <p>
        <strong>Type:</strong> {result.type}
      </p>
      <p>
        <strong>Result:</strong> 
        <span className={result.result.toLowerCase()}>
          {result.result} ({result.confidence}% confidence)
        </span>
      </p>
      {result.type === 'image' && (
        <div className="image-preview">
          <img 
            src={`http://localhost:5000/static/uploads/${result.filename}`} 
            alt="Uploaded content"
          />
        </div>
      )}
      {result.type === 'video' && (
        <div className="video-preview">
          <video controls>
            <source 
              src={`http://localhost:5000/static/uploads/${result.filename}`} 
              type={`video/${result.filename.split('.').pop()}`}
            />
          </video>
        </div>
      )}
    </div>
  );
};

export default Results;