import React, { useState } from 'react';

export default function FootballApp() {
  // Tracking our "Data Stabilization" status
  const [status, setStatus] = useState("Waiting for Report");
  const [score, setScore] = useState(0);

  // This function simulates analyzing the report
  const handleAnalyze = () => {
    setStatus("Analyzing Kickoff...");
    setScore(45); // Yellow light
    
    // Simulate finding the kickoff 2 seconds later
    setTimeout(() => {
      setStatus("Stabilized & Synced");
      setScore(95); // Green light
    }, 2000);
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', backgroundColor: '#0f172a', color: 'white', minHeight: '100vh' }}>
      <h1>⚽ Football Highlight Cutter</h1>
      
      {/* DATA STABILIZATION INDICATOR */}
      <div style={{ 
        margin: '20px 0', 
        padding: '20px', 
        border: '1px solid #334155', 
        borderRadius: '12px', 
        backgroundColor: '#1e293b' 
      }}>
        <h3 style={{ margin: 0, fontSize: '12px', color: '#94a3b8', textTransform: 'uppercase' }}>
          Data Stabilization
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px' }}>
          <div style={{ 
            height: '15px', 
            width: '15px', 
            borderRadius: '50%', 
            backgroundColor: score > 80 ? '#22c55e' : (score > 0 ? '#eab308' : '#ef4444'), 
            marginRight: '10px',
            boxShadow: score > 80 ? '0 0 10px #22c55e' : 'none'
          }} />
          <span style={{ fontWeight: 'bold' }}>{status}</span>
          <span style={{ marginLeft: '10px', color: '#94a3b8' }}>{score}%</span>
        </div>
      </div>

      <div style={{ display: 'grid', gap: '20px', maxWidth: '600px' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '10px' }}>1. Paste Match Report</label>
          <textarea 
            placeholder="Example: 12' Goal by Messi..." 
            style={{ 
                width: '100%', 
                height: '150px', 
                borderRadius: '8px', 
                padding: '12px', 
                backgroundColor: '#0f172a', 
                color: 'white', 
                border: '1px solid #334155' 
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '10px' }}>2. Video Link (or Upload)</label>
          <input 
            type="text" 
            placeholder="Paste video URL here..."
            style={{ 
                width: '100%', 
                padding: '12px', 
                borderRadius: '8px', 
                backgroundColor: '#0f172a', 
                color: 'white', 
                border: '1px solid #334155' 
            }}
          />
        </div>

        <button 
          onClick={handleAnalyze}
          style={{ 
            padding: '15px', 
            backgroundColor: '#2563eb', 
            color: 'white', 
            border: 'none', 
            borderRadius: '8px', 
            fontWeight: 'bold', 
            cursor: 'pointer' 
          }}
        >
          Sync Video & Text
        </button>
      </div>
    </div>
  );
}
