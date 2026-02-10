import React, { useState } from 'react';

export default function FootballApp() {
  const [status, setStatus] = useState("Waiting for Report");
  const [score, setScore] = useState(0);

  const handleAnalyze = () => {
    setStatus("Analyzing...");
    setScore(50);
    setTimeout(() => {
      setStatus("Stabilized & Synced");
      setScore(100);
    }, 2000);
  };

  return (
    <div style={{ padding: '40px', background: '#0f172a', color: 'white', minHeight: '100vh', fontFamily: 'sans-serif' }}>
      <h1>⚽ Football Cutter</h1>
      <div style={{ border: '1px solid #334155', padding: '20px', borderRadius: '10px', marginBottom: '20px' }}>
        <p style={{ fontSize: '12px', color: '#94a3b8' }}>DATA STABILIZATION</p>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ height: '12px', width: '12px', borderRadius: '50%', backgroundColor: score === 100 ? '#22c55e' : '#eab308', marginRight: '10px' }} />
          <span>{status} ({score}%)</span>
        </div>
      </div>
      <textarea placeholder="Paste report here..." style={{ width: '100%', height: '100px', marginBottom: '10px' }} />
      <button onClick={handleAnalyze} style={{ padding: '10px 20px', background: '#2563eb', color: 'white', border: 'none', borderRadius: '5px' }}>
        Sync Now
      </button>
    </div>
  );
}
