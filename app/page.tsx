"use client";
import React, { useState } from 'react';

export default function FootballApp() {
  // This part tracks our "Data Stabilization" status
  const [status, setStatus] = useState("Waiting for Report");
  const [score, setScore] = useState(0);

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', backgroundColor: '#0f172a', color: 'white', minHeight: '100vh' }}>
      <h1>⚽ Football Highlight Cutter</h1>
      
      {/* THE STABILIZATION INDICATOR */}
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
            backgroundColor: score > 80 ? '#22c55e' : '#eab308', 
            marginRight: '10px',
            boxShadow: score > 80 ? '0 0 10px #22c55e' : 'none'
          }} />
          <span>{status} ({score}%)</span>
        </div>
      </div>

      <div style={{ display: 'grid', gap: '20px' }}>
        <div>
          <label>Step 1: Paste Match Report</label>
          <textarea 
            placeholder="Ex: 12' Goal by Messi..." 
            style={{ width: '100%', height: '100px', marginTop: '10px', borderRadius: '8px', padding: '10px' }}
          />
        </div>
        <button 
          onClick={() => { setStatus("Syncing..."); setScore(95); }}
          style={{ padding: '10px', backgroundColor: '#2563eb', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer' }}
        >
          Analyze & Cut Clips
        </button>
      </div>
    </div>
  );
}
