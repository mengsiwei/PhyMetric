import React, { useState } from "react";

export default function LoginPage({ onLogin }) {
  const [name, setName] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (name.trim()) onLogin(name.trim());
  };

  return (
    <div className="login-page">
      <div className="login-card fade-in">
        <div className="header-logo" style={{ margin: "0 auto 20px", width: 60, height: 60, fontSize: 24 }}>
          CP
        </div>
        <h1>
          UNR <span>Curriculum Planner</span>
        </h1>
        <p>Plan your Computer Science & Engineering degree path</p>
        <form className="login-form" onSubmit={handleSubmit}>
          <input
            className="login-input"
            type="text"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
          <button className="login-btn" type="submit" disabled={!name.trim()}>
            Start Planning
          </button>
        </form>
        <p style={{ marginTop: 16, fontSize: 12, color: "var(--text-muted)" }}>
          Your progress is saved locally in your browser.
        </p>
      </div>
    </div>
  );
}
