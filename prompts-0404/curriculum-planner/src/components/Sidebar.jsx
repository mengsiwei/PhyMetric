import React from "react";
import { getCourse } from "../data/catalog";

export default function Sidebar({ view, setView, progress, completedSemesters, totalSemesters, preCollegeCredits }) {
  const circumference = 2 * Math.PI * 42;
  const offset = circumference - (progress.overallPercent / 100) * circumference;

  return (
    <div className="sidebar">
      <div className="progress-ring-container">
        <div className="progress-ring">
          <svg width="100" height="100">
            <circle className="progress-ring-bg" cx="50" cy="50" r="42" />
            <circle
              className="progress-ring-fill"
              cx="50"
              cy="50"
              r="42"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="progress-ring-text">{progress.overallPercent}%</div>
        </div>
        <div className="progress-label">
          {progress.totalCreditsCompleted} / {progress.totalCreditsPlanned} credits
        </div>
        <div className="progress-label" style={{ marginTop: 4 }}>
          Semester {completedSemesters} of {totalSemesters}
        </div>
      </div>

      {preCollegeCredits && preCollegeCredits.length > 0 && (
        <div className="sidebar-section">
          <h3>Pre-College Credits</h3>
          <div style={{ fontSize: 11, display: "flex", flexDirection: "column", gap: 4 }}>
            {preCollegeCredits.map((id) => {
              const c = getCourse(id);
              return c ? (
                <div key={id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ color: "var(--accent-green)", fontSize: 10 }}>&#10003;</span>
                  <span style={{ color: "var(--text-secondary)" }}>{c.code}</span>
                </div>
              ) : null;
            })}
          </div>
        </div>
      )}

      <div className="sidebar-section">
        <h3>Views</h3>
        <div className={`nav-item ${view === "timeline" ? "active" : ""}`} onClick={() => setView("timeline")}>
          <span className="nav-icon">&#9776;</span>
          Semester Timeline
        </div>
        <div className={`nav-item ${view === "skilltree" ? "active" : ""}`} onClick={() => setView("skilltree")}>
          <span className="nav-icon">&#9733;</span>
          Skill Tree
        </div>
        <div className={`nav-item ${view === "progress" ? "active" : ""}`} onClick={() => setView("progress")}>
          <span className="nav-icon">&#9632;</span>
          Degree Progress
        </div>
      </div>

      <div className="sidebar-section">
        <h3>Legend</h3>
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="course-indicator indicator-core" /> CS Core
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="course-indicator indicator-math" /> Mathematics
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="course-indicator indicator-science" /> Science
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="course-indicator indicator-gened" /> General Ed
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="course-indicator indicator-elective" /> Tech Elective
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="course-indicator indicator-free" /> Free Elective
          </div>
        </div>
      </div>
    </div>
  );
}
