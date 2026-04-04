import React from "react";
import { TOTAL_CREDITS_REQUIRED } from "../data/catalog";

const fillClasses = ["fill-blue", "fill-purple", "fill-cyan", "fill-orange", "fill-green", "fill-gray"];

export default function ProgressDashboard({ progress, plan, completedSemesters }) {
  return (
    <div>
      {/* Overall stats */}
      <div className="progress-dashboard">
        <div className="progress-card">
          <h4>Overall Completion</h4>
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill fill-blue"
              style={{ width: `${progress.overallPercent}%` }}
            />
          </div>
          <div className="progress-stat">
            {progress.totalCompleted} / {progress.totalRequired} requirements
          </div>
        </div>

        <div className="progress-card">
          <h4>Credits Completed</h4>
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill fill-green"
              style={{
                width: `${Math.min(100, Math.round((progress.totalCreditsCompleted / TOTAL_CREDITS_REQUIRED) * 100))}%`,
              }}
            />
          </div>
          <div className="progress-stat">
            {progress.totalCreditsCompleted} / {TOTAL_CREDITS_REQUIRED} credits
          </div>
        </div>

        <div className="progress-card">
          <h4>Credits Planned</h4>
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill fill-cyan"
              style={{
                width: `${Math.min(100, Math.round((progress.totalCreditsPlanned / TOTAL_CREDITS_REQUIRED) * 100))}%`,
              }}
            />
          </div>
          <div className="progress-stat">
            {progress.totalCreditsPlanned} / {TOTAL_CREDITS_REQUIRED} credits planned
          </div>
        </div>

        <div className="progress-card">
          <h4>Semesters Remaining</h4>
          <div style={{ fontSize: 28, fontWeight: 700, color: "var(--accent-blue-light)" }}>
            {plan.length - completedSemesters}
          </div>
          <div className="progress-stat">
            of {plan.length} total semesters
          </div>
        </div>
      </div>

      {/* Per-group progress */}
      <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Requirement Groups</h3>
      <div className="progress-dashboard">
        {progress.groupProgress.map((group, i) => (
          <div className="progress-card" key={group.name}>
            <h4>{group.name}</h4>
            <div className="progress-bar-container">
              <div
                className={`progress-bar-fill ${fillClasses[i % fillClasses.length]}`}
                style={{ width: `${group.percent}%` }}
              />
            </div>
            <div className="progress-stat">
              {group.completed} / {group.total} completed ({group.percent}%)
            </div>
          </div>
        ))}
      </div>

      {/* Graduation estimate */}
      <div
        className="progress-card"
        style={{
          background: "linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(168, 85, 247, 0.1))",
          border: "1px solid var(--accent-blue)",
          marginTop: 16,
          maxWidth: 400,
        }}
      >
        <h4 style={{ color: "var(--accent-blue-light)" }}>Estimated Graduation</h4>
        <div style={{ fontSize: 20, fontWeight: 700, marginTop: 8 }}>
          {completedSemesters >= plan.length
            ? "Congratulations! All semesters complete!"
            : `${Math.ceil((plan.length - completedSemesters) / 2)} year${
                Math.ceil((plan.length - completedSemesters) / 2) > 1 ? "s" : ""
              } remaining`}
        </div>
        <div className="progress-stat" style={{ marginTop: 4 }}>
          {plan.length} semesters planned, {completedSemesters} completed
        </div>
      </div>
    </div>
  );
}
